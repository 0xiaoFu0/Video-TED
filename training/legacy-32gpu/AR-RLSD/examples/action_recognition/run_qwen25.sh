#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT_DIR="$PWD"
TRAIN_ENV_ROOT="${TRAIN_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}"
if [[ ! -f "$TRAIN_ENV_ROOT/bin/activate" ]]; then
  echo "找不到旧环境：$TRAIN_ENV_ROOT/bin/activate" >&2
  exit 1
fi
source "$TRAIN_ENV_ROOT/bin/activate"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MODEL_PATH="${MODEL_PATH:-/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct}"
export TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/train.rlsd.think.jsonl}"
export SFT_DATA="${SFT_DATA:-$PROJECT_DIR/data/train.sft.think.jsonl}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs}"
export HEAD_HOST="${HEAD_HOST:-192.0.2.12}"
export WORKER_HOST="${WORKER_HOST:-192.0.2.13}"
export WORKER_HOSTS="${WORKER_HOSTS:-$WORKER_HOST}"
if [[ "$WORKER_HOSTS" == ,* || "$WORKER_HOSTS" == *, || "$WORKER_HOSTS" == *,,* ]]; then
  echo "WORKER_HOSTS 必须是非空、逗号分隔的节点 IP。" >&2
  exit 2
fi
IFS=, read -r -a ar_worker_hosts <<< "$WORKER_HOSTS"
ar_hosts=("$HEAD_HOST" "${ar_worker_hosts[@]}")
for ((ar_index=0; ar_index<${#ar_hosts[@]}; ar_index++)); do
  ar_host="${ar_hosts[$ar_index]}"
  if [[ ! "$ar_host" =~ ^(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})$ ]]; then
    echo "节点 IP 无效：$ar_host" >&2
    exit 2
  fi
  IFS=. read -r -a ar_octets <<< "$ar_host"
  for ar_octet in "${ar_octets[@]}"; do
    if (( ar_octet > 255 )); then echo "节点 IP 无效：$ar_host" >&2; exit 2; fi
  done
  for ((ar_previous=0; ar_previous<ar_index; ar_previous++)); do
    if [[ "$ar_host" == "${ar_hosts[$ar_previous]}" ]]; then
      echo "节点地址必须互不相同：$ar_host" >&2
      exit 2
    fi
  done
done
# Keep the old single-worker variable usable, with the first configured worker.
export WORKER_HOST="${ar_worker_hosts[0]}"
export NODE_HOST="${NODE_HOST:-$WORKER_HOST}"
export NNODES="${NNODES:-${#ar_hosts[@]}}"
export N_GPUS="${N_GPUS:-8}"
if [[ ! "$NNODES" =~ ^[1-9][0-9]*$ || "$NNODES" != "${#ar_hosts[@]}" ]]; then
  echo "NNODES 必须等于 HEAD_HOST 加 WORKER_HOSTS 的节点总数。" >&2
  exit 2
fi
if [[ ! "$N_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_GPUS 必须是正整数。" >&2
  exit 2
fi
export RAY_PORT="${RAY_PORT:-6401}"
export AR_NCCL_PORT="${AR_NCCL_PORT:-29541}"
# Do not reuse Redis/GCS/session routing inherited from another Ray cluster.
# The local node IP is set after checking the requested node below.
unset MY_HOST_IPV6 RAY_REDIS_ADDRESS RAY_external_storage_namespace RAY_gcs_storage RAY_ADDRESS \
  RAY_NODE_IP_ADDRESS RAY_OVERRIDE_NODE_IP_ADDRESS
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_MODELSCOPE=0 USE_MODELSCOPE_HUB=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
for ar_nccl_var in NCCL_NET NCCL_NET_PLUGIN NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_SOCKET_IFNAME NCCL_SOCKET_FAMILY; do
  if [[ -z "${!ar_nccl_var:-}" ]]; then unset "$ar_nccl_var"; fi
done
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
# Internal Ray traffic must bypass any download proxy.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,$HEAD_HOST,$WORKER_HOSTS"
export no_proxy="$NO_PROXY"

mode="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

check_local_ip() {
  if [[ " $(hostname -I) " != *" $1 "* ]]; then
    echo "配置 IP $1 不属于当前节点；当前 IP：$(hostname -I)" >&2
    echo "请设置相同的 HEAD_HOST/WORKER_HOSTS，并设置当前工作节点的 NODE_HOST。" >&2
    exit 1
  fi
}

case "$mode" in
  check-video)
    CUDA_VISIBLE_DEVICES="${CHECK_GPU:-0}" python -m verl.ar.check_video \
      --model "$MODEL_PATH" --data "$TRAIN_DATA" --sft-data "$SFT_DATA" \
      --samples "${CHECK_SAMPLES:-1}" --backward "$@"
    ;;
  head)
    check_local_ip "$HEAD_HOST"
    export MY_HOST_IP="$HEAD_HOST" VLLM_HOST_IP="$HEAD_HOST"
    ar_ray_temp="$(mktemp -d /tmp/ar-action-32gpu-v3.XXXXXX)"
    echo "启动本机 Ray head：${MY_HOST_IP}:${RAY_PORT}；独立 session 目录：$ar_ray_temp"
    ray start --head --node-ip-address="$HEAD_HOST" --port="$RAY_PORT" \
      --num-gpus="$N_GPUS" --include-dashboard=false --disable-usage-stats \
      "$@" --temp-dir="$ar_ray_temp"
    ;;
  worker)
    ar_worker_member=0
    for ar_worker in "${ar_worker_hosts[@]}"; do
      if [[ "$NODE_HOST" == "$ar_worker" ]]; then ar_worker_member=1; fi
    done
    if [[ "$ar_worker_member" != 1 ]]; then
      echo "NODE_HOST 必须属于 WORKER_HOSTS：$NODE_HOST" >&2
      exit 2
    fi
    check_local_ip "$NODE_HOST"
    export MY_HOST_IP="$NODE_HOST" VLLM_HOST_IP="$NODE_HOST"
    echo "启动本机 Ray worker：${MY_HOST_IP}；连接 $HEAD_HOST:$RAY_PORT"
    ray start --address="$HEAD_HOST:$RAY_PORT" --node-ip-address="$NODE_HOST" \
      --num-gpus="$N_GPUS" --disable-usage-stats "$@"
    ;;
  cluster)
    export RAY_ADDRESS="$HEAD_HOST:$RAY_PORT"
    python -m verl.ar.check_cluster "$@"
    ;;
  smoke|train)
    check_local_ip "$HEAD_HOST"
    export MY_HOST_IP="$HEAD_HOST" VLLM_HOST_IP="$HEAD_HOST"
    export RAY_ADDRESS="$HEAD_HOST:$RAY_PORT"
    export RUN_NAME="${RUN_NAME:-ar_qwen25_mesd_ema010_ref010_${mode}_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$OUTPUT_ROOT/logs"
    log="$OUTPUT_ROOT/logs/$RUN_NAME.log"
    if [[ -e "$log" || -e "$OUTPUT_ROOT/$RUN_NAME" ]]; then
      echo "运行目录或日志已存在，请使用新的 RUN_NAME：$RUN_NAME" >&2
      exit 1
    fi
    python -m verl.ar.check_cluster 2>&1 | tee "$log"
    overrides=("trainer.nnodes=$NNODES" "trainer.n_gpus_per_node=$N_GPUS")
    if [[ "$mode" == smoke ]]; then
      overrides+=(trainer.max_steps=2 trainer.save_freq=-1 data.video_max_frames=8
                  data.max_response_length=512 worker.rollout.enforce_eager=true)
    fi
    # The new project aligns AR training with the MeSD EMA/schedule objective.
    # Keep the AR class/protocol reward and reference(.1) objective intact.
    # Keep these settings last so CLI overrides cannot silently turn it into
    # another experiment.
    fixed_alignment=(opsd.ar_alignment_mode=mesd
      opsd.ar_teacher_update_mode=ema opsd.ar_teacher_update_rate=0.01
      opsd.ar_teacher_update_interval=1 opsd.ar_teacher_disable_after_step=1000
      opsd.ar_teacher_max_prompt_length=18432
      opsd.rlsd_lambda=0.5 opsd.rlsd_lambda_warmup_steps=0 opsd.rlsd_lambda_decay_steps=1000
      data.truncation=left worker.actor.loss_avg_mode=seq
      opsd.ar_reward_mode=protocol opsd.ar_reference_coef=0.1
      'worker.reward.reward_function_kwargs={"reward_mode":"protocol"}')
    echo "固定 AR v3：alignment=mesd，teacher=EMA(rate=0.01, interval=1)，teacher prompt=18432；lambda=0.5，warmup=0，decay=1000，teacher disable-after=1000；left truncation、seq mean、protocol、reference_coef=0.1。以上字段在用户覆盖参数之后生效。" | tee -a "$log"
    bash examples/action_recognition/train.sh "${overrides[@]}" "$@" \
      "${fixed_alignment[@]}" 2>&1 | tee -a "$log"
    ;;
  *)
    echo "用法：bash examples/action_recognition/run_qwen25.sh {check-video|head|worker|cluster|smoke|train}"
    exit 2
    ;;
esac

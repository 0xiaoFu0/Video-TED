#!/usr/bin/env bash
# Four-node RLSD on the already exported action-understanding merge.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT_DIR="$PWD"
mode="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in prepare|head|worker|cluster|check-video|smoke|train) ;;
  *) echo '用法：bash examples/action_recognition/run_merged_32gpu.sh {prepare|head|worker|cluster|check-video|smoke|train} [训练覆盖参数]' >&2; exit 2 ;;
esac
: "${HEAD_HOST:?先 source examples/action_recognition/merged_32gpu.env}"
: "${WORKER_HOSTS:?先 source examples/action_recognition/merged_32gpu.env}"
export NNODES="${NNODES:-4}" N_GPUS="${N_GPUS:-8}"
export RAY_PORT="${RAY_PORT:-6401}" AR_NCCL_PORT="${AR_NCCL_PORT:-29541}"
if [[ "$NNODES" != 4 || "$N_GPUS" != 8 ]]; then
  echo '本入口使用四节点，每节点八卡。' >&2; exit 2
fi
TRAIN_ENV_ROOT="${TRAIN_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}"
if [[ ! -f "$TRAIN_ENV_ROOT/bin/activate" ]]; then echo "找不到环境：$TRAIN_ENV_ROOT" >&2; exit 1; fi
source "$TRAIN_ENV_ROOT/bin/activate"
export TRAIN_ENV_ROOT PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_MODELSCOPE_HUB=0 VLLM_USE_MODELSCOPE=0
export MODEL_PATH="${MODEL_PATH:-/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct}"
MERGED_DATA_ROOT="${MERGED_DATA_ROOT:-/path/to/workspace/zwj/AR-action-training-merged-v1}"
export TRAIN_DATA="$MERGED_DATA_ROOT/train.rlsd.think.jsonl"
export SFT_DATA="$MERGED_DATA_ROOT/train.sft.think.jsonl"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs}"
report="$PROJECT_DIR/work/merged_data_preflight/report.json"
check_args=(--data-root "$MERGED_DATA_ROOT" --model "$MODEL_PATH" --report "$report")
if [[ "$mode" == prepare ]]; then
  python -m verl.ar.check_merged_data "${check_args[@]}"
  exit
fi
# Validate the immutable dataset/tokenizer identity before any GPU job starts.
reference_length="$(python -m verl.ar.check_merged_data "${check_args[@]}" --check-report)"
if [[ ! "$reference_length" =~ ^[1-9][0-9]*$ ]]; then echo '参考长度检查返回异常。' >&2; exit 1; fi

if [[ "$mode" == check-video ]]; then
  export TRAIN_DATA="$PROJECT_DIR/work/merged_data_preflight/probe.rlsd.think.jsonl"
  export SFT_DATA="$PROJECT_DIR/work/merged_data_preflight/probe.sft.think.jsonl"
  export CHECK_SAMPLES=4
  exec bash examples/action_recognition/run_qwen25.sh check-video \
    --max-frames 16 --max-pixels 100352 --fps 2.0 --selection first "$@"
fi
if [[ "$mode" == smoke || "$mode" == train ]]; then
  export RUN_NAME="${RUN_NAME:-ar_qwen25_action93583_32gpu_mesd_ema010_ref010_${mode}_$(date +%Y%m%d_%H%M%S)_$$}"
  # Prompt batches are dispatched before the four student rollouts are repeated.
  # B=32 is divisible by all 32 ranks; G=32 x n=4 gives 4 trajectories/GPU.
  overrides=(data.rollout_batch_size=32 worker.actor.global_batch_size=32
    data.video_max_frames=16 data.max_response_length=768
    worker.rollout.enforce_eager=false worker.rollout.n=4
    worker.actor.optim.lr=1e-6 trainer.total_epochs=1
    "opsd.ar_reference_max_length=$reference_length"
    trainer.find_last_checkpoint=false trainer.load_checkpoint_path=null)
  if [[ "$mode" == train ]]; then
    # Larger data pool: reduce checkpoint I/O while retaining full resume state.
    overrides+=(trainer.save_freq=100 trainer.save_limit=5 trainer.save_model_only=false)
  fi
  echo "全量 AR v3：93583 条数据，4 节点 × 8 卡；MeSD 对齐、EMA rate=0.01、protocol、参考系数 0.1；参考长度 ${reference_length}。"
  # run_qwen25.sh appends the fixed MeSD alignment/reward settings after user
  # overrides. Other overrides, including explicit resume settings, are retained.
  exec bash examples/action_recognition/run_qwen25.sh "$mode" "${overrides[@]}" "$@"
fi
exec bash examples/action_recognition/run_qwen25.sh "$mode" "$@"

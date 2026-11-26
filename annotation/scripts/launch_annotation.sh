#!/usr/bin/env bash
# Single-node replicas or disjoint shards on multiple nodes with shared POSIX output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONDA_SH="${CONDA_SH:-/path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh}"
ANNOTATION_ENV_NAME="${ANNOTATION_ENV_NAME:-ar-qwen38-annotate}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-1}"
CANDIDATES="${CANDIDATES:-$PWD/work/candidates_v1/candidates.jsonl}"
CONFIG="${CONFIG:-$PWD/config.server.json}"
RUN_DIR="${RUN_DIR:-$PWD/work/full_parallel_v42}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-0}"
for annotation_integer in "$NODE_COUNT" "$NODE_RANK" "$GPUS_PER_WORKER" "$LIMIT_PER_DATASET"; do
  if [[ ! "$annotation_integer" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "Invalid nonnegative integer: $annotation_integer" >&2
    exit 2
  fi
done
if (( NODE_COUNT < 1 || NODE_RANK >= NODE_COUNT || GPUS_PER_WORKER < 1 )); then
  echo 'Invalid NODE_COUNT/NODE_RANK/GPUS_PER_WORKER.' >&2
  exit 2
fi
IFS=',' read -r -a annotation_gpus <<< "$GPU_IDS"
annotation_gpu_count=${#annotation_gpus[@]}
if (( annotation_gpu_count == 0 || annotation_gpu_count % GPUS_PER_WORKER != 0 )); then
  echo 'GPU_IDS count must be divisible by GPUS_PER_WORKER.' >&2
  exit 2
fi
for (( annotation_i=0; annotation_i<annotation_gpu_count; annotation_i++ )); do
  if [[ ! "${annotation_gpus[$annotation_i]}" =~ ^[0-9]+$ ]]; then
    echo 'GPU_IDS must be comma-separated physical GPU indices.' >&2
    exit 2
  fi
  for (( annotation_j=0; annotation_j<annotation_i; annotation_j++ )); do
    if [[ "${annotation_gpus[$annotation_i]}" == "${annotation_gpus[$annotation_j]}" ]]; then
      echo 'Duplicate GPU ID.' >&2; exit 2
    fi
  done
done
annotation_workers=$((annotation_gpu_count / GPUS_PER_WORKER))
annotation_shards=$((annotation_workers * NODE_COUNT))
annotation_pids=()
annotation_python=python
annotation_cleanup() {
  for annotation_pid in "${annotation_pids[@]}"; do kill "$annotation_pid" 2>/dev/null || true; done
  for annotation_pid in "${annotation_pids[@]}"; do wait "$annotation_pid" 2>/dev/null || true; done
}
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  source "$CONDA_SH"
  conda activate "$ANNOTATION_ENV_NAME"
  annotation_python="$CONDA_PREFIX/bin/python"
  [[ -f "$CANDIDATES" && -f "$CONFIG" ]] || { echo 'Missing candidates/config.' >&2; exit 2; }
  mkdir -p "$RUN_DIR/logs"
  trap 'annotation_cleanup; exit 130' INT TERM
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
for (( annotation_worker=0; annotation_worker<annotation_workers; annotation_worker++ )); do
  annotation_visible=''
  for (( annotation_offset=0; annotation_offset<GPUS_PER_WORKER; annotation_offset++ )); do
    annotation_pos=$((annotation_worker * GPUS_PER_WORKER + annotation_offset))
    annotation_visible="${annotation_visible:+$annotation_visible,}${annotation_gpus[$annotation_pos]}"
  done
  annotation_shard=$((NODE_RANK * annotation_workers + annotation_worker))
  annotation_cmd=("$annotation_python" -E -s -u -m ar_annotate annotate --config "$CONFIG" --candidates "$CANDIDATES"
    --out "$RUN_DIR" --num-shards "$annotation_shards" --shard-index "$annotation_shard"
    --limit-per-dataset "$LIMIT_PER_DATASET")
  if [[ "${RETRY_ERRORS:-0}" == 1 ]]; then annotation_cmd+=(--retry-errors); fi
  if [[ -n "${ROLLOUTS:-}" ]]; then annotation_cmd+=(--rollouts "$ROLLOUTS"); fi
  if [[ -n "${MIN_ACCEPTED_ROLLOUTS:-}" ]]; then annotation_cmd+=(--min-accepted-rollouts "$MIN_ACCEPTED_ROLLOUTS"); fi
  if [[ -n "${ANNOTATION_SEED:-}" ]]; then annotation_cmd+=(--seed "$ANNOTATION_SEED"); fi
  if [[ -n "${PROMPT_PROFILE:-}" ]]; then annotation_cmd+=(--prompt-profile "$PROMPT_PROFILE"); fi
  printf 'node=%s shard=%s/%s GPUs=%s\n' "$NODE_RANK" "$annotation_shard" "$annotation_shards" "$annotation_visible"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$annotation_visible"
    printf '%q ' "${annotation_cmd[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="$annotation_visible" "${annotation_cmd[@]}" \
      > "$RUN_DIR/logs/shard-$annotation_shard-$(date +%Y%m%d_%H%M%S)-$$.log" 2>&1 &
    annotation_pids+=("$!")
  fi
done
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo 'Dry run only; no environment activated and no workers launched.'
  exit 0
fi
annotation_failed=0
for annotation_pid in "${annotation_pids[@]}"; do
  if ! wait "$annotation_pid"; then annotation_failed=1; fi
done
if (( annotation_failed )); then
  echo 'At least one local worker failed. Inspect logs and records; other nodes have their own exit status.' >&2
  exit 1
fi
echo 'Local workers completed. Export checks all global shards before accepting the run as complete.'

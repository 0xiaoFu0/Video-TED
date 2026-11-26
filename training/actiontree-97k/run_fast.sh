#!/usr/bin/env bash
# Fast launch: identical 97k MeSD experiment, less per-step overhead on 32 GPUs.
#
#   bash run_fast.sh train qwen3-4b        # same stages/commands as run.sh
#   bash run_fast.sh smoke qwen3-8b
#   bash run_fast.sh logs  qwen3-4b
#
# The training contract is untouched: 32 prompts x n=4, global batch 32, lr,
# EMA/lambda schedules, reward, reference objective, 16 frames, 768 tokens.
# Only *how* each step is executed changes (see docs/FAST_TRAINING.md):
#   * the 4 trajectories a GPU owns per step are packed into one padding-free
#     forward instead of four (fewer FSDP all-gathers, fewer kernel launches);
#   * FSDP shards within a node (HSDP) so every all-gather stays on NVLink and
#     only the gradient reduce crosses the inter-node network.
# Both are pure execution changes: the loss is normalized over the global
# sequence count, so gradients match the default configuration up to float
# rounding.
#
# Tunables (environment variables):
#   FAST_HSDP=8    GPUs per FSDP shard group; 8 = one node (default), 0 = keep
#                  the original 32-way full sharding (lowest memory).
#   FAST_MICRO=4   sequences per GPU per forward; 4 = everything a GPU owns in
#                  this contract (32 prompts x 4 / 32 GPUs). Must divide 4.
#   FAST_VLLM_MEM= vLLM gpu_memory_utilization override (default: unchanged).
# NOTE: resuming a run with trainer.load_checkpoint_path must use the same
# FAST_HSDP value the checkpoint was written with.
set -euo pipefail
AR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NNODES_FIXED=4 GPUS_PER_NODE_FIXED=8 PROMPTS_FIXED=32 ROLLOUT_N_FIXED=4
WORLD=$((NNODES_FIXED * GPUS_PER_NODE_FIXED))
PER_GPU=$((PROMPTS_FIXED * ROLLOUT_N_FIXED / WORLD))   # 4 trajectories per GPU

FAST_HSDP="${FAST_HSDP:-$GPUS_PER_NODE_FIXED}"
FAST_MICRO="${FAST_MICRO:-$PER_GPU}"
FAST_VLLM_MEM="${FAST_VLLM_MEM:-}"

if [[ ! "$FAST_HSDP" =~ ^[0-9]+$ ]] || { (( FAST_HSDP > 0 )) && (( WORLD % FAST_HSDP != 0 )); }; then
  echo "FAST_HSDP must be 0 or a divisor of $WORLD (got '$FAST_HSDP')" >&2; exit 2
fi
if [[ ! "$FAST_MICRO" =~ ^[1-9][0-9]*$ ]] || (( PER_GPU % FAST_MICRO != 0 )); then
  echo "FAST_MICRO must divide the $PER_GPU trajectories each GPU owns (got '$FAST_MICRO')" >&2; exit 2
fi
if [[ -n "$FAST_VLLM_MEM" && ! "$FAST_VLLM_MEM" =~ ^0\.[1-9][0-9]*$ ]]; then
  echo "FAST_VLLM_MEM must be a fraction in (0, 1) such as 0.6 (got '$FAST_VLLM_MEM')" >&2; exit 2
fi

fast=(
  "worker.actor.micro_batch_size_per_device_for_update=$FAST_MICRO"
  "worker.actor.micro_batch_size_per_device_for_experience=$FAST_MICRO"
)
if (( FAST_HSDP > 0 && FAST_HSDP < WORLD )); then
  # Actor and EMA teacher must share one mesh (verl/ar/validation.py).
  fast+=("worker.actor.fsdp.fsdp_size=$FAST_HSDP" "worker.ref.fsdp.fsdp_size=$FAST_HSDP")
fi
if [[ -n "$FAST_VLLM_MEM" ]]; then
  fast+=("worker.rollout.gpu_memory_utilization=$FAST_VLLM_MEM")
fi

# Consumed by examples/action_recognition/run_97430.sh for smoke and train;
# run_model.sh still appends the fixed MeSD contract last, so nothing here can
# redefine the experiment. Every other run.sh command passes straight through.
export AR_FAST_OVERRIDES="${fast[*]}"
case "${1:-}" in
  train|smoke|check|plan)
    echo "run_fast: ${fast[*]}" >&2 ;;
esac
exec bash "$AR_ROOT/run.sh" "$@"

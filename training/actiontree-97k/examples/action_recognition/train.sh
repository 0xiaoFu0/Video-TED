#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL_PATH:?Set MODEL_PATH to the base or cold-start SFT checkpoint}"
TRAIN_DATA="${TRAIN_DATA:-$PWD/data/train.rlsd.think.jsonl}"
RUN_NAME="${RUN_NAME:-ar_think_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gemini-3/space/zwj/AR-RLSD/outputs}"
CHECKPOINT_PATH="$OUTPUT_ROOT/$RUN_NAME"
if [[ -e "$CHECKPOINT_PATH" ]]; then
  echo "Output exists; use a new RUN_NAME or an explicit resume config." >&2
  exit 1
fi
python -m verl.ar.prepare_data --check "$TRAIN_DATA" --check-media
# OMEGACONF CLI overrides are single arguments; avoid eval or embedded shell code.
python -m verl.trainer.opsd_main \
  config=examples/action_recognition/rlsd.yaml \
  "data.train_files=$TRAIN_DATA" \
  "worker.actor.model.model_path=$MODEL_PATH" \
  "trainer.experiment_name=$RUN_NAME" \
  "trainer.save_checkpoint_path=$CHECKPOINT_PATH" \
  "trainer.n_gpus_per_node=${N_GPUS:-8}" \
  "$@"

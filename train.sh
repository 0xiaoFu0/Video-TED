#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash train.sh MODEL_OR_HF_ID TRAIN_JSONL [key=value ...]"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 2 )); then
  usage >&2
  exit 2
fi

model=$1
train_data=$2
shift 2

if [[ -d "$model" ]]; then
  model="$(cd -- "$model" && pwd -P)"
elif [[ "$model" == /* || "$model" == ./* || "$model" == ../* ]]; then
  printf 'Model directory not found: %s\n' "$model" >&2
  exit 2
fi
if [[ ! -f "$train_data" ]]; then
  printf 'Training JSONL not found: %s\n' "$train_data" >&2
  exit 2
fi
train_data="$(cd -- "$(dirname -- "$train_data")" && pwd -P)/$(basename -- "$train_data")"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=1
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
run_name="videoted_$(date +%Y%m%d_%H%M%S)_$$"
exec python -m verl.trainer.opsd_main \
  config=configs/videoted.yaml \
  "trainer.experiment_name=$run_name" \
  "trainer.save_checkpoint_path=outputs/$run_name" \
  "$@" \
  "worker.actor.model.model_path=$model" \
  "data.train_files=$train_data"

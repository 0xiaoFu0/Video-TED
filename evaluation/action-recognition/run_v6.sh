#!/usr/bin/env bash
# Convenience launchers for the two models, sharing a new v6 results table.
set -euo pipefail
EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  base)
    EVAL_MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
    EVAL_MODEL_ROLE=base
    EVAL_MODEL_NAME=Qwen2.5-VL-7B-Instruct
    EVAL_RUN_NAME=qwen25_base_8gpu_v6
    ;;
  trained)
    EVAL_MODEL_PATH=/gemini-3/space/zwj/AR-RLSD/eval_models/ar_v3_025233_step292
    EVAL_MODEL_ROLE=ar-trained
    EVAL_MODEL_NAME=AR-RLSD-step292
    EVAL_RUN_NAME=ar_rlsd_step292_8gpu_v6
    ;;
  *)
    echo "Usage: bash run_v6.sh {base|trained} [run_8gpu.sh options, e.g. --resume]" >&2
    exit 2
    ;;
esac
shift
EVAL_V6_RESULTS_ROOT="${EVAL_V6_RESULTS_ROOT:-/gemini-3/space/zwj/VideoSTAR-Eval-results-v6}"
exec bash "$EVAL_SCRIPT_DIR/run_8gpu.sh" \
  --model "$EVAL_MODEL_PATH" \
  --model-name "$EVAL_MODEL_NAME" \
  --model-role "$EVAL_MODEL_ROLE" \
  --allow-train-overlap \
  --run-name "$EVAL_RUN_NAME" \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-root "$EVAL_V6_RESULTS_ROOT/runs" \
  --table-dir "$EVAL_V6_RESULTS_ROOT" \
  "$@"

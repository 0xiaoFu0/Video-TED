#!/usr/bin/env bash
# Separate v6 results because native Qwen3 video inputs have a distinct manifest.
set -euo pipefail
EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN3_RESULTS_ROOT="${QWEN3_RESULTS_ROOT:-/gemini-3/space/zwj/VideoSTAR-Eval-results-v6-qwen3}"
QWEN3_MODEL="${QWEN3_MODEL:-/path/to/workspace/models/Qwen/Qwen3-VL-8B-Instruct}"
mode="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  smoke) run_name=qwen3vl_8b_instruct_smoke_v6; extra=(--gpus 0 --limit 2) ;;
  run) run_name=qwen3vl_8b_instruct_8gpu_v6; extra=(--gpus 0,1,2,3,4,5,6,7) ;;
  check) run_name=qwen3vl_8b_instruct_8gpu_v6; extra=(--gpus 0,1,2,3,4,5,6,7 --prepare-only) ;;
  *) echo "Usage: bash run_qwen3.sh {check|smoke|run} [options, e.g. --resume]" >&2; exit 2 ;;
esac
exec bash "$EVAL_SCRIPT_DIR/run_8gpu.sh" \
  --model "$QWEN3_MODEL" --model-name Qwen3-VL-8B-Instruct --model-role base \
  --run-name "$run_name" --allow-train-overlap --batch-size 2 \
  --output-root "$QWEN3_RESULTS_ROOT/runs" --table-dir "$QWEN3_RESULTS_ROOT" \
  "${extra[@]}" "$@"

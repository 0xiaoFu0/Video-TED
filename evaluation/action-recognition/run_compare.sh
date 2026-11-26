#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${TUNED_MODEL:?请设置 TUNED_MODEL 为训练后导出的 Hugging Face 模型目录}"
BASE_MODEL="${BASE_MODEL:-/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct}"
COMPARE_NAME="${COMPARE_NAME:-qwen25_vs_ar_$(date +%Y%m%d_%H%M%S)}"
EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}/bin/python}"
"$EVAL_PYTHON" - "$BASE_MODEL" "$TUNED_MODEL" <<'PY'
import sys
from evaluate import model_identity
for model in sys.argv[1:]:
    model_identity(model)
PY
# Additional arguments (e.g. --benchmarks / --limit / --batch-size / --resume)
# apply identically to both models. Do not pass --model or --run-name here.
for arg in "$@"; do
  case "$arg" in
    --model|--model=*|--run-name|--run-name=*) echo "请用 BASE_MODEL/TUNED_MODEL/COMPARE_NAME 设置模型和名称。" >&2; exit 2;;
  esac
done
status=0
bash run_eval.sh run --model "$BASE_MODEL" --run-name "${COMPARE_NAME}_base" "$@" || status=$?
bash run_eval.sh run --model "$TUNED_MODEL" --run-name "${COMPARE_NAME}_trained" "$@" || status=$?
"$EVAL_PYTHON" compare_results.py --name "$COMPARE_NAME" "$@" || status=$?
exit "$status"

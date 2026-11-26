#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ $# -ne 2 ]]; then
  echo "用法：bash export_model.sh /完整路径/global_step_N/actor /独立导出目录" >&2
  exit 2
fi
EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}/bin/python}"
CUDA_VISIBLE_DEVICES="" "$EVAL_PYTHON" export_checkpoint.py "$1" "$2"

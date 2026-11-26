#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
EVAL_ENV_ROOT="${EVAL_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}"
EVAL_PYTHON="${EVAL_PYTHON:-$EVAL_ENV_ROOT/bin/python}"
if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "找不到 Python：$EVAL_PYTHON；可设置 EVAL_PYTHON=/实际环境/bin/python" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_MODELSCOPE=0 USE_MODELSCOPE_HUB=0
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$EVAL_PYTHON" parallel_eval.py "$@"

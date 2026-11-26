#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONDA_SH="${CONDA_SH:-/path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh}"
ANNOTATION_ENV_NAME="${ANNOTATION_ENV_NAME:-ar-qwen38-annotate}"
if [[ ! "$ANNOTATION_ENV_NAME" =~ ^[a-zA-Z0-9_-]+$ || "$ANNOTATION_ENV_NAME" == base ]]; then
  echo 'Choose a dedicated conda environment name (not base).' >&2
  exit 1
fi
source "$CONDA_SH"
if conda run -n "$ANNOTATION_ENV_NAME" python -c 'pass' >/dev/null 2>&1; then
  if [[ "${RESUME_INSTALL:-0}" != 1 ]]; then
    echo "Environment exists: $ANNOTATION_ENV_NAME. To resume ONLY this annotation install, use RESUME_INSTALL=1." >&2
    exit 1
  fi
else
  conda create -y -n "$ANNOTATION_ENV_NAME" python=3.11 pip
fi
conda activate "$ANNOTATION_ENV_NAME"
# Environment-local configuration: same requested keys, no edits to other environments.
unset PIP_CONFIG_FILE PIP_EXTRA_INDEX_URL
export PIP_INDEX_URL=http://pypi.chinatelecom.ai/simple
export PIP_TRUSTED_HOST=pypi.chinatelecom.ai
python -m pip config --site set global.index-url "$PIP_INDEX_URL"
python -m pip config --site set install.trusted-host "$PIP_TRUSTED_HOST"
python -m pip config --site set global.extra-index-url ''
python -m pip config --site set install.extra-index-url ''
# Every install uses the requested mirror; no fallback to public PyPI/CUDA wheel sites.
python -m pip install --upgrade pip --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
python -m pip install -r requirements.txt --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
python -m pip check
python -m pip freeze > "$CONDA_PREFIX/environment.freeze.txt"
conda env export --no-builds > "$CONDA_PREFIX/environment.conda.yml"
python -c 'import torch; print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda, "GPU count:", torch.cuda.device_count())'
printf 'Activate with: source %s && conda activate %s\n' "$CONDA_SH" "$ANNOTATION_ENV_NAME"

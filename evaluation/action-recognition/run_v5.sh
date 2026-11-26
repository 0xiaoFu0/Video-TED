#!/usr/bin/env bash
# Compatibility entry point after the v6 prompt update.
set -euo pipefail
EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Prompt protocol is v6; launching run_v6.sh with separate v6 results." >&2
exec bash "$EVAL_SCRIPT_DIR/run_v6.sh" "$@"

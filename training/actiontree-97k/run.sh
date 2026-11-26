#!/usr/bin/env bash
set -euo pipefail
AR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${AR_LAUNCH_PYTHON:-python3}" -B "$AR_ROOT/tools/launch.py" "$@"

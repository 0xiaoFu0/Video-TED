#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TRAIN_ENV_ROOT="${TRAIN_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}"
if [[ ! -x "$TRAIN_ENV_ROOT/bin/python" ]]; then
    echo "Python not found in the existing environment: $TRAIN_ENV_ROOT/bin/python"
    exit 1
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
"$TRAIN_ENV_ROOT/bin/python" -u - <<'CHECK_PY'
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import socket
import sys

errors = []
print('Node:', socket.gethostname())
print('Python:', sys.executable, sys.version.split()[0])
print('Project:', Path.cwd())
expected = {'transformers': '4.57.3', 'vllm': '0.11.0', 'qwen-vl-utils': '0.0.14'}
for name in ('torch', 'torchvision', 'transformers', 'vllm', 'ray', 'qwen-vl-utils',
             'decord', 'flash-attn', 'tensorboard', 'tensordict', 'torchdata',
             'datasets', 'omegaconf', 'peft', 'accelerate', 'liger-kernel'):
    try:
        version = metadata.version(name)
        suffix = ' [requirements.txt: ' + expected[name] + ']' if name in expected else ''
        print(name + ': ' + version + suffix)
        if name in expected and version != expected[name]:
            errors.append(name + ' version differs from requirements.txt; further review is required')
    except metadata.PackageNotFoundError:
        print(name + ': not installed')
        errors.append(name + ' not installed')

for name in ('torch', 'transformers', 'vllm', 'ray', 'qwen_vl_utils', 'flash_attn', 'tensorboard'):
    try:
        importlib.import_module(name)
        print('Import succeeded:', name)
    except Exception as exc:
        errors.append('Import ' + name + ': ' + repr(exc))
try:
    import torch
    print('PyTorch CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())
    count = torch.cuda.device_count()
    print('Visible GPUs:', count)
    for i in range(count):
        print('GPU', i, torch.cuda.get_device_name(i))
    if not torch.cuda.is_available() or count != 8:
        errors.append('The current process cannot see the expected 8 available GPUs')
except Exception as exc:
    errors.append('CUDA check: ' + repr(exc))

model = Path('/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct')
print('Model directory:', model)
try:
    cfg = json.loads((model / 'config.json').read_text())
    print('model_type:', cfg.get('model_type'))
    if cfg.get('model_type') != 'qwen2_5_vl':
        raise ValueError('Model type does not match Qwen2.5-VL')
    indices = list(model.glob('*.index.json'))
    for index in indices:
        weight_map = json.loads(index.read_text()).get('weight_map', {})
        missing = sorted({name for name in weight_map.values() if not (model / name).is_file()})
        if missing:
            raise FileNotFoundError('Missing weight shards: ' + str(missing))
    if not list(model.glob('*.safetensors')) and not list(model.glob('pytorch_model*.bin')):
        raise FileNotFoundError('No weight files found in the model directory')
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
    print('Processor:', type(processor).__name__)
except Exception as exc:
    errors.append('Model configuration/Processor check: ' + repr(exc))

try:
    import verl
    print('verl source:', verl.__file__)
    if Path(verl.__file__).resolve() != (Path.cwd() / 'verl/__init__.py').resolve():
        raise RuntimeError('The imported verl is not from the current AR-RLSD project')
    from verl.ar.prepare_data import read_rows, validate_rows
    for name in ('train.sft.jsonl', 'train.rlsd.jsonl'):
        rows = read_rows(Path('data') / name)
        print(name, json.dumps(validate_rows(rows, check_media=True), ensure_ascii=False))
except Exception as exc:
    errors.append('Annotation/video path check: ' + repr(exc))

if errors:
    print('\nItems requiring review:')
    for error in errors:
        print('-', error)
else:
    print('\nBasic checks passed.')
print('This check did not start training or validate video decoding, weight loading, model forward/backward passes, or communication across nodes.')
print('Qwen2.5-VL support is included in this update; next run the real-video/GPU checks described in UPLOAD_README.md.')
sys.exit(1 if errors else 0)
CHECK_PY

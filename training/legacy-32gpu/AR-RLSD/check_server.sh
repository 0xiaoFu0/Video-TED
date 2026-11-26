#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TRAIN_ENV_ROOT="${TRAIN_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}"
if [[ ! -x "$TRAIN_ENV_ROOT/bin/python" ]]; then
    echo "未找到旧环境 Python：$TRAIN_ENV_ROOT/bin/python"
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
print('节点:', socket.gethostname())
print('Python:', sys.executable, sys.version.split()[0])
print('工程:', Path.cwd())
expected = {'transformers': '4.57.3', 'vllm': '0.11.0', 'qwen-vl-utils': '0.0.14'}
for name in ('torch', 'torchvision', 'transformers', 'vllm', 'ray', 'qwen-vl-utils',
             'decord', 'flash-attn', 'tensorboard', 'tensordict', 'torchdata',
             'datasets', 'omegaconf', 'peft', 'accelerate', 'liger-kernel'):
    try:
        version = metadata.version(name)
        suffix = ' [requirements.txt: ' + expected[name] + ']' if name in expected else ''
        print(name + ': ' + version + suffix)
        if name in expected and version != expected[name]:
            errors.append(name + ' 版本与 requirements.txt 不一致，需进一步核对')
    except metadata.PackageNotFoundError:
        print(name + ': 未安装')
        errors.append(name + ' 未安装')

for name in ('torch', 'transformers', 'vllm', 'ray', 'qwen_vl_utils', 'flash_attn', 'tensorboard'):
    try:
        importlib.import_module(name)
        print('导入成功:', name)
    except Exception as exc:
        errors.append('导入 ' + name + ': ' + repr(exc))
try:
    import torch
    print('PyTorch CUDA:', torch.version.cuda, 'CUDA 可用:', torch.cuda.is_available())
    count = torch.cuda.device_count()
    print('可见 GPU 数:', count)
    for i in range(count):
        print('GPU', i, torch.cuda.get_device_name(i))
    if not torch.cuda.is_available() or count != 8:
        errors.append('当前进程未看到计划中的 8 张可用 GPU')
except Exception as exc:
    errors.append('CUDA 检查: ' + repr(exc))

model = Path('/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct')
print('模型目录:', model)
try:
    cfg = json.loads((model / 'config.json').read_text())
    print('model_type:', cfg.get('model_type'))
    if cfg.get('model_type') != 'qwen2_5_vl':
        raise ValueError('模型类型与 Qwen2.5-VL 不一致')
    indices = list(model.glob('*.index.json'))
    for index in indices:
        weight_map = json.loads(index.read_text()).get('weight_map', {})
        missing = sorted({name for name in weight_map.values() if not (model / name).is_file()})
        if missing:
            raise FileNotFoundError('缺少权重分片: ' + str(missing))
    if not list(model.glob('*.safetensors')) and not list(model.glob('pytorch_model*.bin')):
        raise FileNotFoundError('模型目录没有找到权重文件')
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
    print('Processor:', type(processor).__name__)
except Exception as exc:
    errors.append('模型配置/Processor 检查: ' + repr(exc))

try:
    import verl
    print('verl 来源:', verl.__file__)
    if Path(verl.__file__).resolve() != (Path.cwd() / 'verl/__init__.py').resolve():
        raise RuntimeError('导入的不是当前 AR-RLSD 的 verl')
    from verl.ar.prepare_data import read_rows, validate_rows
    for name in ('train.sft.jsonl', 'train.rlsd.jsonl'):
        rows = read_rows(Path('data') / name)
        print(name, json.dumps(validate_rows(rows, check_media=True), ensure_ascii=False))
except Exception as exc:
    errors.append('标注/视频路径检查: ' + repr(exc))

if errors:
    print('\n需要核对的项目:')
    for error in errors:
        print('-', error)
else:
    print('\n基础检查通过。')
print('本检查未启动训练，未验证视频解码、权重加载、模型前反向或跨节点通信。')
print('Qwen2.5-VL 适配已加入本更新包；下一步运行真实视频/GPU 检查，详见 UPLOAD_README.md。')
sys.exit(1 if errors else 0)
CHECK_PY

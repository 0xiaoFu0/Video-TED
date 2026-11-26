"""Exercise the real shell launcher with recording substitutes for Python/Ray."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    bindir = tmp_path / 'env/bin'
    bindir.mkdir(parents=True)
    (bindir / 'activate').write_text('export PATH="' + str(bindir) + ':$PATH"\n')
    recorder = '#!' + sys.executable + '''
import json, os, sys
with open(os.environ['AR_TEST_CALLS'], 'a') as f:
    f.write(json.dumps({'tool': sys.argv[0], 'args': sys.argv[1:], 'cuda': os.getenv('CUDA_VISIBLE_DEVICES')}) + '\\n')
sys.exit(7 if os.getenv('AR_TEST_FAIL_CLUSTER') == '1' and 'verl.ar.check_cluster' in sys.argv else 0)
'''
    for name in ('python', 'ray'):
        (bindir / name).write_text(recorder)
        (bindir / name).chmod(0o755)
    (bindir / 'hostname').write_text('#!/bin/sh\nprintf "%s\\n" "$AR_TEST_NODE_IP"\n')
    (bindir / 'hostname').chmod(0o755)
    env = dict(os.environ, TRAIN_ENV_ROOT=str(bindir.parent), OUTPUT_ROOT=str(tmp_path / 'outputs'),
               MODEL_PATH='/models/model with spaces', TRAIN_DATA='/data/train with spaces.jsonl',
               HEAD_HOST='192.0.2.12', WORKER_HOST='192.0.2.13',
               AR_TEST_NODE_IP='192.0.2.12', RUN_NAME='launch_test',
               AR_TEST_CALLS=str(tmp_path / 'calls.jsonl'))
    def run(mode, **overrides):
        result = subprocess.run(['bash', str(ROOT / 'examples/action_recognition/run_qwen25.sh'), mode],
            env={**env, **overrides}, capture_output=True, text=True, timeout=15)
        path = tmp_path / 'calls.jsonl'
        calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return result, calls
    return run


def test_smoke_launches_driver_once_and_preserves_paths(launcher):
    result, calls = launcher('smoke')
    assert result.returncode == 0, result.stderr
    drivers = [c for c in calls if 'verl.trainer.opsd_main' in c['args']]
    assert len(drivers) == 1
    args = drivers[0]['args']
    assert 'worker.actor.model.model_path=/models/model with spaces' in args
    assert 'data.train_files=/data/train with spaces.jsonl' in args
    assert 'trainer.nnodes=2' in args and 'trainer.max_steps=2' in args


def test_failed_cluster_check_blocks_training(launcher):
    result, calls = launcher('smoke', AR_TEST_FAIL_CLUSTER='1')
    assert result.returncode != 0
    assert not any('verl.trainer.opsd_main' in c['args'] for c in calls)


def test_worker_role_and_single_gpu_probe(launcher):
    result, calls = launcher('worker', AR_TEST_NODE_IP='192.0.2.13')
    assert result.returncode == 0, result.stderr
    assert '--address=192.0.2.12:6401' in calls[0]['args']
    result, calls = launcher('check-video')
    assert result.returncode == 0, result.stderr
    assert calls[-1]['cuda'] == '0'
    assert '--backward' in calls[-1]['args']


def test_wrong_node_cannot_start_head(launcher):
    result, calls = launcher('head', AR_TEST_NODE_IP='192.0.2.13')
    assert result.returncode != 0 and not calls


def test_default_launch_uses_think_data_for_training_and_video_probe(launcher):
    result, calls = launcher('train', TRAIN_DATA='', SFT_DATA='')
    assert result.returncode == 0, result.stderr
    args = next(c['args'] for c in calls if 'verl.trainer.opsd_main' in c['args'])
    assert f'data.train_files={ROOT}/data/train.rlsd.think.jsonl' in args
    result, calls = launcher('check-video', TRAIN_DATA='', SFT_DATA='')
    assert result.returncode == 0, result.stderr
    args = calls[-1]['args']
    assert args[args.index('--data') + 1] == str(ROOT / 'data/train.rlsd.think.jsonl')
    assert args[args.index('--sft-data') + 1] == str(ROOT / 'data/train.sft.think.jsonl')

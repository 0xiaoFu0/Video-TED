"""Validate configured Ray nodes, matching inputs/dependencies, and NCCL ranks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import sys


def cluster_topology(environment=None):
    """Resolve the old two-node defaults or an explicit worker-host list."""
    env = os.environ if environment is None else environment
    head = env.get('HEAD_HOST') or '192.0.2.12'
    workers = env.get('WORKER_HOSTS')
    if workers is None or workers == '':
        workers = env.get('WORKER_HOST') or '192.0.2.13'
    hosts = (head, *workers.split(','))
    for host in hosts:
        try:
            if str(ipaddress.IPv4Address(host)) != host:
                raise ValueError('Noncanonical IPv4 address')
        except ipaddress.AddressValueError as error:
            raise ValueError('Invalid configured node IPv4 address: ' + host) from error
    if len(set(hosts)) != len(hosts):
        raise ValueError('Configured node IPs must differ')
    values = {}
    for name, default in (('NNODES', len(hosts)), ('N_GPUS', 8)):
        value = env.get(name) or str(default)
        if not isinstance(value, str) or not value.isascii() or not value.isdigit() or value.startswith('0'):
            raise ValueError(name + ' must be a positive integer')
        values[name] = int(value)
    if values['NNODES'] != len(hosts):
        raise ValueError('NNODES must equal the configured head and worker-host count')
    return hosts, values['N_GPUS']


def network_environment(environment=None):
    """Explicitly carry cluster networking choices into remote Ray processes."""
    env = os.environ if environment is None else environment
    return {name: value for name, value in env.items()
            if value and (name in {'NO_PROXY', 'no_proxy'}
                          or name.startswith(('NCCL_', 'TORCH_NCCL_')))}


def sha256_file(path):
    """Hash large JSONL files without allocating their full payload in memory."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def fingerprint():
    import verl
    root = Path(__file__).resolve().parents[2]
    if Path(verl.__file__).resolve() != root / 'verl/__init__.py':
        raise RuntimeError('Wrong verl imported: ' + str(verl.__file__))
    code = hashlib.sha256()
    paths = sorted((root / 'verl').rglob('*.py'))
    paths += sorted((root / 'examples/action_recognition').glob('*'))
    for path in paths:
        if path.is_file():
            code.update(str(path.relative_to(root)).encode())
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    code.update(chunk)
    model = Path(os.environ['MODEL_PATH'])
    train = Path(os.environ['TRAIN_DATA'])
    if not train.is_file():
        raise FileNotFoundError(train)
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'transformers', 'vllm', 'ray', 'qwen-vl-utils', 'flash-attn')}
    sft = os.environ.get('SFT_DATA')
    asset_names = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                   'vocab.json', 'merges.txt', 'added_tokens.json', 'special_tokens_map.json',
                   'tokenizer.model', 'preprocessor_config.json', 'video_preprocessor_config.json',
                   'processor_config.json', 'chat_template.json', 'chat_template.jinja',
                   'model.safetensors.index.json')
    return {'root': str(root), 'python': list(sys.version_info[:2]), 'packages': packages,
            'code': code.hexdigest(), 'data': sha256_file(train),
            'sft_data': sha256_file(sft) if sft else None,
            'model_config': sha256_file(model / 'config.json'),
            'model_assets': {name: sha256_file(model / name) for name in asset_names if (model / name).is_file()}}


def nccl_sum(rank, world_size, address):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(0)  # Each Ray task is allocated exactly one visible GPU.
    try:
        dist.init_process_group('nccl', init_method=address, rank=rank,
                                world_size=world_size, timeout=timedelta(seconds=120))
        value = torch.tensor(float(rank + 1), device='cuda:0')
        dist.all_reduce(value)
        actual = value.item()
        expected = world_size * (world_size + 1) / 2
        if actual != expected:
            raise RuntimeError(f'NCCL rank {rank}: expected {expected}, got {actual}')
        return {'rank': rank, 'sum': actual}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-nccl', action='store_true')
    a = p.parse_args()
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    hosts, gpus_per_node = cluster_topology()
    head, world_size = hosts[0], len(hosts) * gpus_per_node
    runtime_vars = {name: os.environ[name] for name in
                    ('PYTHONPATH', 'MODEL_PATH', 'TRAIN_DATA', 'SFT_DATA') if name in os.environ}
    runtime_vars.update(network_environment())
    runtime = {'env_vars': runtime_vars}
    ray.init(address=os.environ['RAY_ADDRESS'], runtime_env=runtime)
    jobs = []
    try:
        nodes = [node for node in ray.nodes() if node['Alive']]
        if len(nodes) != len(hosts) or {n['NodeManagerAddress'] for n in nodes} != set(hosts):
            raise RuntimeError('Expected exactly the configured ' + str(len(hosts)) + ' Ray nodes; got ' + str(
                [(n['NodeManagerAddress'], n['Resources'].get('GPU', 0)) for n in nodes]))
        nodes.sort(key=lambda n: hosts.index(n['NodeManagerAddress']))
        for node in nodes:
            if node['Resources'].get('GPU', 0) != gpus_per_node:
                raise RuntimeError(f'Each node must advertise {gpus_per_node} GPUs: ' + node['NodeManagerAddress'])
        print('Hashing source, RLSD/SFT data and model configuration on each node.', flush=True)
        fingerprints = ray.get([
            ray.remote(num_cpus=1)(fingerprint).options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node['NodeID'], soft=False)).remote() for node in nodes], timeout=600)
        print(json.dumps(dict(zip((n['NodeManagerAddress'] for n in nodes), fingerprints)), indent=2), flush=True)
        if any(value != fingerprints[0] for value in fingerprints[1:]) or fingerprints[0] != fingerprint():
            raise RuntimeError('Code, RLSD/SFT data, model config, Python or package versions differ between nodes/driver')
        print(f'{len(hosts)} nodes, {world_size} GPUs; source/data/model configuration/dependencies match.', flush=True)
        if not a.skip_nccl:
            if ray.available_resources().get('GPU', 0) < world_size:
                raise RuntimeError(f'The {world_size} Ray GPUs are not all available for the NCCL check')
            address = 'tcp://' + head + ':' + os.environ.get('AR_NCCL_PORT', '29511')
            for node_index, node in enumerate(nodes):
                for local_rank in range(gpus_per_node):
                    jobs.append(ray.remote(num_cpus=1, num_gpus=1, max_calls=1)(nccl_sum).options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node['NodeID'], soft=False)
                    ).remote(node_index * gpus_per_node + local_rank, world_size, address))
            result = ray.get(jobs, timeout=180)
            print(f'NCCL all_reduce passed on all {world_size} ranks: sum=' + str(result[0]['sum']), flush=True)
    finally:
        for job in jobs:
            ray.cancel(job, force=True)
        ray.shutdown()


if __name__ == '__main__':
    main()

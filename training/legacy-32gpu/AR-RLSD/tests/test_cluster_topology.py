"""CPU-only checks for old two-node and new four-node launch topology."""
from __future__ import annotations

import hashlib
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import types
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ar_cluster_cpu', ROOT / 'verl/ar/check_cluster.py')
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)
HOSTS = ['192.0.2.12', '192.0.2.13', '192.0.2.10', '192.0.2.11']


class TopologyTest(unittest.TestCase):
    def test_old_defaults_and_single_worker_override(self):
        self.assertEqual(cluster.cluster_topology({}), (tuple(HOSTS[:2]), 8))
        self.assertEqual(cluster.cluster_topology(dict(HEAD_HOST='10.0.0.1', WORKER_HOST='10.0.0.2')),
                         (('10.0.0.1', '10.0.0.2'), 8))

    def test_four_nodes_with_ordered_workers_and_32_nccl_ranks(self):
        hosts, gpus = cluster.cluster_topology(dict(HEAD_HOST=HOSTS[0],
            WORKER_HOSTS=','.join(HOSTS[1:]), NNODES='4', N_GPUS='8'))
        self.assertEqual(hosts, tuple(HOSTS))
        ranks = [node * gpus + rank for node in range(len(hosts)) for rank in range(gpus)]
        self.assertEqual(ranks, list(range(32)))
        self.assertEqual(sum(rank + 1 for rank in ranks), 528)

    def test_host_list_automatically_sets_node_count(self):
        self.assertEqual(cluster.cluster_topology(dict(HEAD_HOST=HOSTS[0],
            WORKER_HOSTS=','.join(HOSTS[1:]), N_GPUS='4')), (tuple(HOSTS), 4))

    def test_invalid_hosts_and_topology_are_rejected(self):
        base = dict(HEAD_HOST=HOSTS[0], WORKER_HOSTS=','.join(HOSTS[1:]))
        cases = [
            dict(HEAD_HOST=HOSTS[1]), dict(WORKER_HOSTS=HOSTS[1] + ',' + HOSTS[1]),
            dict(WORKER_HOSTS=HOSTS[1] + ','), dict(WORKER_HOSTS=',' + HOSTS[1]),
            dict(WORKER_HOSTS=HOSTS[1] + ',,' + HOSTS[2]), dict(WORKER_HOSTS='192.0.2.10'),
            dict(WORKER_HOSTS='192.0.2.14'), dict(WORKER_HOSTS='node.example'),
            dict(WORKER_HOSTS='::1'), dict(WORKER_HOSTS='10.0.0.1;echo bad'),
            dict(NNODES='2'), dict(NNODES='0'), dict(NNODES='04'),
            dict(N_GPUS='0'), dict(N_GPUS='-1'), dict(N_GPUS='8.0'), dict(N_GPUS='８'),
        ]
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                cluster.cluster_topology({**base, **updates})

    def test_network_runtime_propagates_all_nccl_choices_and_proxy_bypass(self):
        supplied = dict(NO_PROXY=','.join(HOSTS), no_proxy=','.join(HOSTS),
            NCCL_DEBUG='INFO', NCCL_SOCKET_IFNAME='eth0', NCCL_IB_HCA='mlx5_0',
            TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_EMPTY='', UNRELATED_TOKEN='secret')
        expected = {key: value for key, value in supplied.items()
                    if key not in {'NCCL_EMPTY', 'UNRELATED_TOKEN'}}
        self.assertEqual(cluster.network_environment(supplied), expected)

    def test_large_jsonl_hash_uses_streaming_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            value = (b'{"sample": "large jsonl fixture"}\n' * 100000)
            path.write_bytes(value)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('full-file allocation')):
                self.assertEqual(cluster.sha256_file(path), hashlib.sha256(value).hexdigest())

    def test_fingerprint_includes_rlsd_and_sft_payloads(self):
        import types
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, sft, model = root / 'train.jsonl', root / 'sft.jsonl', root / 'model'
            train.write_text('{"rlsd":1}\n')
            sft.write_text('{"sft":1}\n')
            model.mkdir()
            (model / 'config.json').write_text('{}')
            fake_verl = types.ModuleType('verl')
            fake_verl.__file__ = str(ROOT / 'verl/__init__.py')
            with patch.dict(sys.modules, {'verl': fake_verl}), patch.dict(os.environ, {
                'TRAIN_DATA': str(train), 'SFT_DATA': str(sft), 'MODEL_PATH': str(model)}), \
                    patch.object(cluster.importlib.metadata, 'version', return_value='test'):
                first = cluster.fingerprint()
                sft.write_text('{"sft":2}\n')
                second = cluster.fingerprint()
            self.assertEqual(first['data'], second['data'])
            self.assertNotEqual(first['sft_data'], second['sft_data'])


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindir = self.root / 'env/bin'
        self.bindir.mkdir(parents=True)
        (self.bindir / 'activate').write_text('export PATH="' + str(self.bindir) + ':$PATH"\n')
        recorder = '#!' + sys.executable + '''
import json, os, sys
keys = ('HEAD_HOST', 'WORKER_HOST', 'WORKER_HOSTS', 'NODE_HOST', 'NNODES', 'N_GPUS', 'NO_PROXY',
        'MY_HOST_IP', 'MY_HOST_IPV6', 'RAY_ADDRESS', 'RAY_REDIS_ADDRESS',
        'RAY_external_storage_namespace', 'RAY_gcs_storage', 'RAY_NODE_IP_ADDRESS',
        'RAY_OVERRIDE_NODE_IP_ADDRESS', 'VLLM_HOST_IP')
with open(os.environ['AR_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(dict(args=sys.argv[1:], env={k: os.environ.get(k) for k in keys})) + '\\n')
sys.exit(7 if os.environ.get('AR_TEST_FAIL_CLUSTER') == '1' and 'verl.ar.check_cluster' in sys.argv else 0)
'''
        for name in ('python', 'ray'):
            path = self.bindir / name
            path.write_text(recorder)
            path.chmod(0o755)
        (self.bindir / 'hostname').write_text('#!/bin/sh\nprintf "%s\\n" "$AR_TEST_NODE_IP"\n')
        (self.bindir / 'hostname').chmod(0o755)
        self.calls = self.root / 'calls.jsonl'
        self.environment = {key: value for key, value in os.environ.items() if key not in {
            'HEAD_HOST', 'WORKER_HOST', 'WORKER_HOSTS', 'NODE_HOST', 'NNODES', 'N_GPUS', 'RAY_PORT'}}
        self.environment.update(TRAIN_ENV_ROOT=str(self.bindir.parent),
            OUTPUT_ROOT=str(self.root / 'outputs'), RUN_NAME='test',
            AR_TEST_CALLS=str(self.calls), AR_TEST_NODE_IP=HOSTS[0])

    def run_launcher(self, mode, *args, **updates):
        if self.calls.exists():
            self.calls.unlink()
        result = subprocess.run(['bash', str(ROOT / 'examples/action_recognition/run_qwen25.sh'),
            mode, *args], env={**self.environment, **updates}, capture_output=True, text=True, timeout=15)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return result, calls

    def test_old_two_node_default_worker_still_works(self):
        result, calls = self.run_launcher('worker', AR_TEST_NODE_IP=HOSTS[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--node-ip-address=' + HOSTS[1], calls[0]['args'])
        self.assertEqual(calls[0]['env']['NNODES'], '2')

    def test_all_three_workers_join_same_head_on_their_own_ips(self):
        for host in HOSTS[1:]:
            with self.subTest(host=host):
                result, calls = self.run_launcher('worker', WORKER_HOSTS=','.join(HOSTS[1:]),
                    NNODES='4', NODE_HOST=host, AR_TEST_NODE_IP=host)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--node-ip-address=' + host, calls[0]['args'])
                self.assertIn('--address=' + HOSTS[0] + ':6401', calls[0]['args'])
                self.assertIn('--num-gpus=8', calls[0]['args'])
                self.assertEqual(calls[0]['env']['MY_HOST_IP'], host)
                for member in HOSTS:
                    self.assertIn(member, calls[0]['env']['NO_PROXY'].split(','))

    def test_four_node_training_passes_topology_to_driver_once(self):
        result, calls = self.run_launcher('train', 'worker.actor.global_batch_size=32',
            WORKER_HOSTS=','.join(HOSTS[1:]), NNODES='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        drivers = [call for call in calls if 'verl.trainer.opsd_main' in call['args']]
        self.assertEqual(len(drivers), 1)
        self.assertIn('trainer.nnodes=4', drivers[0]['args'])
        self.assertIn('trainer.n_gpus_per_node=8', drivers[0]['args'])
        self.assertIn('worker.actor.global_batch_size=32', drivers[0]['args'])
        self.assertIn('opsd.ar_reward_mode=protocol', drivers[0]['args'])
        self.assertIn('opsd.ar_reference_coef=0.1', drivers[0]['args'])
        self.assertIn('opsd.ar_alignment_mode=mesd', drivers[0]['args'])
        self.assertIn('opsd.ar_teacher_update_mode=ema', drivers[0]['args'])
        self.assertIn('opsd.ar_teacher_update_rate=0.01', drivers[0]['args'])
        self.assertIn('data.truncation=left', drivers[0]['args'])
        self.assertIn('worker.actor.loss_avg_mode=seq', drivers[0]['args'])
        self.assertEqual(json.loads(next(arg.split('=', 1)[1] for arg in drivers[0]['args']
            if arg.startswith('worker.reward.reward_function_kwargs='))), {'reward_mode': 'protocol'})
        self.assertEqual(drivers[0]['env']['MY_HOST_IP'], HOSTS[0])
        self.assertEqual(drivers[0]['env']['RAY_ADDRESS'], HOSTS[0] + ':6401')

    def test_inherited_cluster_routing_is_removed_and_current_node_ip_is_used(self):
        stale = dict(MY_HOST_IP='10.0.0.99', MY_HOST_IPV6='::ffff:10.0.0.99',
            RAY_ADDRESS='10.0.0.99:6379', RAY_REDIS_ADDRESS='10.0.0.99:6379',
            RAY_external_storage_namespace='unrelated-session', RAY_gcs_storage='redis',
            RAY_NODE_IP_ADDRESS='10.0.0.99', RAY_OVERRIDE_NODE_IP_ADDRESS='10.0.0.99',
            VLLM_HOST_IP='10.0.0.99')
        for mode, host in (('head', HOSTS[0]), ('worker', HOSTS[2]), ('train', HOSTS[0])):
            with self.subTest(mode=mode):
                result, calls = self.run_launcher(mode, WORKER_HOSTS=','.join(HOSTS[1:]),
                    NNODES='4', NODE_HOST=host, AR_TEST_NODE_IP=host,
                    RUN_NAME='clean_' + mode, **stale)
                self.assertEqual(result.returncode, 0, result.stderr)
                relevant = next(call for call in calls if ('verl.trainer.opsd_main' in call['args']
                    if mode == 'train' else 'start' in call['args']))
                self.assertEqual(relevant['env']['MY_HOST_IP'], host)
                self.assertEqual(relevant['env']['VLLM_HOST_IP'], host)
                for key in ('MY_HOST_IPV6', 'RAY_REDIS_ADDRESS',
                            'RAY_external_storage_namespace', 'RAY_gcs_storage',
                            'RAY_NODE_IP_ADDRESS', 'RAY_OVERRIDE_NODE_IP_ADDRESS'):
                    self.assertIsNone(relevant['env'][key])
                self.assertEqual(relevant['env']['RAY_ADDRESS'],
                    HOSTS[0] + ':6401' if mode == 'train' else None)
                if mode == 'head':
                    temp_arg = next(arg for arg in relevant['args'] if arg.startswith('--temp-dir='))
                    session = Path(temp_arg.split('=', 1)[1])
                    self.addCleanup(session.rmdir)
                    self.assertTrue(session.is_dir())
                    self.assertIn('ar-action-32gpu-v3.', session.name)
                    self.assertNotIn('--block', relevant['args'])

    def test_bad_topology_and_wrong_worker_cannot_start_ray(self):
        cases = [dict(NNODES='2'), dict(NODE_HOST=HOSTS[0]), dict(NODE_HOST='10.1.1.1'),
                 dict(WORKER_HOSTS=HOSTS[1] + ','), dict(N_GPUS='0'),
                 dict(AR_TEST_NODE_IP=HOSTS[0])]
        for updates in cases:
            with self.subTest(updates=updates):
                result, calls = self.run_launcher('worker', **{
                    'WORKER_HOSTS': ','.join(HOSTS[1:]), 'NNODES': '4',
                    'NODE_HOST': HOSTS[1], 'AR_TEST_NODE_IP': HOSTS[1], **updates})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])

    def test_cluster_failure_blocks_training_driver(self):
        result, calls = self.run_launcher('train', WORKER_HOSTS=','.join(HOSTS[1:]),
            NNODES='4', AR_TEST_FAIL_CLUSTER='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any('verl.trainer.opsd_main' in call['args'] for call in calls))

    def test_smoke_defaults_can_be_overridden_to_full_shape(self):
        result, calls = self.run_launcher('smoke', 'data.video_max_frames=16',
            'data.max_response_length=768', 'worker.rollout.enforce_eager=false',
            WORKER_HOSTS=','.join(HOSTS[1:]), NNODES='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        driver = next(call for call in calls if 'verl.trainer.opsd_main' in call['args'])
        args = driver['args']
        self.assertLess(args.index('data.video_max_frames=8'), args.index('data.video_max_frames=16'))
        self.assertLess(args.index('worker.rollout.enforce_eager=true'),
                        args.index('worker.rollout.enforce_eager=false'))
        self.assertIn('opsd.ar_reward_mode=protocol', args)
        self.assertIn('opsd.ar_reference_coef=0.1', args)
        self.assertEqual(json.loads(next(arg.split('=', 1)[1] for arg in args
            if arg.startswith('worker.reward.reward_function_kwargs='))), {'reward_mode': 'protocol'})

    def test_training_cli_cannot_silently_replace_the_fixed_e1_experiment(self):
        result, calls = self.run_launcher('train', 'opsd.ar_reward_mode=answer_only',
            'opsd.ar_reference_coef=0.2',
            'worker.reward.reward_function_kwargs={"reward_mode":"answer_only"}',
            'opsd.ar_alignment_mode=legacy','opsd.ar_teacher_update_rate=1.0',
            'opsd.rlsd_lambda_decay_steps=0','data.truncation=error',
            'worker.actor.loss_avg_mode=token')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = next(call['args'] for call in calls if 'verl.trainer.opsd_main' in call['args'])
        for key, expected in (('opsd.ar_reward_mode','protocol'),('opsd.ar_reference_coef','0.1'),
            ('opsd.ar_alignment_mode','mesd'),('opsd.ar_teacher_update_rate','0.01'),
            ('opsd.rlsd_lambda_decay_steps','1000'),('data.truncation','left'),
            ('worker.actor.loss_avg_mode','seq')):
            self.assertEqual([arg.split('=', 1)[1] for arg in args if arg.startswith(key+'=')][-1], expected)
        effective_kwargs = [arg.split('=', 1)[1] for arg in args
            if arg.startswith('worker.reward.reward_function_kwargs=')][-1]
        self.assertEqual(json.loads(effective_kwargs), {'reward_mode': 'protocol'})


class ClusterMainTest(unittest.TestCase):
    """Run the real cluster orchestration with recorded Ray RPCs, never GPUs."""
    def fixture(self, *, extra_node=False, gpu_count=8, differing_node=None, available=32):
        fake = types.ModuleType('ray')
        fake.calls = []
        fake.nodes = lambda: [dict(Alive=True, NodeManagerAddress=host, NodeID=host,
            Resources={'GPU': gpu_count if index == 2 else 8})
            for index, host in enumerate(HOSTS + (['10.0.0.99'] if extra_node else []))]
        fake.available_resources = lambda: {'GPU': available}
        fake.init = lambda **values: fake.calls.append(('init', values))
        fake.cancel = lambda *args, **kwargs: None
        fake.shutdown = lambda: None

        class Strategy:
            def __init__(self, node_id, soft):
                self.node_id = node_id

        class Remote:
            def __init__(self, function):
                self.function = function
            def options(self, **values):
                self.node = values['scheduling_strategy'].node_id
                return self
            def remote(self, *args):
                return self.function, self.node, args

        fake.remote = lambda **kwargs: lambda function: Remote(function)

        def get(requests, **kwargs):
            output = []
            for function, node, args in requests:
                fake.calls.append(('remote', node, args))
                if not args:
                    output.append({'match': node != differing_node})
                else:
                    rank, world, address = args
                    output.append(dict(rank=rank, sum=world * (world + 1) / 2))
            return output

        fake.get = get
        util = types.ModuleType('ray.util')
        strategy = types.ModuleType('ray.util.scheduling_strategies')
        strategy.NodeAffinitySchedulingStrategy = Strategy
        modules = {'ray': fake, 'ray.util': util, 'ray.util.scheduling_strategies': strategy}
        environment = dict(HEAD_HOST=HOSTS[0], WORKER_HOSTS=','.join(HOSTS[1:]), NNODES='4',
            N_GPUS='8', RAY_ADDRESS=HOSTS[0] + ':6390', SFT_DATA='/data/sft.jsonl',
            NCCL_SOCKET_IFNAME='eth0', NO_PROXY=','.join(HOSTS))
        return fake, modules, environment

    def run_main(self, **settings):
        fake, modules, environment = self.fixture(**settings)
        stdout = io.StringIO()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, environment), \
                patch.object(sys, 'argv', ['check_cluster']), \
                patch.object(cluster, 'fingerprint', return_value={'match': True}), \
                contextlib.redirect_stdout(stdout):
            cluster.main()
        return fake, stdout.getvalue()

    def test_four_nodes_run_all_32_nccl_tasks_and_forward_network(self):
        fake, stdout = self.run_main()
        jobs = [call for call in fake.calls if call[0] == 'remote' and call[2]]
        self.assertEqual([call[2][0] for call in jobs], list(range(32)))
        self.assertTrue(all(call[2][1] == 32 for call in jobs))
        self.assertIn('all 32 ranks: sum=528.0', stdout)
        runtime = fake.calls[0][1]['runtime_env']['env_vars']
        self.assertEqual(runtime['SFT_DATA'], '/data/sft.jsonl')
        self.assertEqual(runtime['NCCL_SOCKET_IFNAME'], 'eth0')
        self.assertEqual(runtime['NO_PROXY'], ','.join(HOSTS))

    def test_extra_node_wrong_gpu_unavailable_gpu_and_worker3_difference_fail(self):
        for setting in (dict(extra_node=True), dict(gpu_count=7), dict(available=31),
                        dict(differing_node=HOSTS[3])):
            with self.subTest(setting=setting), self.assertRaises(RuntimeError):
                self.run_main(**setting)


if __name__ == '__main__':
    unittest.main()

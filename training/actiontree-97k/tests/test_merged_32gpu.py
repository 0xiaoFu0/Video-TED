"""CPU contract tests: paired exports, tokenizer receipts and the real shell entry."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('ar_merged_cpu')
package.__path__ = [str(ROOT / 'verl/ar')]
sys.modules[package.__name__] = package
format_spec = importlib.util.spec_from_file_location('ar_merged_cpu.response_format',
    ROOT / 'verl/ar/response_format.py')
formatter = importlib.util.module_from_spec(format_spec)
sys.modules[format_spec.name] = formatter
format_spec.loader.exec_module(formatter)
spec = importlib.util.spec_from_file_location('ar_merged_cpu.check_merged_data',
    ROOT / 'verl/ar/check_merged_data.py')
checker = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'verl.ar.response_format': formatter}):
    spec.loader.exec_module(checker)
from ar_merged_cpu import validation


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with (ROOT / 'data/train.rlsd.think.jsonl').open() as stream:
            template = json.loads(stream.readline())
        self.rows = []
        for index, batch in enumerate(('V1', 'LLaVA', 'K700', 'SSv2')):
            row = copy.deepcopy(template)
            video = self.root / f'video-{index}.avi'
            video.write_bytes(b'fixture')
            row['videos'] = [str(video)]
            row['metadata'].update(sample_id=f'sample-{index}', video_relpath=video.name,
                                   merge_batch=batch, dataset=batch)
            self.rows.append(row)
        self.expected = dict.fromkeys(('V1', 'LLaVA', 'K700', 'SSv2'), 1)

    def write(self, sft_mutation=None):
        pairs = [{'messages': [{'role': 'user', 'content': r['prompt']},
                              {'role': 'assistant', 'content': r['reference_response']}],
                  'answer': r['answer'], 'videos': r['videos'], 'metadata': r['metadata']}
                 for r in self.rows]
        if sft_mutation:
            sft_mutation(pairs)
        for name, rows in zip(checker.PAYLOADS, (self.rows, pairs)):
            (self.root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest = {'total_rows': len(self.rows), 'batches': {
            name: {'retained': count} for name, count in self.expected.items()},
            'output_sha256': {name: checker.sha256_file(self.root / name) for name in checker.PAYLOADS}}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))

    def check(self):
        return checker.validate_dataset(self.root, lambda text: 1500, self.expected)

    def test_valid_pair_checks_each_source_and_measures_long_references(self):
        self.write()
        report, probes = self.check()
        self.assertEqual(report['rows'], 4)
        self.assertEqual(report['reference_max_length'], 1501)
        self.assertEqual([r['metadata']['merge_batch'] for r, s in probes], list(self.expected))
        self.assertEqual(report['input'], checker.input_identity(self.root))
        self.assertEqual(report['schema_version'], 2)
        self.assertEqual(report['reward_mode'], 'protocol')
        self.assertEqual(report['reference_protocol_valid_rows'], 4)
        self.assertEqual(report['reward_validators'], checker.reward_validator_identity())

    def test_wrong_pair_and_missing_rows_fail_without_dropping_data(self):
        for mutate in (lambda pairs: pairs[0].update(answer='wrong'), lambda pairs: pairs.pop()):
            self.write(mutate)
            with self.assertRaises(ValueError):
                self.check()
            self.assertEqual(len(list(checker.read_rows(self.root / checker.PAYLOADS[0]))), 4)

    def test_duplicate_id_path_empty_video_and_rejected_reference_fail(self):
        original = copy.deepcopy(self.rows)
        for issue in ('id', 'path', 'empty_video', 'rejected'):
            self.rows = copy.deepcopy(original)
            if issue == 'id':
                self.rows[1]['metadata']['sample_id'] = self.rows[0]['metadata']['sample_id']
            elif issue == 'path':
                self.rows[1]['videos'] = self.rows[0]['videos']
            elif issue == 'empty_video':
                Path(self.rows[0]['videos'][0]).write_bytes(b'')
            else:
                self.rows[0]['reference_status'] = 'rejected'
            self.write()
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                self.check()
            Path(original[0]['videos'][0]).write_bytes(b'fixture')

    def test_changed_payload_is_rejected_before_training(self):
        self.write()
        path = self.root / checker.PAYLOADS[0]
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.check()

    def test_reference_target_mismatch_and_unknown_source_fail(self):
        original = copy.deepcopy(self.rows)
        for issue in ('target', 'source'):
            self.rows = copy.deepcopy(original)
            if issue == 'target':
                self.rows[0]['answer'] = 'unrelated_action'
            else:
                self.rows[0]['metadata']['merge_batch'] = 'unexpected'
            self.write()
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                self.check()

    def test_tokenizer_fingerprint_changes_when_tokenizer_changes(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'tokenizer.json').write_text('{"vocab": 1}')
        first = checker.tokenizer_identity(model)
        (model / 'tokenizer.json').write_text('{"vocab": 2}')
        self.assertNotEqual(first, checker.tokenizer_identity(model))

    def test_protocol_invalid_reference_stops_without_rewriting_paired_payloads(self):
        # The existing strict envelope accepts this inner tag, but E1 does not.
        self.rows[0]['reference_response'] = self.rows[0]['reference_response'].replace(
            '<think>', '<think><object>ambiguous</object>', 1)
        self.write()
        before = {name: (self.root / name).read_bytes() for name in checker.PAYLOADS}
        with self.assertRaisesRegex(ValueError, 'reward protocol'):
            self.check()
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in checker.PAYLOADS})
        self.assertEqual(len(list(checker.read_rows(self.root / checker.PAYLOADS[0]))), 4)

    def test_prepared_receipt_requires_current_reward_validators_and_protocol(self):
        self.write()
        report, _ = self.check()
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'tokenizer.json').write_text('{}')
        report['tokenizer'] = checker.tokenizer_identity(model)
        report['probe_sha256'] = {}
        for payload in checker.PAYLOADS:
            name = 'probe.' + payload
            path = self.root / name
            path.write_bytes((self.root / payload).read_bytes())
            report['probe_sha256'][name] = checker.sha256_file(path)
        receipt = self.root / 'report.json'
        with patch.object(checker, 'EXPECTED_BATCHES', self.expected):
            receipt.write_text(json.dumps(report))
            self.assertEqual(checker.check_report(self.root, model, receipt), report)
            mutations = [dict(schema_version=1), dict(reward_mode='answer_only'),
                         dict(reference_protocol_valid_rows=3)]
            altered_hash = copy.deepcopy(report['reward_validators'])
            altered_hash['response_format.py'] = '0' * 64
            mutations.append(dict(reward_validators=altered_hash))
            for update in mutations:
                with self.subTest(update=update):
                    receipt.write_text(json.dumps({**report, **update}))
                    with self.assertRaisesRegex(ValueError, 'prepare again'):
                        checker.check_report(self.root, model, receipt)


class BatchTest(unittest.TestCase):
    def test_32_rank_prompt_dispatch_rejects_small_batches_before_model_loading(self):
        for batch, mini in ((16, None), (32, 16), (48, None), (32, 0)):
            with self.subTest(batch=batch, mini=mini), self.assertRaises(ValueError):
                validation.validate_dispatch_batches(types.SimpleNamespace(
                    rollout_batch_size=batch, mini_rollout_batch_size=mini), 32)

    def test_valid_old_and_new_prompt_batches_pass(self):
        for world, batch, mini in ((16, 16, None), (32, 32, None), (32, 64, 32)):
            validation.validate_dispatch_batches(types.SimpleNamespace(
                rollout_batch_size=batch, mini_rollout_batch_size=mini), world)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        bindir = root / 'env/bin'
        bindir.mkdir(parents=True)
        (bindir / 'activate').write_text('export PATH="' + str(bindir) + ':$PATH"\n')
        recorder = '#!' + sys.executable + '''
import json, os, sys
args = sys.argv[1:]
with open(os.environ['AR_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(dict(args=args, train=os.getenv('TRAIN_DATA'), sft=os.getenv('SFT_DATA'),
        env={k: os.getenv(k) for k in ('MY_HOST_IP', 'MY_HOST_IPV6', 'RAY_ADDRESS',
            'RAY_REDIS_ADDRESS', 'RAY_external_storage_namespace', 'RAY_gcs_storage',
            'RAY_NODE_IP_ADDRESS', 'RAY_OVERRIDE_NODE_IP_ADDRESS', 'VLLM_HOST_IP')})) + '\\n')
if '--check-report' in args:
    if os.getenv('AR_TEST_FAIL_REPORT') == '1': sys.exit(9)
    print('1501')
if 'verl.ar.check_cluster' in args and os.getenv('AR_TEST_FAIL_CLUSTER') == '1': sys.exit(7)
'''
        for name in ('python', 'ray'):
            (bindir / name).write_text(recorder)
            (bindir / name).chmod(0o755)
        (bindir / 'hostname').write_text('#!/bin/sh\nprintf "%s\\n" "$AR_TEST_NODE_IP"\n')
        (bindir / 'hostname').chmod(0o755)
        self.calls = root / 'calls.jsonl'
        self.env = dict(os.environ, TRAIN_ENV_ROOT=str(bindir.parent),
            HEAD_HOST='192.0.2.12', WORKER_HOSTS='192.0.2.13,192.0.2.10,192.0.2.11',
            NNODES='4', N_GPUS='8', RUN_NAME='merged_fixture', RAY_PORT='6390', AR_NCCL_PORT='29530',
            MERGED_DATA_ROOT='/data/action merge', OUTPUT_ROOT=str(root / 'outputs'),
            AR_TEST_CALLS=str(self.calls), AR_TEST_NODE_IP='192.0.2.12')

    def run_launcher(self, mode, **updates):
        result = subprocess.run(['bash', str(ROOT / 'examples/action_recognition/run_merged_32gpu.sh'), mode],
            env={**self.env, **updates}, capture_output=True, text=True, timeout=15)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return result, calls

    def test_train_starts_driver_once_with_selected_configuration_and_measured_budget(self):
        result, calls = self.run_launcher('train')
        self.assertEqual(result.returncode, 0, result.stderr)
        drivers = [c for c in calls if 'verl.trainer.opsd_main' in c['args']]
        self.assertEqual(len(drivers), 1)
        args = drivers[0]['args']
        for setting in ('trainer.nnodes=4', 'trainer.n_gpus_per_node=8', 'trainer.total_epochs=1',
                        'data.rollout_batch_size=32', 'worker.actor.global_batch_size=32',
                        'worker.rollout.n=4', 'data.video_max_frames=16', 'data.max_response_length=768',
                        'worker.rollout.enforce_eager=false', 'opsd.ar_reference_max_length=1501',
                        'opsd.ar_reward_mode=protocol',
                        'opsd.ar_reference_coef=0.1',
                        'opsd.ar_alignment_mode=mesd','opsd.ar_teacher_update_mode=ema',
                        'opsd.ar_teacher_update_rate=0.01','opsd.ar_teacher_update_interval=1',
                        'opsd.ar_teacher_disable_after_step=1000','opsd.ar_teacher_max_prompt_length=18432',
                        'opsd.rlsd_lambda=0.5','opsd.rlsd_lambda_warmup_steps=0',
                        'opsd.rlsd_lambda_decay_steps=1000',
                        'data.truncation=left','worker.actor.loss_avg_mode=seq',
                        'trainer.load_checkpoint_path=null', 'trainer.find_last_checkpoint=false'):
            self.assertIn(setting, args)
        self.assertEqual(json.loads(next(arg.split('=', 1)[1] for arg in args
            if arg.startswith('worker.reward.reward_function_kwargs='))), {'reward_mode': 'protocol'})
        self.assertIn('data.train_files=/data/action merge/train.rlsd.think.jsonl', args)
        self.assertEqual(drivers[0]['env']['MY_HOST_IP'], '192.0.2.12')
        self.assertEqual(drivers[0]['env']['RAY_ADDRESS'], '192.0.2.12:6390')

    def test_bad_data_report_or_cluster_prevents_training(self):
        for flag in ('AR_TEST_FAIL_REPORT', 'AR_TEST_FAIL_CLUSTER'):
            if self.calls.exists(): self.calls.unlink()
            result, calls = self.run_launcher('train', **{flag: '1'})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(any('verl.trainer.opsd_main' in c['args'] for c in calls))

    def test_smoke_uses_production_limits_after_default_overrides(self):
        result, calls = self.run_launcher('smoke')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = next(c['args'] for c in calls if 'verl.trainer.opsd_main' in c['args'])
        self.assertIn('trainer.max_steps=2', args)
        self.assertEqual([v for v in args if v.startswith('data.video_max_frames=')][-1], 'data.video_max_frames=16')
        self.assertEqual([v for v in args if v.startswith('data.max_response_length=')][-1], 'data.max_response_length=768')
        self.assertEqual([v for v in args if v.startswith('worker.rollout.enforce_eager=')][-1], 'worker.rollout.enforce_eager=false')
        self.assertIn('opsd.ar_reward_mode=protocol', args)
        self.assertEqual(json.loads(next(arg.split('=', 1)[1] for arg in args
            if arg.startswith('worker.reward.reward_function_kwargs='))), {'reward_mode': 'protocol'})
        self.assertIn('opsd.ar_reference_coef=0.1', args)
        self.assertIn('opsd.ar_alignment_mode=mesd', args)
        self.assertIn('opsd.ar_teacher_update_mode=ema', args)
        self.assertIn('opsd.rlsd_lambda_decay_steps=1000', args)
        self.assertIn('data.truncation=left', args)
        self.assertIn('worker.actor.loss_avg_mode=seq', args)

    def test_video_probe_uses_small_source_balanced_pair_and_same_resolution(self):
        result, calls = self.run_launcher('check-video')
        self.assertEqual(result.returncode, 0, result.stderr)
        call = next(c for c in calls if 'verl.ar.check_video' in c['args'])
        self.assertTrue(call['train'].endswith('/probe.rlsd.think.jsonl'))
        self.assertTrue(call['sft'].endswith('/probe.sft.think.jsonl'))
        self.assertIn('--backward', call['args'])
        self.assertEqual(call['args'][call['args'].index('--max-frames') + 1], '16')
        self.assertEqual(call['args'][call['args'].index('--max-pixels') + 1], '100352')
        self.assertEqual(call['args'][call['args'].index('--selection') + 1], 'first')

    def test_three_worker_ip_selections_join_the_one_head(self):
        for host in self.env['WORKER_HOSTS'].split(','):
            if self.calls.exists(): self.calls.unlink()
            result, calls = self.run_launcher('worker', NODE_HOST=host, AR_TEST_NODE_IP=host)
            self.assertEqual(result.returncode, 0, result.stderr)
            ray = calls[-1]['args']
            self.assertIn('--address=192.0.2.12:6390', ray)
            self.assertIn('--node-ip-address=' + host, ray)
            self.assertEqual(calls[-1]['env']['MY_HOST_IP'], host)

    def test_merged_entry_clears_inherited_ray_session_before_starting_driver(self):
        result, calls = self.run_launcher('train', MY_HOST_IP='10.0.0.99',
            MY_HOST_IPV6='::ffff:10.0.0.99', RAY_ADDRESS='10.0.0.99:6379',
            RAY_REDIS_ADDRESS='10.0.0.99:6379', RAY_external_storage_namespace='old-session',
            RAY_gcs_storage='redis', RAY_NODE_IP_ADDRESS='10.0.0.99',
            RAY_OVERRIDE_NODE_IP_ADDRESS='10.0.0.99', VLLM_HOST_IP='10.0.0.99')
        self.assertEqual(result.returncode, 0, result.stderr)
        driver = next(call for call in calls if 'verl.trainer.opsd_main' in call['args'])
        self.assertEqual(driver['env']['MY_HOST_IP'], '192.0.2.12')
        self.assertEqual(driver['env']['VLLM_HOST_IP'], '192.0.2.12')
        self.assertEqual(driver['env']['RAY_ADDRESS'], '192.0.2.12:6390')
        for key in ('MY_HOST_IPV6', 'RAY_REDIS_ADDRESS',
                    'RAY_external_storage_namespace', 'RAY_gcs_storage',
                    'RAY_NODE_IP_ADDRESS', 'RAY_OVERRIDE_NODE_IP_ADDRESS'):
            self.assertIsNone(driver['env'][key])


if __name__ == '__main__':
    unittest.main()

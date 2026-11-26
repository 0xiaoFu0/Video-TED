"""Exercise the real overlap -> 16-GPU -> annotation shell chain without models."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from ar_annotate.common import ROOT


class OverlapLaunchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidates = self.root / 'incremental candidates.jsonl'
        self.candidates.write_text(''.join(json.dumps({'sample_id': f'sample_{index}'}) + '\n'
                                          for index in range(20)))
        self.output = self.root / 'shared overlap run'
        self.calls = self.root / 'worker_calls.jsonl'
        self.activation = self.root / 'activation.txt'
        bindir = self.root / 'mock env/bin'
        bindir.mkdir(parents=True)
        recorder = '#!' + sys.executable + '''
import json, os, sys
with open(os.environ['OVERLAP_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:],
        'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
        'node_rank': os.environ['NODE_RANK']}) + '\\n')
sys.exit(7 if os.getenv('OVERLAP_TEST_FAIL_SHARD') in sys.argv else 0)
'''
        (bindir / 'python').write_text(recorder)
        (bindir / 'python').chmod(0o755)
        (bindir / 'hostname').write_text('#!/bin/sh\nprintf "%s\\n" "$OVERLAP_TEST_NODE_IP"\n')
        (bindir / 'hostname').chmod(0o755)
        conda_sh = self.root / 'mock conda.sh'
        conda_sh.write_text('''
printf 'activated\\n' >> "$OVERLAP_TEST_ACTIVATION"
conda() {
  [[ "$1" == activate && "$2" == ar-qwen38-annotate ]] || return 2
  export CONDA_PREFIX="$OVERLAP_TEST_ENV"
}
''')
        self.env = {**os.environ,
                    'CANDIDATES': str(self.candidates), 'RUN_DIR': str(self.output),
                    'CONDA_SH': str(conda_sh), 'ANNOTATION_ENV_NAME': 'ar-qwen38-annotate',
                    'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
                    'OVERLAP_TEST_ENV': str(bindir.parent),
                    'OVERLAP_TEST_ACTIVATION': str(self.activation),
                    'OVERLAP_TEST_CALLS': str(self.calls),
                    'GPU_IDS': '0,1,2,3,4,5,6,7', 'DRY_RUN': '0',
                    'NODE_COUNT': '1', 'NODE_RANK': '99', 'GPUS_PER_WORKER': '2',
                    'LIMIT_PER_DATASET': '10', 'ROLLOUTS': '1',
                    'MIN_ACCEPTED_ROLLOUTS': '3', 'PROMPT_PROFILE': 'legacy',
                    'RETRY_ERRORS': '1'}
        self.script = ROOT / 'scripts/launch_overlap_node.sh'

    def tearDown(self):
        self.temporary.cleanup()

    def run_script(self, rank, *args, **overrides):
        variables = {**self.env, 'OVERLAP_TEST_NODE_IP':
                     '192.0.2.10' if rank == '0' else '192.0.2.11', **overrides}
        return subprocess.run(['bash', str(self.script), rank, *args], env=variables,
                              cwd=self.root, text=True, capture_output=True, timeout=15)

    def read_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    @staticmethod
    def option(args, name):
        return args[args.index(name) + 1]

    def test_real_chain_launches_disjoint_sixteen_shards_with_one_shared_run(self):
        for rank, host in [('0', '192.0.2.10'), ('1', '192.0.2.11')]:
            result = self.run_script(rank)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f'Overlap node rank {rank} = {host}', result.stdout)
            self.assertIn('Local workers completed', result.stdout)
        calls = self.read_calls()
        self.assertEqual(len(calls), 16)
        shards = [int(self.option(call['args'], '--shard-index')) for call in calls]
        self.assertEqual(sorted(shards), list(range(16)))
        for call in calls:
            args = call['args']
            shard = int(self.option(args, '--shard-index'))
            self.assertEqual(call['node_rank'], '0' if shard < 8 else '1')
            self.assertEqual(call['gpu'], str(shard % 8))
            self.assertEqual(self.option(args, '--out'), str(self.output))
            self.assertEqual(self.option(args, '--candidates'), str(self.candidates))
            self.assertEqual(self.option(args, '--num-shards'), '16')
            self.assertEqual(self.option(args, '--limit-per-dataset'), '0')
            self.assertEqual(self.option(args, '--rollouts'), '4')
            self.assertEqual(self.option(args, '--min-accepted-rollouts'), '1')
            self.assertEqual(self.option(args, '--prompt-profile'), 'tree_v1')
            self.assertNotIn('--retry-errors', args)
            self.assertIn('-E', args)
            self.assertIn('-s', args)

    def test_dry_run_uses_real_subordinate_shard_plan_without_activation(self):
        for rank in ('0', '1'):
            result = self.run_script(rank, DRY_RUN='1', CONDA_SH='/missing/conda.sh')
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line)[1:] for line in result.stdout.splitlines()
                        if line.startswith('CUDA_VISIBLE_DEVICES=')]
            self.assertEqual(len(commands), 8)
            start = int(rank) * 8
            self.assertEqual([int(self.option(command, '--shard-index')) for command in commands],
                             list(range(start, start + 8)))
            for command in commands:
                self.assertEqual(self.option(command, '--out'), str(self.output))
                self.assertEqual(self.option(command, '--limit-per-dataset'), '0')
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.calls.exists())
        self.assertFalse(self.output.exists())

    def test_empty_incremental_pool_exits_without_environment_or_workers(self):
        self.candidates.write_text('\n \n\t\n')
        for rank in ('0', '1'):
            for dry_run in ('0', '1'):
                result = self.run_script(rank, DRY_RUN=dry_run, CONDA_SH='/missing/conda.sh')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('增量候选为 0', result.stdout)
                self.assertIn('未加载模型', result.stdout)
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.calls.exists())
        self.assertFalse(self.output.exists())

    def test_missing_or_invalid_candidates_fail_before_activation(self):
        for contents in ('not json\n', '{}\n', '{"sample_id": ""}\n'):
            self.candidates.write_text(contents)
            result = self.run_script('0')
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('无法读取增量候选', result.stderr)
        self.candidates.unlink()
        result = self.run_script('0', DRY_RUN='1')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.calls.exists())

    def test_wrong_rank_host_cannot_start_workers(self):
        result = self.run_script('0', OVERLAP_TEST_NODE_IP='192.0.2.11')
        self.assertEqual(result.returncode, 2)
        self.assertIn('192.0.2.10', result.stderr)
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.calls.exists())

    def test_invalid_rank_argument_count_gpu_count_and_dry_run_fail(self):
        result = subprocess.run(['bash', str(self.script)], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        for rank, args, overrides in [('2', (), {}), ('0', ('extra',), {}),
                                      ('0', (), {'GPU_IDS': '0,1'}), ('0', (), {'DRY_RUN': 'yes'})]:
            result = self.run_script(rank, *args, **overrides)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.calls.exists())

    def test_existing_run_is_forwarded_without_overwrite_or_retry_flag(self):
        self.output.mkdir()
        saved = self.output / 'run.json'
        saved.write_text('{"run_hash": "existing provenance"}\n')
        before = saved.read_bytes()
        result = self.run_script('0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(saved.read_bytes(), before)
        for call in self.read_calls():
            self.assertEqual(self.option(call['args'], '--out'), str(self.output))
            self.assertNotIn('--retry-errors', call['args'])

    def test_default_paths_use_incremental_candidates_and_full_overlap_run(self):
        project = self.root / 'independent annotation package'
        scripts = project / 'scripts'
        scripts.mkdir(parents=True)
        for name in ('launch_overlap_node.sh', 'launch_16gpu_node.sh', 'launch_annotation.sh'):
            shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
        candidates = project / 'work/candidates_overlap_incremental/candidates.jsonl'
        candidates.parent.mkdir(parents=True)
        shutil.copyfile(self.candidates, candidates)
        variables = {**self.env, 'DRY_RUN': '1'}
        variables.pop('CANDIDATES')
        variables.pop('RUN_DIR')
        variables.pop('CONFIG', None)
        result = subprocess.run(['bash', str(scripts / 'launch_overlap_node.sh'), '0'],
                                env=variables, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line)[1:] for line in result.stdout.splitlines()
                    if line.startswith('CUDA_VISIBLE_DEVICES=')]
        self.assertEqual(len(commands), 8)
        for command in commands:
            self.assertEqual(self.option(command, '--candidates'), str(candidates))
            self.assertEqual(self.option(command, '--out'), str(project / 'work/full_overlap_16gpu'))
            self.assertEqual(self.option(command, '--config'), str(project / 'config.server.json'))
        self.assertFalse(self.activation.exists())
        self.assertFalse(self.calls.exists())

    def test_any_worker_failure_propagates_to_node_exit_status(self):
        result = self.run_script('1', OVERLAP_TEST_FAIL_SHARD='12')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(len(self.read_calls()), 8)
        self.assertIn('At least one local worker failed', result.stderr)


if __name__ == '__main__':
    unittest.main()

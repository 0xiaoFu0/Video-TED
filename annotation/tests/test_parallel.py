import os
import subprocess
import time
import unittest

from ar_annotate.common import ROOT
from ar_annotate.decoder_pool import DecoderProcess


def echo_worker(conn):
    while True:
        try:
            job = conn.recv()
        except EOFError:
            return
        if job.get('hang'):
            time.sleep(10)
        if job.get('fail'):
            conn.send({'ok': False, 'error': 'bad codec'})
        else:
            conn.send({'ok': True, 'pid': os.getpid()})


class ParallelTest(unittest.TestCase):
    def dry_run(self, **env):
        variables = dict(os.environ, DRY_RUN='1', GPU_IDS='0,1,2,3,4,5,6,7',
                         NODE_COUNT='1', NODE_RANK='0', GPUS_PER_WORKER='1')
        variables.update(env)
        return subprocess.run(['bash', str(ROOT/'scripts/launch_annotation.sh')],
                              env=variables, text=True, capture_output=True, timeout=10)

    def test_eight_gpus_eight_workers(self):
        result = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        plans = [line for line in result.stdout.splitlines() if line.startswith('node=')]
        self.assertEqual(len(plans), 8)
        self.assertIn('node=0 shard=7/8 GPUs=7', plans)

    def test_rollout_overrides_reach_every_worker(self):
        result = self.dry_run(ROLLOUTS='4', MIN_ACCEPTED_ROLLOUTS='3',
                              ANNOTATION_SEED='123', PROMPT_PROFILE='temporal_v1')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [x for x in result.stdout.splitlines() if x.startswith('CUDA_VISIBLE_DEVICES=')]
        self.assertEqual(len(commands), 8)
        for command in commands:
            for option in ('--rollouts 4', '--min-accepted-rollouts 3', '--seed 123',
                           '--prompt-profile temporal_v1'):
                self.assertIn(option, command)

    def test_two_nodes_disjoint_sixteen_shards(self):
        left = self.dry_run(NODE_COUNT='2', NODE_RANK='0')
        right = self.dry_run(NODE_COUNT='2', NODE_RANK='1')
        self.assertEqual((left.returncode, right.returncode), (0, 0))
        self.assertIn('shard=0/16 GPUs=0', left.stdout)
        self.assertIn('shard=7/16 GPUs=7', left.stdout)
        self.assertIn('shard=8/16 GPUs=0', right.stdout)
        self.assertIn('shard=15/16 GPUs=7', right.stdout)

    def test_two_gpu_fallback_and_input_validation(self):
        result = self.dry_run(GPUS_PER_WORKER='2')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('shard=3/4 GPUs=6,7',result.stdout)
        self.assertNotEqual(self.dry_run(GPUS_PER_WORKER='3').returncode,0)
        self.assertNotEqual(self.dry_run(NODE_COUNT='2', NODE_RANK='2').returncode,0)
        self.assertNotEqual(self.dry_run(GPU_IDS='0,0').returncode,0)

    def test_decoder_reuses_process_and_recovers_after_timeout(self):
        decoder = DecoderProcess(target=echo_worker)
        try:
            first = decoder.request({},timeout=5)['pid']
            self.assertEqual(first,decoder.request({},timeout=5)['pid'])
            with self.assertRaises(TimeoutError):
                decoder.request({'hang':True},timeout=0.1)
            self.assertIsNone(decoder.process)
            self.assertNotEqual(first,decoder.request({},timeout=5)['pid'])
        finally:
            decoder.close()

    def test_decoder_restarts_after_codec_error(self):
        decoder = DecoderProcess(target=echo_worker)
        try:
            first = decoder.request({},timeout=5)['pid']
            with self.assertRaises(RuntimeError):
                decoder.request({'fail':True},timeout=5)
            self.assertNotEqual(first,decoder.request({},timeout=5)['pid'])
        finally:
            decoder.close()


if __name__ == '__main__':
    unittest.main()

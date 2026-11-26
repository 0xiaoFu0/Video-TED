import json
import os
import subprocess
import unittest
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import test_pipeline as support
from ar_annotate.annotate import annotate, annotate_one
from ar_annotate.common import ROOT, atomic_json, file_hash, read_json, rows, write_rows
from ar_annotate.export import export_data
from ar_annotate.hmdb51 import combine, hmdb_candidates, safe_relative
from ar_annotate.report import summarize


class AllDatasetBackend:
    hierarchy = True
    def __init__(self, cfg=None):
        self.prompts = []

    def load_video(self, path):
        return dict(path=path, audit=dict(frame_indices=[0, 1], timestamps=[0, 1], frame_sha256='fixture'))

    def generate(self, video, question, seed):
        self.prompts.append(question)
        if 'Options:\n' not in question:
            assert 'DO_NOT_COPY_OLD_COT' not in question
            assert video['path'] not in question
            return json.dumps(support.EVIDENCE)
        options = json.loads(question.split('Options:\n')[1].split('\nDraft observations:')[0])
        if '/run/' in video['path']:
            target = 'run'
        elif '/push/' in video['path']:
            target = 'push'
        elif '/hurdling/' in video['path']:
            target = 'hurdling'
        else:
            target = 'pushing'
        choice = next(int(k) for k, v in options.items() if v == target)
        result = dict(choice=choice, evidence_supported=True, reason='Fixture grounding and hierarchy check.')
        if self.hierarchy is not None:
            result['hierarchy_supported'] = self.hierarchy
        return json.dumps(result)


class CombinedTest(unittest.TestCase):
    setUp = support.PipelineTest.setUp
    tearDown = support.PipelineTest.tearDown
    prepared = support.PipelineTest.prepared
    qa = support.PipelineTest.qa
    media = support.PipelineTest.media

    def fixture(self):
        old = self.prepared()
        hmdb = self.root / 'hmdb'
        values = []
        for i, cls in enumerate(('run', 'push')):
            path = hmdb / cls / f'{i}.avi'
            self.media(path)
            values.append(dict(answer=cls, videos=[str(path)], reference_response='DO_NOT_COPY_OLD_COT',
                               metadata=dict(sample_id=f'vs_{i}', video_relpath=f'{cls}/{i}.avi')))
        source = self.root / 'hmdb.jsonl'
        write_rows(source, values)
        catalog = self.root / 'membership.json'
        atomic_json(catalog, dict(sources=[dict(benchmark='hmdb51-split2', sha256='fixture')],
                                 membership={'run/0.avi': ['hmdb51-split2']}))
        self.cfg.update(hmdb51_source=str(source), hmdb51_root=str(hmdb),
                        hmdb51_expected_classes=2, hmdb51_eval_membership=str(catalog))
        self.cfg['annotation'].update(prompt_profile='tree_v1', rollouts=4)
        atomic_json(self.config, self.cfg)
        return old, values

    def test_hmdb_candidates_preserve_provenance_not_old_cot(self):
        old, values = self.fixture()
        accepted, rejected, report = hmdb_candidates(self.cfg)
        self.assertEqual(len(accepted), 2)
        self.assertFalse(rejected)
        self.assertEqual(report['overlap_counts'], {'hmdb51-split2': 1})
        self.assertEqual(accepted[0]['evaluation_overlaps'], ['hmdb51-split2'])
        self.assertEqual(accepted[0]['source_sample_id'], 'vs_0')
        self.assertNotIn('DO_NOT_COPY_OLD_COT', json.dumps(accepted))
        self.assertEqual(set(accepted[0]['options']), {'run', 'push'})
        # Reordering the source does not change media identity.
        reverse = self.root / 'reverse.jsonl'
        write_rows(reverse, list(reversed(values)))
        changed, _, _ = hmdb_candidates(dict(self.cfg, hmdb51_source=str(reverse)))
        self.assertEqual({r['sample_id'] for r in accepted}, {r['sample_id'] for r in changed})

    def test_missing_and_conflicting_hmdb_media_are_reported(self):
        old, values = self.fixture()
        from pathlib import Path
        Path(values[1]['videos'][0]).unlink()
        accepted, rejected, report = hmdb_candidates(self.cfg)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]['reason'], 'hmdb51_media_missing_or_empty')
        values.append(dict(values[0], answer='push', metadata=dict(values[0]['metadata'], sample_id='conflict')))
        source = self.root / 'conflict.jsonl'
        write_rows(source, values)
        accepted, rejected, _ = hmdb_candidates(dict(self.cfg, hmdb51_source=str(source)))
        self.assertFalse(accepted)
        self.assertEqual(Counter(r['reason'] for r in rejected)['conflicting_source_labels'], 2)
        for rel in ('../x.avi', '/tmp/x.avi', 'run/../../x.avi', 'x.avi'):
            with self.assertRaises(ValueError): safe_relative(rel)

    def test_combination_preserves_existing_rows_and_rejects_changed_inputs(self):
        old, _ = self.fixture()
        out = self.root / 'combined'
        combine(self.config, old, out)
        combined = [r for _, r in rows(out / 'candidates.jsonl')]
        self.assertEqual(Counter(r['dataset'] for r in combined), {'ssv2': 1, 'k710': 1, 'hmdb51': 2})
        existing = {r['sample_id']: r for _, r in rows(old)}
        self.assertEqual({r['sample_id']: r for r in combined if r['dataset'] != 'hmdb51'}, existing)
        with self.assertRaises(FileExistsError): combine(self.config, old, out)
        old.write_text(old.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            combine(self.config, old, self.root / 'changed')

    def test_tree_gate_rejects_hierarchy_defects_and_missing_field(self):
        self.fixture()
        candidate = hmdb_candidates(self.cfg)[0][0]
        fake = AllDatasetBackend()
        fake.hierarchy = False
        record = annotate_one(candidate, fake, self.cfg['annotation'])
        self.assertEqual(record['status'], 'needs_review')
        self.assertEqual(len(fake.prompts), 2)
        fake = AllDatasetBackend()
        fake.hierarchy = None
        record = annotate_one(candidate, fake, self.cfg['annotation'])
        self.assertEqual(record['status'], 'error')

    def test_sixteen_shards_and_pool_export_without_appending_old_hmdb(self):
        old, _ = self.fixture()
        prepared, run = self.root / 'combined', self.root / 'run'
        combine(self.config, old, prepared)
        source = prepared / 'candidates.jsonl'
        before = file_hash(self.cfg['hmdb51_source'])
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=16,
                 limit_per_dataset=0, candidates=str(source), out=str(run), retry_errors=False)
        with ExitStack() as stack:
            stack.enter_context(patch('ar_annotate.annotate.model_identity', return_value={'fake': True}))
            stack.enter_context(patch('ar_annotate.annotate.environment', return_value={}))
            stack.enter_context(patch('ar_annotate.annotate.QwenBackend', AllDatasetBackend))
            for i in range(16):
                args.shard_index = i
                annotate(args)
        self.assertEqual(sum(read_json(p)['assigned'] for p in run.glob('summary-*.json')), 4)
        self.assertEqual(len(list(run.glob('rollouts/*/*.json'))), 16)
        progress = summarize(run, source)
        self.assertTrue(progress['complete'])
        self.assertEqual(progress['datasets']['hmdb51']['accepted'], 2)
        with self.assertRaisesRegex(ValueError, '--pool'):
            export_data(source, run, self.cfg['hmdb51_source'], self.root / 'bad_export', lambda s: 1)
        out = self.root / 'pool'
        report = export_data(source, run, None, out, lambda s: len(s.split()), pool_only=True)
        self.assertEqual((report['base_rows'], report['total_rows']), (0, 4))
        self.assertEqual(report['evaluation_overlap_counts'], {'hmdb51-split2': 1})
        rlsd = [r for _, r in rows(out / 'pool.rlsd.think.jsonl')]
        sft = [r for _, r in rows(out / 'pool.sft.think.jsonl')]
        for r, s in zip(rlsd, sft):
            self.assertEqual(r['metadata'], s['metadata'])
            self.assertEqual(r['reference_response'], s['messages'][1]['content'])
        self.assertEqual(len({r['metadata']['sample_id'] for r in rlsd}), 4)
        self.assertEqual(file_hash(self.cfg['hmdb51_source']), before)
        (run / 'summary-15.json').unlink()
        self.assertEqual(summarize(run, source)['missing_shards'], [15])
        with self.assertRaisesRegex(ValueError, 'global shards'):
            export_data(source, run, None, self.root / 'incomplete', lambda s: 1, pool_only=True)

    def test_node_wrapper_assigns_disjoint_global_shards(self):
        plans = []
        for rank in (0, 1):
            result = subprocess.run(['bash', str(ROOT / 'scripts/launch_16gpu_node.sh'), str(rank)],
                env=dict(os.environ, DRY_RUN='1', GPU_IDS='0,1,2,3,4,5,6,7', MIN_ACCEPTED_ROLLOUTS=''),
                text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            plans.extend(x for x in result.stdout.splitlines() if x.startswith('node='))
            self.assertIn('--prompt-profile tree_v1', result.stdout)
            self.assertIn('--min-accepted-rollouts 1', result.stdout)
        self.assertEqual(len(plans), 16)
        self.assertIn('node=0 shard=0/16 GPUs=0', plans)
        self.assertIn('node=1 shard=15/16 GPUs=7', plans)


if __name__ == '__main__':
    unittest.main()

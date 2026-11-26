import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from ar_annotate.annotate import annotate, annotate_one, annotate_rollouts, evidence_prompt
from ar_annotate.common import atomic_json, file_hash, read_json, rows
from ar_annotate.export import export_data
from ar_annotate.rollouts import generation_seed, settings, validate_aggregate
from test_pipeline import FakeBackend
import test_pipeline as support


class ObservedBackend(FakeBackend):
    def __init__(self, cfg=None, decisions=None):
        super().__init__(cfg)
        self.decisions = iter(decisions or [])
        self.loads, self.seeds, self.closed = 0, [], False

    def load_video(self, path):
        self.loads += 1
        return super().load_video(path)

    def generate(self, video, question, seed):
        self.seeds.append(seed)
        raw = super().generate(video, question, seed)
        if 'Options:\n' in question:
            result = json.loads(raw)
            if next(self.decisions, True) is False:
                result['choice'] = -1
            return json.dumps(result)
        return raw

    def close(self):
        self.closed = True


class RolloutTest(unittest.TestCase):
    setUp = support.PipelineTest.setUp
    tearDown = support.PipelineTest.tearDown
    prepared = support.PipelineTest.prepared
    media = support.PipelineTest.media
    qa = support.PipelineTest.qa
    setup_export = support.PipelineTest.setup_export

    def fixture(self):
        row = next(rows(self.prepared()))[1]
        cfg = dict(self.cfg['annotation'], rollouts=4, min_accepted_rollouts=3)
        return row, cfg, self.root / 'multi'

    def test_fixed_budget_keeps_disagreements_and_decodes_once(self):
        row, cfg, out = self.fixture()
        backend = ObservedBackend(decisions=[True, False, True, False])
        record = annotate_rollouts(row, lambda: backend, cfg, out, 'run')
        self.assertEqual(record['status'], 'needs_review')
        self.assertEqual(record['rollout_summary']['accepted'], 2)
        self.assertEqual(record['rollout_summary']['minimum_accepted'], 3)
        self.assertEqual(backend.loads, 1)
        self.assertEqual(len(backend.prompts), 8)
        self.assertEqual(len(set(backend.seeds)), 8)
        self.assertEqual(len(list((out / 'rollouts' / row['sample_id']).glob('*.json'))), 4)
        # Completed disagreements are not retried on resume, even with retry-errors.
        resumed = annotate_rollouts(row, lambda: self.fail('Unexpected inference'), cfg, out, 'run', True)
        self.assertEqual(record, resumed)

    def test_three_of_four_pass_and_representative_is_first_accepted(self):
        row, cfg, out = self.fixture()
        backend = ObservedBackend(decisions=[False, True, True, True])
        record = annotate_rollouts(row, lambda: backend, cfg, out, 'run')
        self.assertEqual(record['status'], 'accepted')
        self.assertEqual(record['rollout_summary']['selected_rollout'], 1)
        self.assertEqual(record['rollout_summary']['accepted_indices'], [1, 2, 3])
        validate_aggregate(record, row, cfg, 'run', out)

    def test_one_pass_is_kept_without_semantic_retries(self):
        row, cfg, out = self.fixture()
        cfg['min_accepted_rollouts'] = 1
        backend = ObservedBackend(decisions=[False, False, True, False])
        record = annotate_rollouts(row, lambda: backend, cfg, out, 'run')
        self.assertEqual(record['status'], 'accepted')
        self.assertEqual(record['rollout_summary']['selected_rollout'], 2)
        self.assertEqual(record['rollout_summary']['accepted'], 1)
        self.assertEqual(record['rollout_summary']['needs_review'], 3)
        self.assertEqual(len(backend.prompts), 8)
        self.assertEqual(record, annotate_rollouts(row, lambda: self.fail('Unexpected rerun'), cfg, out, 'run'))

    def test_one_pass_three_errors_can_export_verified_representative(self):
        source, _, base, candidates = self.setup_export()
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
            limit_per_dataset=0, candidates=str(source), out=str(self.root / 'anypass'), retry_errors=False,
            rollouts=4, min_accepted_rollouts=1)
        def partially_failing(row, backend, cfg, index):
            return annotate_one(row, backend if index == 1 else FakeBackend(malformed=True), cfg, index)
        with ExitStack() as stack:
            stack.enter_context(patch('ar_annotate.annotate.model_identity', return_value={'fake': True}))
            stack.enter_context(patch('ar_annotate.annotate.environment', return_value={}))
            stack.enter_context(patch('ar_annotate.annotate.QwenBackend', FakeBackend))
            stack.enter_context(patch('ar_annotate.annotate.annotate_one', side_effect=partially_failing))
            annotate(args)
        run = self.root / 'anypass'
        for row in candidates:
            record = read_json(run / 'records' / (row['sample_id'] + '.json'))
            self.assertEqual(record['status'], 'accepted')
            self.assertNotIn('error', record)
            self.assertIn('rollout_warning', record)
            self.assertEqual(record['rollout_summary']['error'], 3)
            self.assertEqual(record['rollout_summary']['selected_rollout'], 1)
        from ar_annotate.report import summarize
        report = summarize(run, source)
        self.assertTrue(report['complete'])
        self.assertEqual(sum(c['error'] for c in report['rollouts'].values()), 6)
        result = export_data(source, run, base, self.root / 'export_anypass', lambda s: 1)
        self.assertEqual(result['added_rows'], 2)
        saved = [r for _, r in rows(self.root / 'export_anypass/train.rlsd.think.jsonl')]
        self.assertEqual(json.loads(saved[-1]['metadata']['annotation_rollout_summary_json'])['error'], 3)

    def test_no_pass_is_never_accepted(self):
        row, cfg, out = self.fixture()
        cfg['min_accepted_rollouts'] = 1
        result = annotate_rollouts(row, lambda: ObservedBackend(decisions=[False]*4), cfg, out, 'run')
        self.assertEqual(result['status'], 'needs_review')
        other = annotate_rollouts(row, lambda: FakeBackend(malformed=True), cfg, self.root/'all_errors', 'run')
        self.assertEqual(other['status'], 'error')

    def test_interrupt_mid_video_resumes_only_missing_draws(self):
        row, cfg, out = self.fixture()
        backend = ObservedBackend()
        def interrupted(row, backend, cfg, index):
            if index == 2:
                raise KeyboardInterrupt()
            return annotate_one(row, backend, cfg, index)
        with patch('ar_annotate.annotate.annotate_one', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                annotate_rollouts(row, lambda: backend, cfg, out, 'run')
        saved = {p: file_hash(p) for p in (out / 'rollouts' / row['sample_id']).glob('*.json')}
        self.assertEqual(len(saved), 2)
        fresh = ObservedBackend()
        record = annotate_rollouts(row, lambda: fresh, cfg, out, 'run')
        self.assertEqual(len(fresh.prompts), 4)
        self.assertEqual(saved, {p: file_hash(p) for p in saved})
        self.assertEqual(record['status'], 'accepted')

    def test_retry_only_technical_errors_and_archive(self):
        row, cfg, out = self.fixture()
        backend = ObservedBackend(decisions=[False, True, True])
        def failing(row, cached, cfg, index):
            return annotate_one(row, FakeBackend(malformed=True) if index == 1 else cached, cfg, index)
        with patch('ar_annotate.annotate.annotate_one', side_effect=failing):
            record = annotate_rollouts(row, lambda: backend, cfg, out, 'run')
        self.assertEqual(record['status'], 'error')
        saved = {p: file_hash(p) for p in (out / 'rollouts' / row['sample_id']).glob('*.json') if p.stem != '0001'}
        unchanged = annotate_rollouts(row, lambda: self.fail('Unexpected retry'), cfg, out, 'run')
        self.assertEqual(record, unchanged)
        fresh = ObservedBackend()
        retried = annotate_rollouts(row, lambda: fresh, cfg, out, 'run', True)
        self.assertEqual(retried['status'], 'accepted')
        self.assertEqual(len(fresh.prompts), 2)
        self.assertEqual(saved, {p: file_hash(p) for p in saved})
        self.assertEqual(len(list((out / 'errors_history/rollouts' / row['sample_id']).glob('*.json'))), 1)

    def test_changed_checkpoint_or_sampled_frames_rejected(self):
        row, cfg, out = self.fixture()
        record = annotate_rollouts(row, ObservedBackend, cfg, out, 'run')
        target = out / 'rollouts' / row['sample_id'] / '0002.json'
        draw = read_json(target)
        draw['sampling']['frame_sha256'] = 'different input'
        atomic_json(target, draw)
        with self.assertRaisesRegex(ValueError, 'sampled frames'):
            validate_aggregate(record, row, cfg, 'run', out)
        draw['run_hash'] = 'other run'
        atomic_json(target, draw)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            annotate_rollouts(row, ObservedBackend, cfg, out, 'run')

    def test_prompt_profiles_are_blind_and_legacy_preserved(self):
        row, cfg, out = self.fixture()
        cfg['prompt_profile'] = 'temporal_v1'
        result = annotate_one(row, FakeBackend(), cfg)
        self.assertEqual(result['status'], 'accepted')
        prompt = evidence_prompt(row, cfg)
        self.assertNotIn(row['video'], prompt)
        self.assertNotIn(row['original_video'], prompt)
        self.assertNotIn(row['answer'], prompt)
        self.assertNotEqual(prompt, evidence_prompt(row, dict(cfg, prompt_profile='legacy')))

    def test_settings_and_seed_validation(self):
        self.assertEqual(settings({'rollouts': 4}), (4, 1, 0))
        self.assertEqual(settings({}), (1, 1, 0))
        for cfg in ({'rollouts': 0}, {'rollouts': True}, {'rollouts': 2, 'min_accepted_rollouts': 3},
                    {'rollouts': 2, 'min_accepted_rollouts': 0}, {'seed': -1}):
            with self.assertRaises(ValueError): settings(cfg)
        self.assertEqual(generation_seed('id', {}, 1, 0, 'evidence'),
                         generation_seed('id', {'seed': 0}, 1, 0, 'evidence'))

    def test_format_feedback_recovers_missing_field_then_excess_observations(self):
        row, cfg, _ = self.fixture()
        cfg['attempts'] = 3
        class MalformedThenValid(FakeBackend):
            evidence_calls = 0
            def generate(self, video, question, seed):
                if 'Options:\n' not in question:
                    self.evidence_calls += 1
                    if self.evidence_calls <= 2:
                        self.prompts.append(question)
                        data = json.loads(json.dumps(support.EVIDENCE))
                        if self.evidence_calls == 1:
                            data.pop('sufficient_evidence')
                        else:
                            data['branches'] = [dict(name=f'Group {i}', observations=['Visible motion.'] * 3)
                                                for i in range(3)]
                        return json.dumps(data)
                return super().generate(video, question, seed)
        backend = MalformedThenValid()
        result = annotate_one(row, backend, cfg)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(len(backend.prompts), 4)
        self.assertIn('Missing required evidence fields: sufficient_evidence', backend.prompts[1])
        self.assertIn('received 9', backend.prompts[2])
        self.assertEqual(result['attempts'][0]['error_stage'], 'evidence')
        self.assertEqual(result['attempts'][1]['error_stage'], 'evidence')

    def test_verification_format_retry_reuses_valid_evidence(self):
        row, cfg, _ = self.fixture()
        class BadVerifierOnce(FakeBackend):
            verification_calls = 0
            def generate(self, video, question, seed):
                if 'Options:\n' in question:
                    self.verification_calls += 1
                    if self.verification_calls == 1:
                        self.prompts.append(question)
                        return '{}'
                return super().generate(video, question, seed)
        backend = BadVerifierOnce()
        result = annotate_one(row, backend, cfg)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(len(backend.prompts), 3)
        self.assertTrue(result['attempts'][1]['evidence_reused'])
        self.assertEqual(result['attempts'][0]['error_stage'], 'verification')
        self.assertIn('FORMAT RETRY', backend.prompts[2])

    def test_sample_id_selection_does_not_rewrite_candidate_manifest(self):
        source = self.prepared()
        selected = next(rows(source))[1]['sample_id']
        original = file_hash(source)
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
            limit_per_dataset=0, candidates=str(source), out=str(self.root / 'isolated'), retry_errors=False,
            sample_id=[selected])
        with ExitStack() as stack:
            stack.enter_context(patch('ar_annotate.annotate.model_identity', return_value={'fake': True}))
            stack.enter_context(patch('ar_annotate.annotate.environment', return_value={}))
            stack.enter_context(patch('ar_annotate.annotate.QwenBackend', FakeBackend))
            annotate(args)
            self.assertEqual(read_json(self.root / 'isolated/run.json')['spec']['selected_ids'], [selected])
            self.assertEqual(len(list((self.root / 'isolated/records').glob('*.json'))), 1)
            args.sample_id = ['nonexistent']
            with self.assertRaisesRegex(ValueError, 'Unknown sample IDs'):
                annotate(args)
        self.assertEqual(file_hash(source), original)

    def test_interrupted_error_retry_rebuilds_stale_aggregate_and_closes_backend(self):
        source = self.prepared()
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
            limit_per_dataset=0, candidates=str(source), out=str(self.root / 'v3'),
            retry_errors=False, rollouts=4)
        backends = []
        def factory(cfg):
            backend = ObservedBackend(cfg)
            backends.append(backend)
            return backend
        def interrupted(row, backend, cfg, index):
            if index == 1:
                raise KeyboardInterrupt()
            return annotate_one(row, backend, cfg, index)
        with ExitStack() as stack:
            stack.enter_context(patch('ar_annotate.annotate.model_identity', return_value={'fake': True}))
            stack.enter_context(patch('ar_annotate.annotate.environment', return_value={}))
            with patch('ar_annotate.annotate.QwenBackend', side_effect=lambda cfg: FakeBackend(malformed=True)):
                with self.assertRaises(RuntimeError): annotate(args)
            args.retry_errors = True
            with patch('ar_annotate.annotate.QwenBackend', side_effect=factory):
                with patch('ar_annotate.annotate.annotate_one', side_effect=interrupted):
                    with self.assertRaises(KeyboardInterrupt): annotate(args)
                self.assertTrue(backends[-1].closed)
                annotate(args)
                self.assertTrue(backends[-1].closed)
                self.assertEqual(len(backends[-1].prompts), 14)  # Seven missing/error draws.
        records = list((self.root / 'v3/records').glob('*.json'))
        self.assertEqual(len(records), 2)
        self.assertTrue(all(read_json(p)['status'] == 'accepted' for p in records))

    def test_full_pipeline_export_one_row_per_video_and_integrity(self):
        source, old_run, base, candidates = self.setup_export()
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
            limit_per_dataset=0, candidates=str(source), out=str(self.root / 'v3'), retry_errors=False,
            rollouts=4, prompt_profile='temporal_v1')
        with ExitStack() as stack:
            stack.enter_context(patch('ar_annotate.annotate.model_identity', return_value={'fake': True}))
            stack.enter_context(patch('ar_annotate.annotate.environment', return_value={}))
            factory = stack.enter_context(patch('ar_annotate.annotate.QwenBackend', side_effect=ObservedBackend))
            annotate(args)
            before = factory.call_count
            annotate(args)
            self.assertEqual(factory.call_count, before)
            args.rollouts = 3
            with self.assertRaisesRegex(ValueError, 'NEW output'): annotate(args)
        run = self.root / 'v3'
        report = export_data(source, run, base, self.root / 'export', lambda s: len(s.split()))
        self.assertEqual(report['added_rows'], 2)  # Never eight duplicates.
        rl = [r for _, r in rows(self.root / 'export/train.rlsd.think.jsonl')]
        sft = [r for _, r in rows(self.root / 'export/train.sft.think.jsonl')]
        for r, s in zip(rl, sft):
            self.assertEqual(r['reference_response'], s['messages'][1]['content'])
            self.assertEqual(r['metadata'], s['metadata'])
        self.assertEqual(json.loads(rl[-1]['metadata']['annotation_rollout_summary_json'])['total'], 4)
        target = run / 'rollouts' / candidates[0]['sample_id'] / '0003.json'
        draw = read_json(target)
        draw['verdict']['reason'] = 'Edited after aggregation'
        atomic_json(target, draw)
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            export_data(source, run, base, self.root / 'tampered', lambda s: 1)
        self.assertFalse((self.root / 'tampered').exists())


if __name__ == '__main__':
    unittest.main()

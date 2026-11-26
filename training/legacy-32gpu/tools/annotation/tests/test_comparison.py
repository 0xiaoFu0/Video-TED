import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_pipeline as support
from ar_annotate.annotate import annotate
from ar_annotate.common import atomic_json, digest, file_hash, read_json, rows, write_rows
from ar_annotate.comparison import build_comparison


class ComparisonBackend(support.FakeBackend):
    def load_video(self, path):
        return dict(super().load_video(path), path=path)

    def generate(self, video, question, seed):
        if video['path'].endswith('124.webm'):
            return '{broken'
        if '/kin/' in video['path']:
            if 'Options:\n' not in question:
                evidence = dict(support.EVIDENCE, activity='UNCHECKED_GENERATED_TREE',
                                uncertainties=['The movement is ambiguous.'], sufficient_evidence=False)
                return json.dumps(evidence)
            options = json.loads(question.split('Options:\n')[1].split('\nDraft observations:')[0])
            choice = next(int(k) for k, v in options.items() if v == 'pulling')
            return json.dumps(dict(choice=choice, evidence_supported=False,
                                   hierarchy_supported=False, reason='Ambiguous unsupported alternative.'))
        return super().generate(video, question, seed)


class ComparisonTest(unittest.TestCase):
    setUp = support.PipelineTest.setUp
    tearDown = support.PipelineTest.tearDown
    prepared = support.PipelineTest.prepared
    qa = support.PipelineTest.qa
    media = support.PipelineTest.media

    def fixture(self):
        self.media(self.ss / '123.webm')
        self.media(self.ss / '124.webm')
        self.media(self.kin / 'train/hurdling/abcdefghijk_000001_000011.mp4')
        write_rows(self.source, [self.qa('ssv2', '123.webm'), self.qa('ssv2', '124.webm'),
                    self.qa('k710', 'abcdefghijk_000001_000011.mp4', 'hurdling')])
        from ar_annotate.prepare import prepare
        directory = self.root / 'prepared'
        prepare(self.config, directory)
        source = directory / 'candidates.jsonl'
        values = [r for _, r in rows(source)]
        for row in values:
            row['evaluation_overlaps'] = ['fixture-evaluation-overlap']
            row['candidate_hash'] = digest({k: v for k, v in row.items() if k != 'candidate_hash'})
        source.unlink()
        write_rows(source, values)
        manifest = read_json(directory / 'manifest.json')
        manifest['candidates_sha256'] = file_hash(source)
        atomic_json(directory / 'manifest.json', manifest)
        self.cfg['annotation'].update(rollouts=4, min_accepted_rollouts=1, prompt_profile='tree_v1')
        atomic_json(self.config, self.cfg)
        run = self.root / 'run'
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
                   limit_per_dataset=0, candidates=str(source), out=str(run), retry_errors=False)
        with patch('ar_annotate.annotate.model_identity', return_value={'fake': True}), \
             patch('ar_annotate.annotate.environment', return_value={}), \
             patch('ar_annotate.annotate.QwenBackend', ComparisonBackend):
            with self.assertRaisesRegex(RuntimeError, 'failed samples'):
                annotate(args)
        return source, run

    def test_full_routes_retain_unaccepted_trees_and_keep_overlaps(self):
        source, run = self.fixture()
        before = {str(p): file_hash(p) for p in run.rglob('*.json')}
        out = self.root / 'comparison'
        result = build_comparison(source, run, out, lambda s: 2400 if 'UNCHECKED_GENERATED_TREE' in s else 1200)
        self.assertEqual((result['qualified_rows'], result['full_rows']), (1, 2))
        self.assertEqual(result['statuses'], dict(accepted=1, error=1, needs_review=1))
        self.assertEqual(result['reference_max_length'], 2401)
        self.assertEqual(result['unaccepted_sft_rows'], 1)
        self.assertEqual(result['full_reference_policy'], 'all_nonerror_generated_trees_with_reference_loss')
        self.assertEqual(result['reference_answer_policy'], 'original_qa_label')
        self.assertEqual(result['evaluation_overlap_counts'], {'fixture-evaluation-overlap': 2})
        qualified = [r for _, r in rows(out / 'qualified/train.rlsd.think.jsonl')]
        full = [r for _, r in rows(out / 'full/train.rlsd.think.jsonl')]
        self.assertEqual([r for r in full if r['metadata']['annotation_status'] == 'accepted'], qualified)
        self.assertEqual([r['reference_status'] for r in full], ['unreviewed'] * 2)
        review = next(r for r in full if r['metadata']['annotation_status'] == 'needs_review')
        self.assertIn('UNCHECKED_GENERATED_TREE', review['reference_response'])
        self.assertIn('The movement is ambiguous.', review['reference_response'])
        self.assertTrue(review['reference_response'].endswith('<answer>hurdling</answer>'))
        self.assertIn('Reference class: hurdling', review['teacher_context'])
        self.assertIn('UNCHECKED_GENERATED_TREE', review['teacher_context'])
        self.assertEqual(review['metadata']['reference_tokens_including_eos'], 2401)
        self.assertEqual(review['metadata']['evidence_status'], 'retained_unaccepted_generated_annotation')
        evidence = json.loads(review['metadata']['annotation_evidence_json'])
        self.assertFalse(evidence['sufficient_evidence'])
        verdict = json.loads(review['metadata']['annotation_verdict_json'])
        options = json.loads(review['metadata']['annotation_verification_options_json'])
        self.assertEqual(options[verdict['choice']], 'pulling')
        self.assertFalse(verdict['evidence_supported'])
        self.assertFalse(verdict['hierarchy_supported'])
        audit = [r for _, r in rows(out / 'selection.audit.jsonl')]
        review_audit = next(r for r in audit if r['status'] == 'needs_review')
        self.assertEqual(review_audit['verdict'], verdict)
        self.assertEqual(review_audit['reference_route'], 'unaccepted_tree')
        self.assertEqual(review_audit['selected_rollout'], 0)
        for name, rlsd in (('qualified', qualified), ('full', full)):
            sft = [r for _, r in rows(out / name / 'train.sft.think.jsonl')]
            self.assertEqual(len(sft), len(rlsd))
            for pair, reference in zip(sft, rlsd):
                self.assertEqual(pair['messages'], [dict(role='user', content=reference['prompt']),
                                                   dict(role='assistant', content=reference['reference_response'])])
                self.assertEqual(pair['answer'], reference['answer'])
                self.assertEqual(pair['videos'], reference['videos'])
                self.assertEqual(pair['metadata'], reference['metadata'])
        from ar_annotate.common import hierarchy
        self.assertEqual(review['reference_response'], '<think>\n' + hierarchy(evidence) +
                         '\n</think>\n<answer>hurdling</answer>')
        self.assertEqual(before, {str(p): file_hash(p) for p in run.rglob('*.json')})
        with self.assertRaises(FileExistsError):
            build_comparison(source, run, out, lambda s: 1)

    def test_missing_shard_and_changed_media_fail_before_outputs(self):
        source, run = self.fixture()
        summary = run / 'summary-0.json'
        saved = read_json(summary)
        summary.unlink()
        out = self.root / 'bad'
        with self.assertRaisesRegex(ValueError, 'Incomplete global shards'):
            build_comparison(source, run, out, lambda s: 1)
        self.assertFalse(out.exists())
        atomic_json(summary, saved)
        self.ss.joinpath('123.webm').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Video changed'):
            build_comparison(source, run, out, lambda s: 1)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

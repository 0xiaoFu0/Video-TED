import contextlib
import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_pipeline as support
import test_comparison as comparison_support
from test_comparison import ComparisonBackend
from ar_annotate.annotate import annotate
from ar_annotate.common import atomic_json, digest, file_hash, hierarchy, read_json, rows, write_rows
from ar_annotate.merged_comparison import build_merged_comparison, main
from ar_annotate.prepare import prepare


class IncrementBackend(ComparisonBackend):
    def generate(self, video, question, seed):
        if video['path'].endswith('126.webm'):
            return '{broken'
        if video['path'].endswith('125.webm'):
            if 'Options:\n' not in question:
                return json.dumps(dict(support.EVIDENCE, activity='NEW_PUNCTUATION_TREE'))
            options = json.loads(question.split('Options:\n')[1].split('\nDraft observations:')[0])
            choice = next(int(k) for k, v in options.items() if v == 'touching (without moving) part of something')
            return json.dumps(dict(choice=choice, evidence_supported=True, hierarchy_supported=True,
                                   reason='The original punctuation action label is preserved.'))
        return super().generate(video, question, seed)


class MergedComparisonTest(unittest.TestCase):
    setUp = support.PipelineTest.setUp
    tearDown = support.PipelineTest.tearDown
    qa = support.PipelineTest.qa
    media = support.PipelineTest.media

    def fixture(self, resolved_error_duplicate=False, review_only=False):
        with contextlib.redirect_stdout(io.StringIO()):
            old_candidates, old_run = comparison_support.ComparisonTest.fixture(self)
        new_source, new_config = self.root / 'increment_source.jsonl', self.root / 'increment_config.json'
        self.media(self.ss / '125.webm')
        self.media(self.ss / '126.webm')
        self.media(self.kin / 'train/hurdling/BBBBBBBBBBB_000001_000011.mp4')
        write_rows(new_source, [self.qa('ssv2', '125.webm', 'touching (without moving) part of something'),
                               self.qa('ssv2', '126.webm'),
                               self.qa('k710', 'BBBBBBBBBBB_000001_000011.mp4', 'hurdling')])
        cfg = dict(self.cfg, flash_jsonl=str(new_source))
        atomic_json(new_config, cfg)
        new_dir, new_run = self.root / 'increment_candidates', self.root / 'increment_run'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(new_config, new_dir, retain_evaluation_overlaps=True)
        new_candidates = new_dir / 'candidates.jsonl'
        manifest = read_json(new_dir / 'manifest.json')
        manifest['scope'] = 'all-missing'
        if resolved_error_duplicate:
            values = [r for _, r in rows(new_candidates)]
            row = next(r for r in values if r['video'].endswith('125.webm'))
            row.update(video=str(self.ss / '124.webm'), video_relpath='124.webm',
                       video_size=(self.ss / '124.webm').stat().st_size,
                       video_mtime_ns=(self.ss / '124.webm').stat().st_mtime_ns)
            row['candidate_hash'] = digest({k: v for k, v in row.items() if k != 'candidate_hash'})
            new_candidates.unlink()
            write_rows(new_candidates, values)
            manifest['candidates_sha256'] = file_hash(new_candidates)
        atomic_json(new_dir / 'manifest.json', manifest)
        args = SimpleNamespace(config=str(new_config), model=None, shard_index=0, num_shards=1,
            limit_per_dataset=0, candidates=str(new_candidates), out=str(new_run), retry_errors=False)
        backend = IncrementBackend
        if review_only:
            class ReviewBackend(IncrementBackend):
                def generate(self, video, question, seed):
                    raw = super().generate(video, question, seed)
                    if video['path'].endswith('125.webm') and 'Options:\n' in question:
                        value = json.loads(raw)
                        value.update(choice=-1, evidence_supported=False, hierarchy_supported=False)
                        return json.dumps(value)
                    return raw
            backend = ReviewBackend
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('ar_annotate.annotate.model_identity', return_value={'fake': True}), \
             patch('ar_annotate.annotate.environment', return_value={}), \
             patch('ar_annotate.annotate.QwenBackend', backend):
            with self.assertRaisesRegex(RuntimeError, 'failed samples'):
                annotate(args)
        return [('old', old_candidates, old_run), ('incremental', new_candidates, new_run)]

    def merge(self, pools, out=None, **kwargs):
        return build_merged_comparison(pools, out or self.root / 'merged',
            lambda text: 3000 if 'NEW_PUNCTUATION_TREE' in text else
                         (2400 if 'UNCHECKED_GENERATED_TREE' in text else 1200), **kwargs)

    def test_two_pools_keep_all_nonerror_trees_and_pair_sft_rlsd(self):
        pools = self.fixture()
        immutable = {p: file_hash(p) for p in self.root.rglob('*') if p.is_file()}
        out = self.root / 'merged'
        report = self.merge(pools, out, expect_qualified=2, expect_full=4, expect_planned=6,
                            student_model='fixture_student')
        self.assertEqual(report['merge_schema_version'], 1)
        self.assertEqual(report['statuses'], dict(accepted=2, error=2, needs_review=2))
        self.assertEqual((report['qualified_rows'], report['full_rows'], report['planned']), (2, 4, 6))
        self.assertEqual(report['source_counts'], {
            name: dict(planned=3, statuses=dict(accepted=1, error=1, needs_review=1),
                       qualified_rows=1, full_rows=2) for name in ('old', 'incremental')})
        self.assertEqual(report['reference_max_length'], 3001)
        self.assertEqual(report['max_observed_reference_tokens'], 3001)
        self.assertEqual(report['unaccepted_sft_rows'], 2)
        self.assertTrue(report['earlier_preparation_exclusions_restored'])
        self.assertEqual(report['evaluation_overlap_counts'], {'fixture-evaluation-overlap': 2})
        self.assertEqual(report['full_reference_policy'], 'all_nonerror_generated_trees_with_reference_loss')
        self.assertEqual(report['error_policy'], 'omit_without_retry')
        groups = {}
        for name in ('qualified', 'full'):
            rlsd = [r for _, r in rows(out / name / 'train.rlsd.think.jsonl')]
            sft = [r for _, r in rows(out / name / 'train.sft.think.jsonl')]
            self.assertEqual(len(rlsd), len(sft))
            self.assertEqual(read_json(out / name / 'sample_ids.json'), [r['metadata']['sample_id'] for r in rlsd])
            for row, pair in zip(rlsd, sft):
                self.assertEqual(row['reference_status'], 'unreviewed')
                self.assertEqual(pair, dict(messages=[dict(role='user', content=row['prompt']),
                    dict(role='assistant', content=row['reference_response'])], videos=row['videos'],
                    answer=row['answer'], metadata=row['metadata']))
            groups[name] = rlsd
        self.assertEqual(groups['qualified'], [r for r in groups['full'] if r['metadata']['annotation_status'] == 'accepted'])
        for row in groups['full']:
            source = next(p for p in pools if p[0] == row['metadata']['source_pool'])
            sid = row['metadata']['sample_id']
            candidate = next(r for _, r in rows(source[1]) if r['sample_id'] == sid)
            record_path = source[2] / 'records' / (sid + '.json')
            record = read_json(record_path)
            self.assertEqual(row['answer'], candidate['answer'])
            self.assertEqual(row['reference_response'], '<think>\n' + hierarchy(record['evidence']) +
                             '\n</think>\n<answer>' + candidate['answer'] + '</answer>')
            self.assertEqual(row['metadata']['annotation_candidate_hash'], candidate['candidate_hash'])
            self.assertEqual(row['metadata']['annotation_run_hash'], record['run_hash'])
            self.assertEqual(row['metadata']['annotation_record_sha256'], file_hash(record_path))
            self.assertEqual(json.loads(row['metadata']['annotation_evidence_json']), record['evidence'])
            if row['metadata']['annotation_status'] == 'needs_review':
                self.assertIn('UNCHECKED_GENERATED_TREE', row['reference_response'])
                self.assertFalse(json.loads(row['metadata']['annotation_verdict_json'])['evidence_supported'])
        punct = next(r for r in groups['full'] if r['metadata']['raw_label'].startswith('touching ('))
        self.assertEqual(punct['answer'], 'touching_(without_moving)_part_of_something')
        audit = [r for _, r in rows(out / 'selection.audit.jsonl')]
        self.assertEqual(sum(r['reference_route'] == 'omitted_error' for r in audit), 2)
        self.assertTrue(all(len(r['rollout_files_sha256']) == 4 for r in audit))
        for relative, sha in report['output_sha256'].items():
            self.assertEqual(file_hash(out / relative), sha)
        checksums = dict(line.split('  ', 1)[::-1] for line in (out / 'SHA256SUMS').read_text().splitlines())
        self.assertEqual(set(checksums), set(report['output_sha256']) | {'manifest.json'})
        for relative, sha in checksums.items():
            self.assertEqual(file_hash(out / relative), sha)
        self.assertEqual(immutable, {p: file_hash(p) for p in immutable})

    def test_pool_without_accepted_can_join_global_qualified_group(self):
        report = self.merge(self.fixture(review_only=True), expect_qualified=1, expect_full=4, expect_planned=6)
        self.assertEqual(report['source_counts']['incremental']['qualified_rows'], 0)
        self.assertEqual(report['statuses'], dict(accepted=1, needs_review=3, error=2))

    def test_duplicate_selected_ids_rejected_without_output(self):
        pools = self.fixture()
        pools[1] = ('incremental', pools[0][1], pools[0][2])
        with self.assertRaisesRegex(ValueError, 'Duplicate selected sample_id across pools'):
            self.merge(pools)
        self.assertFalse((self.root / 'merged').exists())

    def test_duplicate_resolved_media_including_error_candidates_rejected(self):
        pools = self.fixture(resolved_error_duplicate=True)
        new_candidates, new_run = pools[1][1:]
        candidate = next(r for _, r in rows(new_candidates) if r['video'].endswith('124.webm'))
        self.assertEqual(read_json(new_run / 'records' / (candidate['sample_id'] + '.json'))['status'], 'error')
        with self.assertRaisesRegex(ValueError, 'Duplicate selected resolved media across pools'):
            self.merge(pools)
        self.assertFalse((self.root / 'merged').exists())

    def test_missing_record_or_rollout_blocks_export_even_for_error(self):
        pools = self.fixture()
        source, run = pools[0][1:]
        candidate = next(r for _, r in rows(source) if r['video'].endswith('124.webm'))
        paths = [run / 'records' / (candidate['sample_id'] + '.json'),
                 run / 'rollouts' / candidate['sample_id'] / '0001.json']
        for index, path in enumerate(paths):
            saved = path.read_bytes()
            path.unlink()
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                self.merge(pools, self.root / f'missing_{index}')
            self.assertFalse((self.root / f'missing_{index}').exists())
            path.write_bytes(saved)

    def test_candidate_and_run_provenance_mismatches_block_export(self):
        pools = self.fixture()
        source, run = pools[0][1:]
        candidate_bytes = source.read_bytes()
        source.write_bytes(candidate_bytes + b'\n')
        with self.assertRaisesRegex(ValueError, 'preparation manifest'):
            self.merge(pools, self.root / 'changed_candidates')
        self.assertFalse((self.root / 'changed_candidates').exists())
        source.write_bytes(candidate_bytes)
        run_path = run / 'run.json'
        run_info = read_json(run_path)
        run_info['run_hash'] = 'wrong_run'
        atomic_json(run_path, run_info)
        with self.assertRaisesRegex(ValueError, 'Run manifest hash mismatch'):
            self.merge(pools, self.root / 'changed_run')
        self.assertFalse((self.root / 'changed_run').exists())

    def test_tampered_record_and_media_block_export(self):
        pools = self.fixture()
        source, run = pools[0][1:]
        candidate = next(r for _, r in rows(source) if r['video'].endswith('123.webm'))
        path = run / 'records' / (candidate['sample_id'] + '.json')
        saved = path.read_bytes()
        record = read_json(path)
        record['candidate_hash'] = 'other'
        atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, 'Record provenance mismatch'):
            self.merge(pools, self.root / 'changed_record')
        path.write_bytes(saved)
        self.ss.joinpath('123.webm').write_bytes(b'changed media')
        with self.assertRaisesRegex(ValueError, 'Video changed/missing'):
            self.merge(pools, self.root / 'changed_media')
        self.assertFalse((self.root / 'changed_record').exists())
        self.assertFalse((self.root / 'changed_media').exists())

    def test_missing_global_shard_blocks_export(self):
        pools = self.fixture()
        (pools[1][2] / 'summary-0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete global shards for incremental'):
            self.merge(pools)
        self.assertFalse((self.root / 'merged').exists())

    def test_each_expected_count_checked_before_any_output_is_published(self):
        pools = self.fixture()
        for field in ('qualified', 'full', 'planned'):
            out = self.root / ('wrong_' + field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Expected ' + field):
                self.merge(pools, out, **{'expect_' + field: 99})
            self.assertFalse(out.exists())
        self.assertFalse(list(self.root.glob('.merged-comparison-*')))

    def test_write_failure_leaves_no_partial_output(self):
        pools = self.fixture()
        def failure(path, values):
            if path.name == 'train.sft.think.jsonl':
                raise OSError('injected disk failure')
            write_rows(path, values)
        with patch('ar_annotate.merged_comparison.write_rows', side_effect=failure), \
             self.assertRaisesRegex(OSError, 'injected disk failure'):
            self.merge(pools)
        self.assertFalse((self.root / 'merged').exists())
        self.assertFalse(list(self.root.glob('.merged-comparison-*')))

    def test_existing_output_is_never_overwritten(self):
        pools = self.fixture()
        out = self.root / 'existing'
        out.mkdir()
        sentinel = out / 'original.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.merge(pools, out)
        self.assertEqual(sentinel.read_text(), 'keep')
        self.assertEqual(list(out.iterdir()), [sentinel])

    def test_cli_forwards_source_pools_expected_counts_and_student_tokenizer(self):
        tokenizer = SimpleNamespace(encode=Mock(return_value=[1, 2, 3]))
        loader = Mock(return_value=tokenizer)
        argv = ['merged_comparison', '--pool', 'old', 'old.jsonl', 'old_run',
                '--pool', 'incremental', 'new.jsonl', 'new_run', '--out', 'merged',
                '--student-model', 'student', '--expect-qualified', '9014',
                '--expect-full', '19621', '--expect-planned', '19662']
        with patch('sys.argv', argv), \
             patch.dict(sys.modules, {'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader))}), \
             patch('ar_annotate.merged_comparison.build_merged_comparison', return_value={}) as build, \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        loader.assert_called_once_with('student', local_files_only=True, trust_remote_code=False)
        args, kwargs = build.call_args
        self.assertEqual(args[:2], ([['old', 'old.jsonl', 'old_run'],
                                  ['incremental', 'new.jsonl', 'new_run']], 'merged'))
        self.assertEqual(args[2]('reference'), 3)
        tokenizer.encode.assert_called_once_with('reference', add_special_tokens=False)
        self.assertEqual(kwargs, dict(expect_qualified=9014, expect_full=19621,
                                     expect_planned=19662, student_model='student'))


if __name__ == '__main__':
    unittest.main()

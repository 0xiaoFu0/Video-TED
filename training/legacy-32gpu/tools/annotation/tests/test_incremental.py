import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_pipeline as support
from test_comparison import ComparisonBackend
from ar_annotate.__main__ import main
from ar_annotate.annotate import annotate
from ar_annotate.common import atomic_json, digest, file_hash, label, read_json, rows, write_rows
from ar_annotate.hmdb51 import combine
from ar_annotate.incremental import prepare_incremental, prepare_overlap_incremental
from ar_annotate.prepare import prepare
from ar_annotate.report import summarize
from ar_annotate.rollouts import aggregate


class IncrementalTest(unittest.TestCase):
    setUp = support.PipelineTest.setUp
    tearDown = support.PipelineTest.tearDown
    qa = support.PipelineTest.qa
    media = support.PipelineTest.media

    def fixture(self, retained=False, recoverable_missing=False):
        files = [(self.ss / '123.webm', b'old accepted'),
                 (self.ss / '124.webm', b'old technical error'),
                 (self.kin / 'train/hurdling/abcdefghijk_000001_000011.mp4', b'old needs review'),
                 (self.ss / '999.webm', b'new ssv2 overlap'),
                 (self.kin / 'train/hurdling/ZZZZZZZZZZZ_000001_000011.mp4', b'new K710 overlap window one'),
                 (self.kin / 'train/hurdling/ZZZZZZZZZZZ_000002_000012.mp4', b'new K710 overlap window two')]
        if recoverable_missing:
            files.append((self.ss / '125.webm', b'new nonoverlap punctuation label'))
        for path, content in files:
            self.media(path)
            path.write_bytes(content)
        qa = [self.qa('ssv2', '123.webm'), self.qa('ssv2', '124.webm'),
            self.qa('k710', 'abcdefghijk_000001_000011.mp4', 'hurdling'),
            self.qa('ssv2', '999.webm'),
            self.qa('k710', 'ZZZZZZZZZZZ_000001_000011.mp4', 'hurdling'),
            self.qa('k710', 'ZZZZZZZZZZZ_000002_000012.mp4', 'hurdling')]
        if recoverable_missing:
            qa.append(self.qa('ssv2', '125.webm', 'putting something, something on the table'))
        write_rows(self.source, qa)
        self.cfg['annotation'].update(rollouts=4, min_accepted_rollouts=1, prompt_profile='tree_v1')
        atomic_json(self.config, self.cfg)
        old = self.root / 'old_candidates'
        def historical_label(value):
            # Reproduce the old punctuation rejection while leaving the original
            # QA file intact for the corrected incremental rebuild.
            if ',' in value:
                raise ValueError('label incompatible with current AR reward: ' + repr(value))
            return label(value)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('ar_annotate.prepare.label', side_effect=historical_label):
            prepare(self.config, old, retain_evaluation_overlaps=retained)
        source = old / 'candidates.jsonl'
        return source, self.annotated_run(source, 'old_run')

    def annotated_run(self, source, name):
        run = self.root / name
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
                               limit_per_dataset=0, candidates=str(source), out=str(run), retry_errors=False)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('ar_annotate.annotate.model_identity', return_value={'fake': True}), \
             patch('ar_annotate.annotate.environment', return_value={}), \
             patch('ar_annotate.annotate.QwenBackend', ComparisonBackend):
            with self.assertRaisesRegex(RuntimeError, 'failed samples'):
                annotate(args)
        return run

    def combined_fixture(self):
        original, _ = self.fixture()
        hmdb = self.root / 'hmdb'
        values = []
        for i, cls in enumerate(('pushing', 'hurdling')):
            path = hmdb / cls / f'{i}.avi'
            self.media(path)
            path.write_bytes(('old hmdb ' + cls).encode())
            values.append(dict(answer=cls, videos=[str(path)],
                               metadata=dict(sample_id=f'hmdb_source_{i}', video_relpath=f'{cls}/{i}.avi')))
        hmdb_source = self.root / 'hmdb.jsonl'
        membership = self.root / 'membership.json'
        write_rows(hmdb_source, values)
        atomic_json(membership, dict(sources=[dict(benchmark='hmdb51-split2', sha256='fixture')],
                                    membership={'pushing/0.avi': ['hmdb51-split2'],
                                                'hurdling/1.avi': ['hmdb51-split2']}))
        self.cfg.update(hmdb51_source=str(hmdb_source), hmdb51_root=str(hmdb),
                        hmdb51_expected_classes=2, hmdb51_eval_membership=str(membership))
        atomic_json(self.config, self.cfg)
        combined = self.root / 'combined_candidates'
        with contextlib.redirect_stdout(io.StringIO()):
            combine(self.config, original, combined)
        source = combined / 'candidates.jsonl'
        return source, self.annotated_run(source, 'combined_run')

    def remove_old_error_content_hashes(self, source, run):
        row = next(r for _, r in rows(source) if r['video'].endswith('124.webm'))
        manifest = read_json(run / 'run.json')
        draws = []
        for i in range(4):
            path = run / 'rollouts' / row['sample_id'] / f'{i:04d}.json'
            draw = read_json(path)
            self.assertEqual(draw['status'], 'error')
            draw.pop('video_sha256', None)
            atomic_json(path, draw)
            draws.append(draw)
        record = aggregate(row, draws, manifest['spec']['config'], manifest['run_hash'])
        atomic_json(run / 'records' / (row['sample_id'] + '.json'), record)

    def increment(self, source, run, name='incremental'):
        with contextlib.redirect_stdout(io.StringIO()):
            report = prepare_overlap_incremental(self.config, source, run, self.root / name)
        return report, self.root / name

    def test_only_new_overlaps_are_generated_across_sixteen_shards(self):
        source, run = self.fixture()
        immutable = [source, source.parent / 'manifest.json', self.source] + list(run.rglob('*.json'))
        before = {p: file_hash(p) for p in immutable}
        report, out = self.increment(source, run)
        self.assertEqual(report['existing_statuses'], dict(accepted=1, error=1, needs_review=1))
        self.assertEqual(report['planned'], 3)
        self.assertEqual(report['accepted'], dict(k710=2, ssv2=1))
        fresh = [r for _, r in rows(out / 'candidates.jsonl')]
        old_ids = {r['sample_id'] for _, r in rows(source)}
        self.assertFalse(old_ids & {r['sample_id'] for r in fresh})
        self.assertTrue(all(r['evaluation_overlaps'] for r in fresh))
        # Different exact time windows from the same YouTube video stay distinct.
        kinetics = [r for r in fresh if r['dataset'] == 'k710']
        self.assertEqual({r['video_id'] for r in kinetics}, {'ZZZZZZZZZZZ'})
        self.assertEqual(len({r['video'] for r in kinetics}), 2)
        class TrackingBackend(ComparisonBackend):
            seen = set()
            def load_video(self, path):
                self.seen.add(path)
                return super().load_video(path)
        added = self.root / 'increment_run'
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=16,
                               limit_per_dataset=0, candidates=str(out / 'candidates.jsonl'),
                               out=str(added), retry_errors=False)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('ar_annotate.annotate.model_identity', return_value={'fake': True}), \
             patch('ar_annotate.annotate.environment', return_value={}), \
             patch('ar_annotate.annotate.QwenBackend', TrackingBackend):
            for i in range(16):
                args.shard_index = i
                annotate(args)
            self.assertEqual(TrackingBackend.seen, {r['video'] for r in fresh})
            saved = {p: file_hash(p) for p in added.rglob('*.json')}
            with patch('ar_annotate.annotate.QwenBackend', side_effect=AssertionError('Repeated model loading')):
                for i in range(16):
                    args.shard_index = i
                    annotate(args)
            for p, sha in saved.items():
                if p.parent.name != added.name or not p.name.startswith('summary-'):
                    self.assertEqual(file_hash(p), sha)
        self.assertTrue(summarize(added, out / 'candidates.jsonl')['complete'])
        self.assertEqual(before, {p: file_hash(p) for p in immutable})

    def test_copied_video_is_not_generated_again(self):
        source, run = self.fixture()
        self.ss.joinpath('999.webm').write_bytes(self.ss.joinpath('123.webm').read_bytes())
        report, out = self.increment(source, run)
        self.assertEqual(report['planned'], 2)
        self.assertEqual(report['skipped']['already_generated_video_content'], 1)
        self.assertEqual({r['dataset'] for _, r in rows(out / 'candidates.jsonl')}, {'k710'})

    def test_combined_hmdb_baseline_recovers_only_ssv2_and_k710(self):
        source, run = self.combined_fixture()
        immutable = list(source.parent.rglob('*.json*')) + list(run.rglob('*.json'))
        immutable += [self.root / 'old_candidates/candidates.jsonl', self.root / 'old_candidates/manifest.json',
                      self.root / 'hmdb.jsonl', self.root / 'membership.json']
        before = {p: file_hash(p) for p in immutable}
        report, out = self.increment(source, run)
        self.assertEqual(report['existing_planned'], 5)
        self.assertEqual(report['planned'], 3)
        self.assertEqual(report['accepted'], dict(k710=2, ssv2=1))
        self.assertNotIn('hmdb51', {r['dataset'] for _, r in rows(out / 'candidates.jsonl')})
        self.assertEqual(before, {p: file_hash(p) for p in immutable})

    def test_combined_source_manifest_provenance_links_are_checked(self):
        source, run = self.combined_fixture()
        path = source.parent / 'manifest.json'
        baseline = read_json(path)
        for match_nested_link in (False, True):
            with self.subTest(match_nested_link=match_nested_link):
                changed = json.loads(json.dumps(baseline))
                changed['existing_candidates_sha256'] = 'different_original_candidates'
                if match_nested_link:
                    changed['existing_manifest']['candidates_sha256'] = changed['existing_candidates_sha256']
                atomic_json(path, changed)
                with self.assertRaisesRegex(ValueError, 'candidate provenance mismatch'):
                    self.increment(source, run, 'broken_combined')
                self.assertFalse((self.root / 'broken_combined').exists())

    def test_copied_error_media_without_recorded_hash_is_skipped(self):
        source, run = self.fixture()
        self.remove_old_error_content_hashes(source, run)
        self.ss.joinpath('999.webm').write_bytes(self.ss.joinpath('124.webm').read_bytes())
        report, _ = self.increment(source, run)
        self.assertEqual(report['planned'], 2)
        self.assertEqual(report['skipped']['already_generated_video_content'], 1)
        self.assertEqual(report['existing_statuses']['error'], 1)

    def test_unreadable_error_media_without_hash_fails_before_output(self):
        source, run = self.fixture()
        self.remove_old_error_content_hashes(source, run)
        self.ss.joinpath('124.webm').unlink()
        with self.assertRaisesRegex(ValueError, 'media cannot be read'):
            self.increment(source, run)
        self.assertFalse((self.root / 'incremental').exists())

    def test_same_content_within_increment_is_generated_once(self):
        source, run = self.fixture()
        first = self.kin / 'train/hurdling/ZZZZZZZZZZZ_000001_000011.mp4'
        (self.kin / 'train/hurdling/ZZZZZZZZZZZ_000002_000012.mp4').write_bytes(first.read_bytes())
        report, _ = self.increment(source, run)
        self.assertEqual(report['planned'], 2)
        self.assertEqual(report['skipped']['duplicate_incremental_video_content'], 1)

    def test_already_generated_overlaps_and_errors_are_skipped(self):
        source, run = self.fixture(retained=True)
        report, out = self.increment(source, run)
        self.assertEqual(report['planned'], 0)
        self.assertEqual(report['skipped']['already_generated_sample_id'], 3)
        self.assertEqual((out / 'candidates.jsonl').read_text(), '')
        with self.assertRaises(FileExistsError):
            self.increment(source, run)

    def test_resolved_media_alias_is_not_generated_again(self):
        source, run = self.fixture()
        moved = self.root / 'moved_ss'
        moved.mkdir()
        (moved / '999.webm').symlink_to(self.ss / '123.webm')
        self.cfg['ssv2_root'] = str(moved)
        atomic_json(self.config, self.cfg)
        report, _ = self.increment(source, run)
        self.assertEqual(report['planned'], 2)
        self.assertEqual(report['skipped']['already_generated_resolved_video'], 1)

    def test_incomplete_or_changed_history_fails_before_outputs(self):
        source, run = self.fixture()
        record = next((run / 'records').glob('*.json'))
        content = record.read_bytes()
        record.unlink()
        with self.assertRaises(FileNotFoundError):
            self.increment(source, run, 'missing_record')
        self.assertFalse((self.root / 'missing_record').exists())
        record.write_bytes(content)
        info = read_json(record)
        info['run_hash'] = 'changed'
        atomic_json(record, info)
        with self.assertRaisesRegex(ValueError, 'record provenance'):
            self.increment(source, run, 'changed_record')
        self.assertFalse((self.root / 'changed_record').exists())

    def test_pilot_and_missing_shard_are_rejected(self):
        source, run = self.fixture()
        previous = read_json(run / 'run.json')
        pilot = json.loads(json.dumps(previous))
        pilot['spec']['selected_ids'] = pilot['spec']['selected_ids'][:1]
        pilot['run_hash'] = digest(pilot['spec'])
        atomic_json(run / 'run.json', pilot)
        with self.assertRaisesRegex(ValueError, 'FULL candidate pool'):
            self.increment(source, run, 'pilot')
        atomic_json(run / 'run.json', previous)
        (run / 'summary-0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing global shards'):
            self.increment(source, run, 'missing_shard')
        self.assertFalse((self.root / 'missing_shard').exists())

    def test_changed_source_or_overlap_list_is_rejected(self):
        source, run = self.fixture()
        original = self.source.read_bytes()
        self.source.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'QA source changed'):
            self.increment(source, run, 'changed_source')
        self.source.write_bytes(original)
        self.denied.write_bytes(self.denied.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'overlap list changed'):
            self.increment(source, run, 'changed_overlap')

    def test_cli_forwards_incremental_inputs(self):
        with patch('sys.argv', ['ar_annotate', 'prepare-overlap-incremental', '--config', 'cfg.json',
                  '--existing-candidates', 'old.jsonl', '--existing-run', 'old_run', '--out', 'new']), \
             patch('ar_annotate.incremental.prepare_overlap_incremental') as call:
            main()
        call.assert_called_once_with('cfg.json', 'old.jsonl', 'old_run', 'new')

    def test_all_missing_recovers_nonoverlap_and_overlap_without_regenerating_old_records(self):
        source, run = self.fixture(recoverable_missing=True)
        immutable = [source, source.parent / 'manifest.json', self.source] + list(run.rglob('*.json'))
        before = {p: file_hash(p) for p in immutable}
        out = self.root / 'all_missing'
        with contextlib.redirect_stdout(io.StringIO()):
            report = prepare_incremental(self.config, source, run, out)
        self.assertEqual(report['scope'], 'all-missing')
        self.assertEqual(report['purpose'], 'all_previously_omitted_exact_matches_increment_only')
        self.assertEqual(report['planned'], 4)
        self.assertEqual(report['accepted'], dict(k710=2, ssv2=2))
        self.assertEqual(report['existing_statuses'], dict(accepted=1, error=1, needs_review=1))
        self.assertEqual(report['skipped'], dict(already_generated_sample_id=3))
        fresh = [r for _, r in rows(out / 'candidates.jsonl')]
        omitted = next(r for r in fresh if r['video'].endswith('125.webm'))
        self.assertFalse(omitted.get('evaluation_overlaps'))
        self.assertEqual(omitted['answer'], label('putting something, something on the table'))
        self.assertEqual(sum(bool(r.get('evaluation_overlaps')) for r in fresh), 3)
        self.assertFalse({r['sample_id'] for r in fresh} & {r['sample_id'] for _, r in rows(source)})
        audit = [r for _, r in rows(out / 'selection.audit.jsonl')]
        self.assertEqual(sum(r['decision'] == 'new_missing_to_generate' for r in audit), 4)
        self.assertEqual(before, {p: file_hash(p) for p in immutable})

    def test_overlap_scope_remains_narrow_after_label_recovery(self):
        source, run = self.fixture(recoverable_missing=True)
        report, out = self.increment(source, run)
        self.assertEqual(report['scope'], 'evaluation-overlaps')
        self.assertEqual(report['planned'], 3)
        self.assertEqual(report['skipped']['not_a_previously_excluded_evaluation_overlap'], 4)
        self.assertTrue(all(r.get('evaluation_overlaps') for _, r in rows(out / 'candidates.jsonl')))

    def test_cli_defaults_to_all_missing_and_accepts_explicit_overlap_scope(self):
        for extra, expected in (([], 'all-missing'), (['--scope', 'evaluation-overlaps'], 'evaluation-overlaps')):
            with self.subTest(scope=expected), \
                 patch('sys.argv', ['ar_annotate', 'prepare-incremental', '--config', 'cfg.json',
                       '--existing-candidates', 'old.jsonl', '--existing-run', 'old_run', '--out', 'new'] + extra), \
                 patch('ar_annotate.incremental.prepare_incremental') as call:
                main()
            call.assert_called_once_with('cfg.json', 'old.jsonl', 'old_run', 'new', scope=expected)

    def test_unknown_scope_fails_before_outputs(self):
        with self.assertRaisesRegex(ValueError, 'Incremental scope'):
            prepare_incremental(self.config, 'unused.jsonl', 'unused', self.root / 'bad_scope', scope='all')
        self.assertFalse((self.root / 'bad_scope').exists())


if __name__ == '__main__':
    unittest.main()

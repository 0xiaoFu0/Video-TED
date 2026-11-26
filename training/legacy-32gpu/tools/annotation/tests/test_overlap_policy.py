import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ar_annotate.__main__ import main
from ar_annotate.common import atomic_json, read_json, rows, write_rows
from ar_annotate.prepare import MARKERS, prepare


class OverlapPolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ss = self.root / 'ss'
        self.kin = self.root / 'kin'
        self.ss.mkdir()
        (self.kin / 'train/hurdling').mkdir(parents=True)
        self.source = self.root / 'source.jsonl'
        self.denied = self.root / 'deny.json'
        atomic_json(self.denied, {'video_ids': {'ssv2': ['999', '888'], 'k710': ['abcdefghijk']}})
        self.config = self.root / 'config.json'
        atomic_json(self.config, dict(flash_jsonl=str(self.source), ssv2_root=str(self.ss),
                    k710_root=str(self.kin), eval_exclusions=str(self.denied)))

    def tearDown(self):
        self.tmp.cleanup()

    def qa(self, dataset, filename, target='pushing'):
        return dict(video='/old' + MARKERS[dataset] + filename,
                    messages=[dict(role='user', content='Choose. Options:\n(A) ' + target + '\n(B) pulling'),
                              dict(role='assistant', content='Answer: (A) ' + target)])

    def media(self, path):
        path.write_bytes(b'fixture video only - never decoded')

    def test_default_excludes_evaluation_videos(self):
        self.media(self.ss / '123.webm')
        self.media(self.ss / '999.webm')
        write_rows(self.source, [self.qa('ssv2', '123.webm'), self.qa('ssv2', '999.webm')])
        out = self.root / 'prepared'
        prepare(self.config, out)
        selected = [r for _, r in rows(out / 'candidates.jsonl')]
        self.assertEqual([r['video_id'] for r in selected], ['123'])
        manifest = read_json(out / 'manifest.json')
        self.assertEqual(manifest['rejected']['evaluation_video_overlap'], 1)
        self.assertEqual(manifest['overlap_policy'], 'exclude_configured_evaluation_video_overlaps')

    def test_retain_flag_keeps_exact_matches_and_original_labels(self):
        self.media(self.ss / '999.webm')
        self.media(self.kin / 'train/hurdling/abcdefghijk_000001_000011.mp4')
        write_rows(self.source, [self.qa('ssv2', '999.webm'),
                                self.qa('k710', 'abcdefghijk_000001_000011.mp4', 'hurdling')])
        out = self.root / 'prepared'
        prepare(self.config, out, retain_evaluation_overlaps=True)
        selected = {r['dataset']: r for _, r in rows(out / 'candidates.jsonl')}
        self.assertEqual(set(selected), {'ssv2', 'k710'})
        for dataset, target in [('ssv2', 'pushing'), ('k710', 'hurdling')]:
            self.assertEqual(selected[dataset]['answer'], target)
            self.assertEqual(selected[dataset]['evaluation_overlaps'],
                             [dataset + '-configured-evaluation-video'])
        manifest = read_json(out / 'manifest.json')
        self.assertEqual(manifest['rejected'], {})
        self.assertEqual(manifest['overlap_policy'], 'retain_configured_evaluation_video_overlaps')

    def test_retain_flag_still_rejects_missing_empty_and_wrong_time_windows(self):
        self.media(self.ss / '123.webm')
        (self.ss / '888.webm').write_bytes(b'')
        self.media(self.kin / 'train/hurdling/abcdefghijk_000001_000011.mp4')
        write_rows(self.source, [self.qa('ssv2', '123.webm'), self.qa('ssv2', '999.webm'),
                                self.qa('ssv2', '888.webm'),
                                self.qa('k710', 'abcdefghijk_000002_000012.mp4', 'hurdling')])
        out = self.root / 'prepared'
        prepare(self.config, out, retain_evaluation_overlaps=True)
        selected = [r for _, r in rows(out / 'candidates.jsonl')]
        self.assertEqual([r['video_id'] for r in selected], ['123'])
        self.assertEqual(read_json(out / 'manifest.json')['rejected'],
                         {'not_found_exact_filename': 3})

    def test_retain_flag_still_rejects_conflicting_labels_and_duplicate_media(self):
        self.media(self.ss / '123.webm')
        self.media(self.ss / '999.webm')
        write_rows(self.source, [self.qa('ssv2', '123.webm'), self.qa('ssv2', '123.webm'),
                                self.qa('ssv2', '999.webm'), self.qa('ssv2', '999.webm', 'lifting')])
        out = self.root / 'prepared'
        prepare(self.config, out, retain_evaluation_overlaps=True)
        selected = [r for _, r in rows(out / 'candidates.jsonl')]
        self.assertEqual([r['video_id'] for r in selected], ['123'])
        self.assertEqual(read_json(out / 'manifest.json')['rejected'],
                         {'conflicting_source_labels': 2, 'duplicate_resolved_video': 1})

    def test_cli_forwards_retain_flag(self):
        with patch('sys.argv', ['ar_annotate', 'prepare', '--config', 'config.json',
                               '--out', 'new-candidates', '--retain-evaluation-overlaps']), \
             patch('ar_annotate.prepare.prepare') as prepare_mock:
            main()
        prepare_mock.assert_called_once_with('config.json', 'new-candidates', retain_evaluation_overlaps=True)


if __name__ == '__main__':
    unittest.main()

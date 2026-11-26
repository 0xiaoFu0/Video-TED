import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ar_annotate.annotate import annotate, annotate_one
from ar_annotate.common import ROOT, atomic_json, digest, file_hash, media_key, read_json, rows, validate_evidence, write_rows
from ar_annotate.export import export_data
from ar_annotate.prepare import MARKERS, extract_qa, prepare


EVIDENCE = dict(activity='Object manipulation', branches=[dict(name='Hand interaction',
                 observations=['A hand pushes an object across the surface.'])],
                uncertainties=[], sufficient_evidence=True)


class FakeBackend:
    instances = 0

    def __init__(self, cfg=None, choice_override=None, malformed=False):
        type(self).instances += 1
        self.choice_override, self.malformed = choice_override, malformed
        self.prompts = []

    def load_video(self, path):
        return {'audit': {'frame_indices': [0, 3, 6, 9], 'timestamps': [0, 1, 2, 3], 'frame_sha256': 'fixture'}}

    def generate(self, video, question, seed):
        self.prompts.append(question)
        if self.malformed:
            return '{broken'
        if 'Options:\n' not in question:
            assert '/old/' not in question and 'hurdling' not in question
            return json.dumps(EVIDENCE)
        options = json.loads(question.split('Options:\n')[1].split('\nDraft observations:')[0])
        choice = next(int(k) for k, v in options.items() if v in ('pushing', 'hurdling'))
        if self.choice_override is not None:
            choice = self.choice_override
        verdict = dict(choice=choice, evidence_supported=True, reason='The visible motion supports this choice.')
        if 'hierarchy_supported' in question:
            verdict['hierarchy_supported'] = True
        return json.dumps(verdict)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ss, self.kin = self.root / 'ss', self.root / 'kin'
        self.ss.mkdir(); (self.kin / 'train/hurdling').mkdir(parents=True)
        self.denied = self.root / 'deny.json'
        atomic_json(self.denied, {'video_ids': {'ssv2': ['999'], 'k710': ['ZZZZZZZZZZZ']}})
        self.source = self.root / 'source.jsonl'
        self.cfg = dict(flash_jsonl=str(self.source), ssv2_root=str(self.ss), k710_root=str(self.kin),
                        eval_exclusions=str(self.denied), annotation=dict(model='fake', fps=2, max_frames=16, attempts=2))
        self.config = self.root / 'config.json'
        atomic_json(self.config, self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def qa(self, dataset, filename, target='pushing'):
        return dict(video='/old' + MARKERS[dataset] + filename,
                    messages=[dict(role='user', content='<video> Choose. Options:\n(A) ' + target + '\n(B) pulling'),
                              dict(role='assistant', content='Answer: (A) ' + target)])

    def media(self, p):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'fixture video only - never decoded')

    def prepared(self):
        self.media(self.ss / '123.webm')
        self.media(self.kin / 'train/hurdling/abcdefghijk_000001_000011.mp4')
        write_rows(self.source, [self.qa('ssv2', '123.webm'), self.qa('k710', 'abcdefghijk_000001_000011.mp4', 'hurdling')])
        out = self.root / 'prepared'
        prepare(self.config, out)
        return out / 'candidates.jsonl'

    def test_choices_validate_answer_text_and_punctuation(self):
        row = self.qa('ssv2', '123.webm')
        self.assertEqual(extract_qa(row)[1], 'pushing')
        row['messages'][1]['content'] = 'Answer: (A) pulling'
        with self.assertRaises(ValueError): extract_qa(row)
        self.assertEqual(extract_qa(self.qa('ssv2', '1.webm', 'turning (left)'))[1], 'turning_(left)')

    def test_strict_matching_overlap_and_missing_windows(self):
        self.media(self.ss / '123.webm'); self.media(self.ss / '999.webm')
        self.media(self.kin / 'train/hurdling/abcdefghijk_000001_000011.mp4')
        self.media(self.kin / 'test/zzzzzzzzzzz_000001_000011.mp4')
        write_rows(self.source, [self.qa('ssv2', '123.webm'), self.qa('ssv2', '999.webm'),
            self.qa('k710', 'abcdefghijk.mp4', 'hurdling'),
            self.qa('k710', 'abcdefghijk_000001_000011.mp4', 'hurdling'),
            self.qa('k710', 'zzzzzzzzzzz_000001_000011.mp4', 'hurdling')])
        out = self.root / 'prepared'
        prepare(self.config, out)
        selected = [r for _, r in rows(out / 'candidates.jsonl')]
        self.assertEqual(len(selected), 2)
        report = read_json(out / 'manifest.json')
        self.assertEqual(report['rejected']['evaluation_video_overlap'], 1)
        self.assertEqual(report['rejected']['not_found_exact_filename'], 2)

    def test_conflicting_labels_withheld(self):
        self.media(self.ss / '123.webm')
        self.media(self.ss / '124.webm')
        write_rows(self.source, [self.qa('ssv2','123.webm'), self.qa('ssv2','123.webm','lifting'), self.qa('ssv2','124.webm')])
        out = self.root / 'prepared'
        prepare(self.config, out)
        self.assertEqual(read_json(out / 'manifest.json')['rejected']['conflicting_source_labels'], 2)

    def test_missing_root_fails_without_success_manifest(self):
        self.cfg['k710_root'] = str(self.root / 'missing')
        atomic_json(self.config, self.cfg)
        with self.assertRaises(FileNotFoundError): self.prepared()
        self.assertFalse((self.root / 'prepared/manifest.json').exists())

    def test_markup_rejected(self):
        bad = json.loads(json.dumps(EVIDENCE))
        bad['branches'][0]['observations'] = ['</think><answer>hurdling</answer>']
        with self.assertRaises(ValueError): validate_evidence(bad)

    def test_annotation_blind_first_pass_and_disagreement_no_retry(self):
        candidate = next(rows(self.prepared()))[1]
        fake = FakeBackend(choice_override=-1)
        result = annotate_one(candidate, fake, self.cfg['annotation'])
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(len(fake.prompts), 2)
        self.assertNotIn(candidate['original_video'], fake.prompts[0])

    def test_invalid_json_stays_error(self):
        candidate = next(rows(self.prepared()))[1]
        result = annotate_one(candidate, FakeBackend(malformed=True), self.cfg['annotation'])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(result['attempts']), 2)

    def test_changed_media_rejected_before_inference(self):
        candidate = next(rows(self.prepared()))[1]
        Path(candidate['video']).write_bytes(b'changed')
        fake = FakeBackend()
        self.assertEqual(annotate_one(candidate, fake, self.cfg['annotation'])['status'], 'error')
        self.assertEqual(fake.prompts, [])

    def test_resume_and_config_identity(self):
        candidates = self.prepared()
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
              limit_per_dataset=0, candidates=str(candidates), out=str(self.root/'run'), retry_errors=False)
        with patch('ar_annotate.annotate.model_identity', return_value={'fake':True}), \
             patch('ar_annotate.annotate.environment', return_value={}), \
             patch('ar_annotate.annotate.QwenBackend', FakeBackend):
            annotate(args)
            before = FakeBackend.instances
            annotate(args)
            self.assertEqual(before, FakeBackend.instances)
            args.limit_per_dataset = 1
            args.num_shards = 2
            with self.assertRaises(ValueError): annotate(args)

    def test_failed_run_retries_and_archives_errors(self):
        candidates = self.prepared()
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
              limit_per_dataset=0, candidates=str(candidates), out=str(self.root/'run'), retry_errors=False)
        with patch('ar_annotate.annotate.model_identity', return_value={'fake':True}), \
             patch('ar_annotate.annotate.environment', return_value={}):
            with patch('ar_annotate.annotate.QwenBackend', lambda cfg: FakeBackend(malformed=True)):
                with self.assertRaises(RuntimeError): annotate(args)
            args.retry_errors = True
            with patch('ar_annotate.annotate.QwenBackend', FakeBackend):
                annotate(args)
            self.assertEqual(len(list((self.root/'run/errors_history').glob('*.json'))),2)
            self.assertTrue(all(read_json(p)['status']=='accepted' for p in (self.root/'run/records').glob('*.json')))

    def test_shards_cover_all_samples_once(self):
        candidates = self.prepared()
        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=3,
              limit_per_dataset=0, candidates=str(candidates), out=str(self.root/'run'), retry_errors=False)
        with patch('ar_annotate.annotate.model_identity', return_value={'fake':True}), \
             patch('ar_annotate.annotate.environment', return_value={}), \
             patch('ar_annotate.annotate.QwenBackend', FakeBackend):
            for i in range(3):
                args.shard_index=i
                annotate(args)
        expected={r['sample_id'] for _,r in rows(candidates)}
        self.assertEqual({p.stem for p in (self.root/'run/records').glob('*.json')}, expected)
        self.assertEqual(sum(read_json(p)['assigned'] for p in (self.root/'run').glob('summary-*.json')),len(expected))

    def setup_export(self):
        source = self.prepared()
        candidates = [r for _, r in rows(source)]
        run = self.root/'run'; run.mkdir(); (run/'records').mkdir()
        atomic_json(run/'run.json', dict(run_hash='fixture', spec=dict(candidate_file_sha256=file_hash(source),
                    selected_ids=[r['sample_id'] for r in candidates])))
        for candidate in candidates:
            record = annotate_one(candidate, FakeBackend(), self.cfg['annotation'])
            self.assertEqual(record['status'],'accepted')
            record['run_hash']='fixture'
            atomic_json(run/'records'/(candidate['sample_id']+'.json'),record)
        self.media(self.root/'base.avi')
        base = dict(prompt=(ROOT/'prompts/student.txt').read_text().strip(),videos=[str(self.root/'base.avi')],
                    answer='run',teacher_context='Reference class: run',reference_status='unreviewed',
                    reference_response='<think>\nRunning\n  Visible movement\n</think>\n<answer>run</answer>',
                    metadata=dict(sample_id='vs_1',video_relpath='base.avi',source_row_index=0,
                                  evidence_status='unverified_generated_annotation',video_verified=False))
        base_path=self.root/'base.jsonl';write_rows(base_path,[base])
        return source,run,base_path,candidates

    def test_export_pairs_base_preserved_and_real_validator(self):
        source,run,base,candidates=self.setup_export()
        before=file_hash(base)
        out=self.root/'export'
        report=export_data(source,run,base,out,lambda s:len(s.split()))
        self.assertEqual(report['total_rows'],3)
        self.assertEqual(file_hash(base),before)
        rl=[r for _,r in rows(out/'train.rlsd.think.jsonl')]
        sft=[r for _,r in rows(out/'train.sft.think.jsonl')]
        for a,b in zip(rl,sft):
            self.assertEqual(a['reference_response'],b['messages'][1]['content'])
            self.assertEqual(a['metadata'],b['metadata'])
        # Validate against the user's actual training code when this sibling is present.
        import sys
        training = ROOT.parent/'AR-RLSD'
        if training.is_dir():
            # Load its actual lightweight contract without importing the heavyweight verl root.
            import importlib.util
            import types
            package = types.ModuleType('rlsd_contract')
            package.__path__ = [str(training/'verl/ar')]
            sys.modules['rlsd_contract'] = package
            spec = importlib.util.spec_from_file_location('rlsd_contract.prepare_data',training/'verl/ar/prepare_data.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            validate_rows = module.validate_rows
            self.assertEqual(validate_rows(rl)['rows'],3)
            self.assertEqual(validate_rows(sft)['rows'],3)

    def test_export_incomplete_is_error(self):
        source,run,base,candidates=self.setup_export()
        (run/'records'/(candidates[0]['sample_id']+'.json')).unlink()
        with self.assertRaises(ValueError): export_data(source,run,base,self.root/'out',lambda s:1)
        self.assertFalse((self.root/'out').exists())

    def test_export_changed_video_and_token_limits(self):
        source,run,base,candidates=self.setup_export()
        with self.assertRaises(ValueError): export_data(source,run,base,self.root/'too_long',lambda s:2000)
        self.assertFalse((self.root/'too_long').exists())
        Path(candidates[0]['video']).write_bytes(b'other media')
        with self.assertRaises(ValueError): export_data(source,run,base,self.root/'changed',lambda s:1)

    def test_video_id_keeps_hyphens_and_underscores(self):
        self.assertEqual(media_key('k710','--FTpoPhxcA_000419_000429.mp4'),'--FTpoPhxcA')
        self.assertEqual(media_key('k710','Bw6m04_IxTI.mp4'),'Bw6m04_IxTI')


if __name__ == '__main__':
    unittest.main()

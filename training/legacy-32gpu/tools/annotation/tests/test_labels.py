import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_pipeline as support
from ar_annotate.annotate import annotate
from ar_annotate.common import label, read_json, rows, write_rows
from ar_annotate.export import export_data
from ar_annotate.prepare import extract_qa, prepare


class LabelsTest(unittest.TestCase):
    setUp = support.PipelineTest.setUp
    tearDown = support.PipelineTest.tearDown
    qa = support.PipelineTest.qa
    media = support.PipelineTest.media

    def test_dataset_punctuation_and_negation_are_preserved(self):
        cases = {
            'Pulling rope (game)': 'pulling_rope_(game)',
            'Passing American football (in game)': 'passing_american_football_(in_game)',
            'Lifting up one end of something, then letting it drop down':
                'lifting_up_one_end_of_something,_then_letting_it_drop_down',
            "Tilting something with something on it slightly so it doesn't fall down":
                "tilting_something_with_something_on_it_slightly_so_it_doesn't_fall_down",
            'Twisting (wringing) something wet until water comes out':
                'twisting_(wringing)_something_wet_until_water_comes_out',
            'Touching (without moving) part of something': 'touching_(without_moving)_part_of_something',
            'Laying something on the table on its side, not upright':
                'laying_something_on_the_table_on_its_side,_not_upright',
            "Pretending to pour something out of something, but something is empty":
                'pretending_to_pour_something_out_of_something,_but_something_is_empty',
            'Doing 360-degree turns': 'doing_360-degree_turns',
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(label(raw), expected)
                self.assertEqual(label(expected), expected)
        self.assertNotEqual(label('touching without moving'), label('touching with moving'))
        self.assertNotEqual(label('playing football (in game)'), label('playing football (not in game)'))

    def test_existing_labels_remain_identical(self):
        for raw, expected in [('hurdling', 'hurdling'), ('  Rock  Climbing ', 'rock_climbing'),
                              ('rock__climbing', 'rock_climbing'), ('turning_the_camera_downwards',
                               'turning_the_camera_downwards')]:
            self.assertEqual(label(raw), expected)
        self.assertEqual(label('Something doesn’t move'), "something_doesn't_move")

    def test_bad_labels_still_fail(self):
        for raw in ['', '  ', '()', '123', '<answer>pushing</answer>', 'pushing\npulling',
                    'pushing\x00', 'pushing\\pulling', 'x' * 301]:
            with self.subTest(raw=repr(raw)), self.assertRaises(ValueError):
                label(raw)

    def test_punctuation_in_distractor_no_longer_drops_valid_qa(self):
        row = self.qa('ssv2', '123.webm')
        row['messages'][0]['content'] = '<video> Choose. Options:\n(A) pushing\n(B) pulling rope (game)'
        raw, target, options = extract_qa(row)
        self.assertEqual((raw, target), ('pushing', 'pushing'))
        self.assertEqual(options, ['pushing', 'pulling rope (game)'])
        row['messages'][1]['content'] = 'Answer: (A) pulling rope (game)'
        with self.assertRaisesRegex(ValueError, 'differs'):
            extract_qa(row)

    def test_punctuated_label_prepare_annotate_and_export(self):
        target = "Pushing something so that it almost falls off but doesn't"
        self.media(self.ss / '123.webm')
        write_rows(self.source, [self.qa('ssv2', '123.webm', target)])
        candidates = self.root / 'prepared'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.config, candidates)
        candidate = next(rows(candidates / 'candidates.jsonl'))[1]
        self.assertEqual(candidate['answer'], label(target))
        self.assertEqual(candidate['raw_label'], target)

        class PunctuatedBackend(support.FakeBackend):
            def generate(self, video, question, seed):
                if 'Options:\n' not in question:
                    return super().generate(video, question, seed)
                options = json.loads(question.split('Options:\n')[1].split('\nDraft observations:')[0])
                choice = next(int(k) for k, value in options.items() if value == target)
                return json.dumps(dict(choice=choice, evidence_supported=True, hierarchy_supported=True,
                                       reason='The visible motion supports this option.'))

        args = SimpleNamespace(config=str(self.config), model=None, shard_index=0, num_shards=1,
                               limit_per_dataset=0, candidates=str(candidates / 'candidates.jsonl'),
                               out=str(self.root / 'run'), retry_errors=False, rollouts=4,
                               min_accepted_rollouts=1, prompt_profile='tree_v1')
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('ar_annotate.annotate.model_identity', return_value={'fake': True}), \
             patch('ar_annotate.annotate.environment', return_value={}), \
             patch('ar_annotate.annotate.QwenBackend', PunctuatedBackend):
            annotate(args)
            export_data(args.candidates, args.out, None, self.root / 'export', lambda _: 100,
                        pool_only=True)
        record = read_json(self.root / 'run/records' / (candidate['sample_id'] + '.json'))
        self.assertEqual(record['status'], 'accepted')
        rlsd = next(rows(self.root / 'export/pool.rlsd.think.jsonl'))[1]
        sft = next(rows(self.root / 'export/pool.sft.think.jsonl'))[1]
        self.assertEqual(rlsd['answer'], label(target))
        self.assertTrue(rlsd['reference_response'].endswith('<answer>' + label(target) + '</answer>'))
        self.assertEqual(sft['messages'][1]['content'], rlsd['reference_response'])
        self.assertEqual(rlsd['metadata']['raw_label'], target)


if __name__ == '__main__':
    unittest.main()

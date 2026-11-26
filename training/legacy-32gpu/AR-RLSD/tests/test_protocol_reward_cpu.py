"""Exercise production reward/config guards with stdlib-only CPU fixtures."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formatter = load('protocol_cpu_formatter', 'verl/ar/response_format.py')
# Match the real custom reward file loader, without importing training packages.
with patch.dict(sys.modules, {'verl.ar.response_format': formatter}):
    reward = load('custom_reward_fn_cpu', 'verl/ar/reward.py')
validation = load('protocol_cpu_validation', 'verl/ar/validation.py')


def response(answer='pushup', body='Motion\n  Interaction\n    The arms bend.'):
    return '<think>' + body + '</think><answer>' + answer + '</answer>'


class ProtocolRewardTest(unittest.TestCase):
    def score(self, text, target='pushup', mode='protocol'):
        return reward.compute_score([dict(response=text, ground_truth=target)], reward_mode=mode)[0]

    def test_complete_correct_missing_think_and_wrong_class_are_distinct(self):
        self.assertEqual(self.score(response()),
                         dict(overall=1., accuracy=1., answer_valid=1., protocol_valid=1.))
        missing = '<answer>pushup</answer>'
        self.assertEqual(self.score(missing),
                         dict(overall=0., accuracy=1., answer_valid=1., protocol_valid=0.))
        self.assertEqual(self.score(missing, mode='answer_only')['overall'], 1.)
        self.assertEqual(reward.compute_score([dict(response=missing, ground_truth='pushup')])[0]['overall'], 1.)
        self.assertEqual(self.score(response('run')),
                         dict(overall=0., accuracy=0., answer_valid=1., protocol_valid=1.))

    def test_original_full_data_label_spelling_remains_supported_in_protocol(self):
        pairs = [
            ('Pulling rope (game)', 'pulling_rope_(game)'),
            ('Passing American football (in game)', 'passing_american_football_(in_game)'),
            ('Counting to 10', 'counting_to_10'),
            ('Playing rock-paper-scissors', 'playing_rock-paper-scissors'),
            ("Tilting something with something on it slightly so it doesn't fall down",
             "tilting_something_with_something_on_it_slightly_so_it_doesn't_fall_down"),
            ('Lifting up one end of something, then letting it drop down',
             'lifting_up_one_end_of_something,_then_letting_it_drop_down'),
            ('Something doesn’t move', "something_doesn't_move"),
            ('“Push & pull” / [test] + stop!?;: then.', '"push_&_pull"_/_[test]_+_stop!?;:_then.'),
        ]
        for raw, canonical in pairs:
            with self.subTest(raw=raw):
                self.assertEqual(reward.target_label(raw), canonical)
                self.assertEqual(self.score(response(raw), target=canonical)['overall'], 1.)
                self.assertEqual(self.score('<answer>'+raw+'</answer>', target=canonical)['accuracy'], 1.)
                self.assertEqual(self.score('<answer>'+raw+'</answer>', target=canonical)['overall'], 0.)

    def test_negation_punctuation_and_digits_are_not_erased_into_other_classes(self):
        for prediction, target in [
            ('pulling_rope', 'pulling_rope_(game)'),
            ('something_does_move', "something_doesn't_move"),
            ('counting_to_1', 'counting_to_10'),
            ('playing_rock_paper_scissors', 'playing_rock-paper-scissors')]:
            with self.subTest(prediction=prediction):
                value = self.score(response(prediction), target=target)
                self.assertEqual(value['overall'], 0.)
                self.assertEqual(value['accuracy'], 0.)
                self.assertEqual(value['protocol_valid'], 1.)

    def test_invalid_protocol_is_not_rescued_by_correct_answer(self):
        bad = [
            '<think> </think><answer>pushup</answer>',
            '<think>Evidence<answer>pushup</answer>',
            '<answer>pushup</answer><think>Evidence</think>',
            response() + ' trailing',
            response() + '<answer>pushup</answer>',
            response(body='<think>Evidence</think>'),
            response(body='<THINK>Evidence</THINK>'),
            response(body='<think id="nested">Evidence</think>'),
            response(body='<action_tree>Evidence</action_tree>'),
            response(body='<thought id="legacy">Evidence</thought>'),
            response(body='<object>Evidence</object>'),
            response(body='<unknown/>Evidence'),
        ]
        for text in bad:
            with self.subTest(text=text):
                value = self.score(text)
                self.assertEqual(value['overall'], 0.)
                self.assertEqual(value['protocol_valid'], 0.)

    def test_protocol_is_not_a_tree_evidence_judge(self):
        self.assertEqual(self.score(response(body='Anything'))['overall'], 1.)
        self.assertEqual(self.score(' \n'+response()+'\n ')['overall'], 1.)
        for answer in ('123', '动作', 'action=class'):
            text = response(answer)
            self.assertTrue(formatter.protocol_valid(text))  # Envelope only.
            self.assertEqual(self.score(text)['protocol_valid'], 0.)  # Existing label parser.

    def test_removed_modes_and_kwargs_fail_without_external_calls(self):
        for mode in ('protocol_tree', 'leaf_weighted', 'invalid'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.score(response(), mode=mode)
        with self.assertRaises(ValueError):
            reward.compute_score([], evidence_judge_endpoint='https://unused.invalid')


def training_config(mode='protocol'):
    ns = types.SimpleNamespace
    return ns(
        opsd=ns(ar_reward_mode=mode, ar_enabled=True, ar_reference_coef=.1,
            ar_reference_kappa=1., ar_reference_mode='clipped', ar_reference_clip=.2,
            ar_reference_max_length=1501, enable_grpo_opsd_simple=True,
            freeze_teacher_model=True, stgca_teacher_sync_interval=1,
            stgca_thought_only=False, stgca_negative_only=False),
        data=ns(rollout_batch_size=32, mini_rollout_batch_size=None,
                truncation='error', trajectory_key=None),
        trainer=ns(nnodes=4, n_gpus_per_node=8),
        worker=ns(
            actor=ns(global_batch_size=32, micro_batch_size_per_device_for_update=1,
                     ulysses_size=1, dynamic_batching=False, loss_avg_mode='token'),
            rollout=ns(n=4, enable_mix_policy=False),
            reward=ns(reward_function_kwargs={'reward_mode': mode})),
        algorithm=ns(use_kl_loss=False, disable_kl=True, online_filtering=False))


class ProtocolConfigTest(unittest.TestCase):
    def test_baseline_and_protocol_configs_pass_real_validator(self):
        validation.validate_training_config(training_config())
        baseline = training_config('answer_only')
        baseline.worker.reward.reward_function_kwargs = {}
        validation.validate_training_config(baseline)

    def test_reward_mode_mismatch_and_unknown_options_are_rejected(self):
        cases = [
            ('protocol_tree', {'reward_mode': 'protocol_tree'}),
            ('protocol', {}),
            ('answer_only', {'reward_mode': 'protocol'}),
            ('protocol', {'reward_mode': 'protocol', 'evidence_lambda': 1.}),
            ('protocol', None),
        ]
        for mode, kwargs in cases:
            with self.subTest(mode=mode, kwargs=kwargs), self.assertRaises(ValueError):
                cfg = training_config(mode)
                cfg.worker.reward.reward_function_kwargs = kwargs
                validation.validate_training_config(cfg)

    def test_full_config_still_rejects_32_rank_prompt_dispatch_failures(self):
        cfg = training_config()
        cfg.data.rollout_batch_size = 16
        cfg.worker.actor.global_batch_size = 16
        with self.assertRaisesRegex(ValueError, 'GPU count'):
            validation.validate_training_config(cfg)
        cfg = training_config()
        cfg.data.mini_rollout_batch_size = 16
        with self.assertRaisesRegex(ValueError, 'GPU count'):
            validation.validate_training_config(cfg)

    def test_reference_coefficient_must_remain_finite_nonnegative(self):
        for value in (-.1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cfg = training_config()
                cfg.opsd.ar_reference_coef = value
                validation.validate_training_config(cfg)


if __name__ == '__main__':
    unittest.main()

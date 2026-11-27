"""Action labels and VideoTED's response format."""
from copy import deepcopy

import pytest

from verl.ar.prepare_data import validate_rows
from verl.ar.response_format import validate_reference_format
from verl.ar.reward import compute_score, normalize_label, parse_answer, target_label


LABELS = [
    ('Pulling rope (game)', 'pulling_rope_(game)'),
    ('Passing American football (in game)', 'passing_american_football_(in_game)'),
    ('Lifting up one end of something, then letting it drop down',
     'lifting_up_one_end_of_something,_then_letting_it_drop_down'),
    ("Tilting something with something on it slightly so it doesn't fall down",
     "tilting_something_with_something_on_it_slightly_so_it_doesn't_fall_down"),
    ('Playing rock-paper-scissors', 'playing_rock-paper-scissors'),
    ('Counting to 10', 'counting_to_10'),
    ('“Push & pull” / [test] + stop!?;: then.', '"push_&_pull"_/_[test]_+_stop!?;:_then.'),
    ('Something doesn’t move', "something_doesn't_move"),
    ('  Brush__ HAIR  ', 'brush_hair'),
]


def response(answer):
    return ('<think>\nObject motion\n  Hand interaction\n'
            '    The hand makes contact with the object.\n</think>\n'
            '<answer>' + answer + '</answer>')


def training_row(answer):
    prompt = ('<video>\nDescribe the action using <think>...</think>, '
              'then <answer>action_class</answer>.')
    return dict(prompt=prompt, videos=['clips/example.webm'], answer=answer,
                reference_response=response(answer), reference_status='unreviewed',
                teacher_context='Reference class: ' + answer)


@pytest.mark.parametrize('raw,canonical', LABELS)
def test_class_spelling_reward_and_training_data_validation(raw, canonical):
    assert normalize_label(raw) == target_label(raw) == canonical
    text = response(raw)
    assert parse_answer(text) == canonical
    assert target_label(text) == canonical
    validate_reference_format(text)
    result = compute_score([dict(response=text, ground_truth=canonical)])[0]
    assert result == dict(overall=1., accuracy=1., answer_valid=1., protocol_valid=1.)
    assert compute_score([dict(response=text, ground_truth=canonical)], reward_mode='protocol')[0] == result
    row = training_row(canonical)
    before = deepcopy(row)
    assert validate_rows([row])['rows'] == 1
    assert row == before  # Preserve the hierarchy and class spelling.


@pytest.mark.parametrize('raw', [
    '', '123', 'a' * 301, 'action\nclass', 'action\tclass', 'action\x00class',
    'action\x7fclass', 'action<class>', 'action\\class', 'action=class', '\u52a8\u4f5c', None,
])
def test_unsafe_empty_or_non_class_labels_are_rejected(raw):
    with pytest.raises(ValueError):
        target_label(raw)
    text = '<answer>' + (raw if isinstance(raw, str) else '') + '</answer>'
    assert parse_answer(text) is None


@pytest.mark.parametrize('text', [
    '<answer>pulling_rope_(game)',
    '<answer>pulling_rope_(game)</answer> trailing',
    '<answer>pulling_rope_(game)</answer><answer>run</answer>',
    '<answer><answer>pulling_rope_(game)</answer></answer>',
    '<Answer>pulling_rope_(game)</Answer>',
    '<answer>pulling_rope_(game)<extra></answer>',
    '<answer></answer>',
])
def test_malformed_answer_envelopes_are_invalid(text):
    assert parse_answer(text) is None
    assert compute_score([dict(response=text, ground_truth='pulling_rope_(game)')])[0] == dict(
        overall=0., accuracy=0., answer_valid=0., protocol_valid=0.)


@pytest.mark.parametrize('prediction,target', [
    ('pulling_rope', 'pulling_rope_(game)'),
    ('something_does_move', "something_doesn't_move"),
    ('counting_to_1', 'counting_to_10'),
    ('playing_rock_paper_scissors', 'playing_rock-paper-scissors'),
])
def test_different_class_or_negation_does_not_receive_accuracy_reward(prediction, target):
    assert compute_score([dict(response=response(prediction), ground_truth=target)])[0] == dict(
        overall=0., accuracy=0., answer_valid=1., protocol_valid=1.)


@pytest.mark.parametrize("text", [
    "<answer>pushup</answer>",
    "<think> </think><answer>pushup</answer>",
    "<think><action_tree>Exercise</action_tree></think><answer>pushup</answer>",
])
def test_correct_label_without_valid_reasoning_has_zero_protocol_reward(text):
    row = dict(response=text, ground_truth="pushup")
    assert compute_score([row], reward_mode="protocol")[0] == dict(
        overall=0., accuracy=1., answer_valid=1., protocol_valid=0.)


def test_reference_batch_encodes_punctuation_without_rewriting():
    from verl.ar.batch import encode_references

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def encode(self, text, add_special_tokens=False):
            return [ord(c) + 3 for c in text]

    labels = [canonical for _, canonical in LABELS[:6]]
    texts = [response(value) for value in labels]
    token_ids, mask = encode_references(Tokenizer(), texts, labels, 1024)
    for i, text in enumerate(texts):
        length = int(mask[i].sum())
        assert token_ids[i, :length - 1].tolist() == [ord(c) + 3 for c in text]
        assert token_ids[i, length - 1].item() == Tokenizer.eos_token_id

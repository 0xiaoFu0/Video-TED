"""Class-only baseline or binary class-and-complete-protocol success."""
import re

from verl.ar.response_format import protocol_valid

REWARD_NAME = 'action_recognition'
REWARD_TYPE = 'batch'


def normalize_label(value):
    """Use the annotation class spelling, including meaningful punctuation.

    Existing alphabetic underscore labels keep their previous normalization.
    Punctuation is not removed: distinct classes and negation must stay distinct.
    """
    if not isinstance(value, str):
        raise ValueError('Action target must be a string')
    if any(ord(c) < 32 or ord(c) == 127 for c in value.strip()):
        raise ValueError('Control character in action label')
    value = value.translate(str.maketrans({'\u2018': "'", '\u2019': "'",
                                          '\u201c': '"', '\u201d': '"'}))
    value = re.sub(r'[\s_]+', '_', value.strip().lower())
    if (len(value) > 300 or not re.search(r'[a-z]', value)
            or not re.fullmatch(r'''[a-z0-9_,.'"()&/+!?;:\[\]-]+''', value)):
        raise ValueError('Invalid action target')
    return value


def parse_answer(response):
    if not isinstance(response, str):
        return None
    matches = list(re.finditer(r'<answer>\s*([^<>]+?)\s*</answer>', response))
    if len(matches) != 1 or response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return None
    if response[matches[0].end():].strip():
        return None
    try:
        return normalize_label(matches[0][1])
    except ValueError:
        return None


def target_label(value):
    if not isinstance(value, str):
        raise ValueError('Action target must be a string')
    label = parse_answer(value) if '<' in value else normalize_label(value)
    if not label:
        raise ValueError('Invalid action target')
    return label


def compute_score(reward_inputs, reward_mode='answer_only', **kwargs):
    if reward_mode not in ('answer_only', 'protocol'):
        raise ValueError('AR reward_mode must be answer_only or protocol')
    if kwargs:
        raise ValueError('Unsupported AR reward kwargs: '+', '.join(sorted(kwargs)))
    results=[]
    for row in reward_inputs:
        prediction=parse_answer(row['response'])
        correct=float(prediction == target_label(row['ground_truth']))
        # Label syntax belongs to the existing full-data parser, including
        # meaningful punctuation; the formatter checks the response envelope.
        formatted=float(prediction is not None and protocol_valid(row['response']))
        results.append(dict(overall=correct if reward_mode == 'answer_only' else correct*formatted,
                            accuracy=correct, answer_valid=float(prediction is not None),
                            protocol_valid=formatted))
    return results

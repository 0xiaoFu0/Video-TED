"""The shared AR assistant format: an indented hierarchy in think, then answer."""
import re


def protocol_think(text):
    """Return the nonempty reasoning body of a complete generated envelope.

    This checks syntax only, without judging indentation, tree structure or
    evidence quality. Label validity is checked by the reward's full-data
    parser, so punctuation and digits in action labels remain supported.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    structural_tags = re.findall(r'<\s*/?\s*(?:think|answer|action_tree|thought)\b[^>]*>',
                                 text, re.IGNORECASE)
    if len(structural_tags) != 4:
        return None
    try:
        validate_reference_format(text)
    except ValueError:
        return None
    match = re.fullmatch(r'<think>(.*?)</think>\s*<answer>([^<>]+)</answer>\s*',
                        text, re.DOTALL)
    if re.search(r'</?[A-Za-z][A-Za-z0-9_:-]*(?:\s[^<>]*)?\s*/?>', match[1]):
        return None
    return match[1]


def protocol_valid(text):
    """Check the envelope; the reward additionally requires a valid label."""
    return protocol_think(text) is not None


def validate_reference_format(text):
    """Validate the envelope without rewriting the hierarchy or its whitespace."""
    if not isinstance(text, str):
        raise ValueError('Reference must be text')
    for tag in ('<think>', '</think>', '<answer>', '</answer>'):
        if text.count(tag) != 1:
            raise ValueError('Reference requires one <think>...</think><answer>...</answer> pair')
    if any(tag in text for tag in ('<action_tree>', '</action_tree>', '<thought>', '</thought>')):
        raise ValueError('Assistant reference must use <think>, not legacy reasoning tags')
    match = re.fullmatch(r'<think>(.*?)</think>\s*<answer>([^<>]+)</answer>\s*', text, re.DOTALL)
    if match is None or not match[1].strip() or not match[2].strip():
        raise ValueError('Expected nonempty think followed by the final answer')


def validate_student_format(prompt):
    if any(tag in prompt for tag in ('<action_tree>', '</action_tree>', '<thought>', '</thought>')):
        raise ValueError('Student output instruction must use <think>, not legacy reasoning tags')
    if any(tag not in prompt for tag in ('<think>', '</think>', '<answer>', '</answer>')):
        raise ValueError('Student prompt must request <think>...</think><answer>...</answer>')

"""Trim oldest text before multimodal encoding, preserving the media skeleton.

Expanded Qwen video tokens cannot be cropped like ordinary text. Both rollout
and actor must receive the same fitted prompt and the same complete media.
"""
import re


def _protected_spans(prompt, tokenizer):
    literals = set(getattr(tokenizer, 'all_special_tokens', ()) or ())
    literals.update(('<|im_start|>', '<|im_end|>', '<|vision_start|>',
                     '<|vision_end|>', '<|image_pad|>', '<|video_pad|>'))
    spans = []
    for literal in literals:
        if literal:
            spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(literal), prompt))
    # Protect video timestamps and every placeholder inside a vision block too.
    spans.extend((m.start(), m.end()) for m in re.finditer(
        r'<\|vision_start\|>.*?<\|vision_end\|>', prompt, flags=re.DOTALL))
    # Qwen chat roles belong to the template, rather than removable user text.
    spans.extend((m.start(), m.end()) for m in re.finditer(
        r'<\|im_start\|>[^\n]*\n', prompt))
    return spans


def _remove_spans(prompt, spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    pieces, cursor = [], 0
    for start, end in merged:
        pieces.append(prompt[cursor:start])
        cursor = end
    pieces.append(prompt[cursor:])
    return ''.join(pieces)


def fit_prompt_length(prompt, tokenizer, encode, max_length, initial_inputs=None):
    """Return (prompt, encoded inputs, removed text tokens, original token count).

    ``encode`` reuses already prepared images/video and produces full, unpadded
    model inputs. Character offsets avoid decoding a partial UTF-8 token. No
    sample, frame, vision token, or chat delimiter is removed.
    """
    if max_length < 1:
        raise ValueError('Prompt budget must be positive')
    inputs = initial_inputs if initial_inputs is not None else encode(prompt)
    original_length = int(inputs['input_ids'].shape[-1])
    raw_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    if max(original_length, raw_length) <= max_length:
        return prompt, inputs, 0, original_length
    try:
        offsets = tokenizer(prompt, add_special_tokens=False,
                            return_offsets_mapping=True)['offset_mapping']
    except (NotImplementedError, TypeError, KeyError) as exc:
        raise ValueError('Safe left truncation requires a fast tokenizer with character offsets') from exc
    if hasattr(offsets, 'tolist'):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    protected = _protected_spans(prompt, tokenizer)
    candidates = [(int(start), int(end)) for start, end in offsets
                  if end > start and not any(start < b and end > a for a, b in protected)]
    removed = 0
    fitted = prompt
    while max(int(inputs['input_ids'].shape[-1]), raw_length) > max_length:
        excess = max(int(inputs['input_ids'].shape[-1]), raw_length) - max_length
        new_removed = min(len(candidates), removed + max(1, excess))
        if new_removed == removed:
            raise ValueError(
                f'Complete visual blocks and chat structure exceed prompt budget {max_length}; '
                'increase the budget or explicitly reduce video frames/pixels')
        removed = new_removed
        fitted = _remove_spans(prompt, candidates[:removed])
        inputs = encode(fitted)
        raw_length = len(tokenizer.encode(fitted, add_special_tokens=False))
    return fitted, inputs, removed, original_length

"""Check full-data preservation and reject mixed or malformed assistant formats."""
import hashlib
import json
from pathlib import Path

from jinja2 import Template
import pytest

from verl.ar.batch import encode_references
from verl.ar.convert_think import build
from verl.ar.prepare_data import read_rows, validate_rows
from verl.ar.response_format import validate_reference_format


ROOT = Path(__file__).resolve().parents[2]


def test_all_rows_keep_body_answer_context_metadata_and_original_order():
    manifest = json.loads((ROOT / 'data/think_format_manifest.json').read_text())
    provenance = json.loads((ROOT / 'docs/ar/provenance.json').read_text())
    for kind in ('rlsd', 'sft'):
        old_path = ROOT / f'data/train.{kind}.jsonl'
        new_path = ROOT / f'data/train.{kind}.think.jsonl'
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == provenance['data_sha256'][old_path.name]
        assert hashlib.sha256(new_path.read_bytes()).hexdigest() == manifest['output_sha256'][new_path.name]
        old, new = read_rows(old_path), read_rows(new_path)
        assert len(old) == len(new) == 4676
        for before, after in zip(old, new):
            excluded = {'prompt', 'reference_response'} if kind == 'rlsd' else {'messages'}
            assert {k: v for k, v in before.items() if k not in excluded} == {
                k: v for k, v in after.items() if k not in excluded}
            old_response = before['reference_response'] if kind == 'rlsd' else before['messages'][1]['content']
            new_response = after['reference_response'] if kind == 'rlsd' else after['messages'][1]['content']
            assert old_response.split('<action_tree>', 1)[1].split('</action_tree>', 1)[0] == (
                new_response.split('<think>', 1)[1].split('</think>', 1)[0])
            assert old_response.split('</action_tree>', 1)[1] == new_response.split('</think>', 1)[1]
            prompt = after['prompt'] if kind == 'rlsd' else after['messages'][0]['content']
            assert 'using indentation' in prompt and '<think>...</think>' in prompt
            assert '<action_tree>' not in prompt


@pytest.mark.parametrize('text', [
    '<action_tree>Evidence</action_tree><answer>pushup</answer>',
    '<thought>Evidence</thought><answer>pushup</answer>',
    '<think>Evidence<answer>pushup</answer>',
    '<answer>pushup</answer><think>Evidence</think>',
    '<think><think>Evidence</think></think><answer>pushup</answer>',
    '<think>Evidence</think><answer>pushup</answer><answer>pushup</answer>',
    '<think>Evidence</think><answer>pushup</answer> trailing',
    '<think> </think><answer>pushup</answer>',
    '<think><action_tree>Evidence</action_tree></think><answer>pushup</answer>',
])
def test_reference_batch_and_dataset_reject_invalid_envelopes(text):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def encode(self, *args, **kwargs):
            pytest.fail('Invalid reference must be rejected before tokenization')
    with pytest.raises(ValueError):
        validate_reference_format(text)
    with pytest.raises(ValueError):
        encode_references(Tokenizer(), [text], ['pushup'], 1024)
    row = read_rows(ROOT / 'data/train.rlsd.think.jsonl')[0]
    row['reference_response'] = text
    row['answer'] = 'pushup'
    with pytest.raises(ValueError):
        validate_rows([row])


def test_teacher_keeps_privileged_tree_and_requests_think_hierarchy():
    row = read_rows(ROOT / 'data/train.rlsd.think.jsonl')[0]
    template = Template((ROOT / 'examples/action_recognition/teacher.jinja').read_text())
    rendered = template.render(question=row['prompt'], hint=row['teacher_context'])
    assert row['teacher_context'] in rendered
    assert '<action_tree>' in row['teacher_context']
    assert 'concise indented action hierarchy inside <think>...</think>' in rendered
    assert 'your response must use <think>...</think>' in rendered


def test_conversion_is_reproducible_and_refuses_overwrite(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    for kind in ('sft', 'rlsd'):
        row = read_rows(ROOT / f'data/train.{kind}.jsonl')[0]
        (sources / f'train.{kind}.jsonl').write_text(json.dumps(row) + '\n')
    report = build(sources, tmp_path / 'new')
    assert report['rows'] == report['classes'] == 1
    with pytest.raises(ValueError, match='already exists'):
        build(sources, tmp_path / 'new')
    for kind in ('sft', 'rlsd'):
        generated = read_rows(tmp_path / f'new/train.{kind}.think.jsonl')[0]
        assert generated == read_rows(ROOT / f'data/train.{kind}.think.jsonl')[0]

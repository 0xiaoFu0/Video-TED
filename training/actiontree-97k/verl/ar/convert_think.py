"""Convert legacy AR exports to think envelopes, preserving every hierarchy body."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .prepare_data import read_rows, validate_rows
from .response_format import validate_reference_format, validate_student_format


def convert_prompt(prompt):
    old = 'in a concise <action_tree> using indentation:'
    if prompt.count(old) != 1:
        raise ValueError('Expected the legacy AR hierarchy instruction')
    prompt = prompt.replace(old, 'in a concise hierarchy inside <think>...</think>, using indentation:')
    validate_student_format(prompt)
    return prompt


def convert_response(response):
    if response.count('<action_tree>') != 1 or response.count('</action_tree>') != 1:
        raise ValueError('Expected one legacy action_tree envelope')
    # No stripping, reflowing or rewriting: all body characters stay identical.
    converted = response.replace('<action_tree>', '<think>').replace('</action_tree>', '</think>')
    validate_reference_format(converted)
    return converted


def convert_row(row):
    result = copy.deepcopy(row)
    if 'messages' in row:
        if [m['role'] for m in row['messages']] != ['user', 'assistant']:
            raise ValueError('Expected one user/assistant pair')
        result['messages'][0]['content'] = convert_prompt(row['messages'][0]['content'])
        result['messages'][1]['content'] = convert_response(row['messages'][1]['content'])
    else:
        result['prompt'] = convert_prompt(row['prompt'])
        result['reference_response'] = convert_response(row['reference_response'])
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_dir, out):
    source_dir, out = Path(source_dir), Path(out)
    sources = {kind: source_dir / f'train.{kind}.jsonl' for kind in ('rlsd', 'sft')}
    outputs = {kind: out / f'train.{kind}.think.jsonl' for kind in sources}
    manifest_path = out / 'think_format_manifest.json'
    if any(p.exists() for p in [*outputs.values(), manifest_path]):
        raise ValueError('Think output already exists; choose a new directory to preserve it')
    source_hashes = {p.name: digest(p) for p in sources.values()}
    arrays = {kind: [convert_row(row) for row in read_rows(path)] for kind, path in sources.items()}
    stats = {kind: validate_rows(rows) for kind, rows in arrays.items()}
    if len(arrays['sft']) != len(arrays['rlsd']):
        raise ValueError('SFT/RLSD count mismatch')
    for s, r in zip(arrays['sft'], arrays['rlsd']):
        if (s['messages'][0]['content'] != r['prompt']
                or s['messages'][1]['content'] != r['reference_response']
                or any(s[key] != r[key] for key in ('videos', 'answer', 'metadata'))):
            raise ValueError('SFT/RLSD pair mismatch')
    out.mkdir(parents=True, exist_ok=True)
    for kind, path in outputs.items():
        with path.open('x', encoding='utf-8') as handle:
            for row in arrays[kind]:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    if source_hashes != {p.name: digest(p) for p in sources.values()}:
        raise ValueError('Source files changed during conversion')
    report = dict(
        schema_version='ar_hierarchy_think_v1', **stats['rlsd'],
        source_sha256=source_hashes,
        output_sha256={p.name: digest(p) for p in outputs.values()},
        converter_sha256=digest(Path(__file__)),
        original_order_preserved=True, hierarchy_body_preserved=True,
        teacher_context_preserved=True, labels_and_media_preserved=True,
        reference_status_preserved=True, filtered_rows=0, resplit=False,
    )
    with manifest_path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('data'))
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

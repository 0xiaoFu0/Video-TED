"""Read-only validation of the 97,430-row RLSD/SFT action merge."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
from uuid import uuid4

from .check_cluster import sha256_file
from .prepare_data import validate_rows
from .response_format import protocol_valid

BASE_BATCHES = {'V1': 10454, 'LLaVA': 2905, 'K700': 46523, 'SSv2': 33701}
ADDED_DATASETS = {'ucf101': 3796, 'hmdb51': 51}
BASE_ROWS = sum(BASE_BATCHES.values())
EXPECTED_BATCHES = {**BASE_BATCHES, **{k + '_new': v for k, v in ADDED_DATASETS.items()}}
PAYLOADS = ('train.rlsd.think.jsonl', 'train.sft.think.jsonl')


def reward_validator_identity():
    """Bind a prepared report to the exact protocol and label validators."""
    directory = Path(__file__).resolve().parent
    return {name: sha256_file(directory / name)
            for name in ('response_format.py', 'reward.py')}


def tokenizer_identity(model):
    model = Path(model).resolve()
    names = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.json', 'merges.txt',
             'added_tokens.json', 'tokenizer.model')
    hashes = {name: sha256_file(model / name) for name in names if (model / name).is_file()}
    if 'config.json' not in hashes or not {'tokenizer.json', 'vocab.json', 'tokenizer.model'} & hashes.keys():
        raise ValueError('Missing local model configuration/tokenizer files: ' + str(model))
    return {'path': str(model), 'sha256': hashes}


def input_identity(data_root):
    root = Path(data_root).resolve()
    return {'path': str(root), 'sha256': {
        name: sha256_file(root / name) for name in (*PAYLOADS, 'manifest.json')}}


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{number}: expected JSON object')
                yield row


def validate_dataset(data_root, token_count, expected_batches=None):
    """Stream both exports; measure reference lengths without changing any row."""
    root = Path(data_root).resolve()
    expected_batches = EXPECTED_BATCHES if expected_batches is None else expected_batches
    identity = input_identity(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    expected_rows = sum(expected_batches.values())
    if manifest['total_rows'] != expected_rows:
        raise ValueError('Merged manifest total_rows does not match the configured dataset')
    if (manifest.get('completed') is not True or manifest.get('base_rows') != BASE_ROWS
            or manifest.get('added_rows') != sum(ADDED_DATASETS.values())
            or manifest.get('added_dataset_counts') != ADDED_DATASETS):
        raise ValueError('Expected the completed original-plus-UCF101/HMDB51 merge')
    base_hashes = [value for name, value in manifest.get('input_sha256', {}).items()
                   if Path(name).name == PAYLOADS[0]]
    if (len(base_hashes) != 1 or not isinstance(base_hashes[0], str)
            or len(base_hashes[0]) != 64
            or any(c not in '0123456789abcdef' for c in base_hashes[0])):
        raise ValueError('Missing original training SHA256 binding')
    base_sha256 = base_hashes[0]
    for name in PAYLOADS:
        if manifest['output_sha256'][name] != identity['sha256'][name]:
            raise ValueError('Merged payload SHA256 mismatch: ' + name)

    seen, videos, classes = set(), set(), set()
    batches, datasets, statuses = Counter(), Counter(), Counter()
    probes, longest, rows = {}, 0, 0
    for r, s in zip_longest(read_rows(root / PAYLOADS[0]), read_rows(root / PAYLOADS[1])):
        if r is None or s is None:
            raise ValueError('RLSD/SFT row counts differ')
        sid = r['metadata']['sample_id']
        try:
            validate_rows([r], check_media=True)
            validate_rows([s], check_media=False)
            if (s['messages'] != [{'role': 'user', 'content': r['prompt']},
                                  {'role': 'assistant', 'content': r['reference_response']}]
                    or any(s[key] != r[key] for key in ('answer', 'videos', 'metadata'))):
                raise ValueError('RLSD/SFT pair content mismatch')
            if not isinstance(sid, str) or not sid or sid in seen:
                raise ValueError('Missing or duplicate sample ID')
            video = Path(r['videos'][0])
            if not video.is_absolute() or video.stat().st_size == 0:
                raise ValueError('Video must be an absolute path to a nonempty file')
            real_video = str(video.resolve())
            if real_video in videos:
                raise ValueError('Duplicate resolved video path')
            if r['reference_status'] not in {'unreviewed', 'reviewed', 'verified'}:
                raise ValueError('Reference is not eligible for the CoT loss')
            if not r['reference_response'].strip():
                raise ValueError('Missing generated reference tree')
            if not protocol_valid(r['reference_response']):
                raise ValueError('Reference does not satisfy the complete think/answer reward protocol')
            if rows < BASE_ROWS:
                batch = r['metadata']['merge_batch']
                if batch not in BASE_BATCHES:
                    raise ValueError('Unknown original merge_batch: ' + str(batch))
            else:
                dataset = r['metadata']['dataset']
                if dataset not in ADDED_DATASETS:
                    raise ValueError('Unknown supplemental dataset: ' + str(dataset))
                if r['metadata'].get('actual_93583_sha256') != base_sha256:
                    raise ValueError('Supplement bound to a different original training set')
                if r['metadata'].get('sampling_profile') != {'mode': 'fps', 'fps': 2.0, 'max_frames': 16}:
                    raise ValueError('Supplement requires a different training video sampler')
                batch = dataset + '_new'
            length = token_count(r['reference_response']) + 1  # Training appends EOS.
            if type(length) is not int or length < 2:
                raise ValueError('Invalid reference token length')
        except (KeyError, ValueError, FileNotFoundError) as error:
            raise ValueError(f'Row {rows + 1}, sample {sid}: {error}') from error
        seen.add(sid)
        videos.add(real_video)
        classes.add(r['answer'])
        batches[batch] += 1
        datasets[r['metadata']['dataset']] += 1
        statuses[r['reference_status']] += 1
        probes.setdefault(batch, (r, s))
        longest = max(longest, length)
        rows += 1
        if rows % 10000 == 0:
            print(f'Validated {rows}/{expected_rows} paired rows; max reference tokens={longest}', flush=True)
    if rows != expected_rows or dict(batches) != expected_batches:
        raise ValueError('Actual paired row/source counts do not match the manifest')
    if input_identity(root) != identity:
        raise ValueError('Dataset changed during preflight')
    report = {'schema_version': 2, 'input': identity, 'rows': rows, 'batches': dict(batches),
              'datasets': dict(datasets), 'classes': len(classes), 'reference_statuses': dict(statuses),
              'reward_mode': 'protocol', 'reference_protocol_valid_rows': rows,
              'reward_validators': reward_validator_identity(),
              'max_reference_tokens_including_eos': longest,
              'reference_max_length': max(1024, longest),
              'video_check': 'All paths exist and are nonempty; decoding is checked separately on representative videos.'}
    return report, [probes[name] for name in expected_batches]


def check_report(data_root, model, report_path):
    path = Path(report_path)
    if not path.is_file():
        raise ValueError('Missing data preflight report; run run_merged_97430_32gpu.sh prepare first')
    report = json.loads(path.read_text(encoding='utf-8'))
    if (report.get('schema_version') != 2 or report['rows'] != sum(EXPECTED_BATCHES.values())
            or report['batches'] != EXPECTED_BATCHES
            or report['input'] != input_identity(data_root)
            or report['tokenizer'] != tokenizer_identity(model)
            or report.get('reward_mode') != 'protocol'
            or report.get('reference_protocol_valid_rows') != report['rows']
            or report.get('reward_validators') != reward_validator_identity()):
        raise ValueError('Data/tokenizer/reward validator changed or wrong dataset report; run prepare again')
    for name, expected in report['probe_sha256'].items():
        if Path(name).name != name or sha256_file(path.parent / name) != expected:
            raise ValueError('Video probe payload changed; run prepare again')
    maximum = report['reference_max_length']
    if type(maximum) is not int or maximum < max(1024, report['max_reference_tokens_including_eos']):
        raise ValueError('Invalid reference budget in preflight report')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--check-report', action='store_true')
    args = p.parse_args()
    if args.check_report:
        report = check_report(args.data_root, args.model, args.report)
        print(report['reference_max_length'])
        return
    from transformers import AutoTokenizer
    identity = tokenizer_identity(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer must define EOS and PAD')
    report, probes = validate_dataset(
        args.data_root, lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    report['tokenizer'] = identity
    args.report.parent.mkdir(parents=True, exist_ok=True)
    probe_hashes = {}
    for index, name in enumerate(('probe.rlsd.think.jsonl', 'probe.sft.think.jsonl')):
        payload = args.report.parent / name
        temporary = payload.with_name(payload.name + '.tmp-' + uuid4().hex)
        temporary.write_text(''.join(json.dumps(pair[index], ensure_ascii=False) + '\n'
                                     for pair in probes), encoding='utf-8')
        temporary.replace(payload)
        probe_hashes[name] = sha256_file(payload)
    report['probe_sha256'] = probe_hashes
    temporary = args.report.with_name(args.report.name + '.tmp-' + uuid4().hex)
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Merge disjoint completed annotation pools into paired qualified/full CoT data.

This is a CPU-only conversion. All selected records and rollout provenance are
validated, errors are omitted without retry, and verification-failed trees are
retained unchanged with the original QA label in the full group.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from .common import ROOT, atomic_json, digest, file_hash, read_json, rows, write_rows
from .export import make_rlsd, validate_rlsd
from .report import shard_completion
from .rollouts import settings, validate_aggregate


def _source_pool(name, candidates_path, run_dir):
    candidates_path, run_dir = Path(candidates_path).resolve(), Path(run_dir).resolve()
    candidate_sha = file_hash(candidates_path)
    preparation = read_json(candidates_path.parent / 'manifest.json')
    if candidate_sha != preparation['candidates_sha256']:
        raise ValueError('Candidate file differs from preparation manifest: ' + name)
    run = read_json(run_dir / 'run.json')
    if digest(run['spec']) != run['run_hash']:
        raise ValueError('Run manifest hash mismatch: ' + name)
    if candidate_sha != run['spec']['candidate_file_sha256']:
        raise ValueError('Candidate file differs from annotation run: ' + name)
    missing = shard_completion(run, run_dir)
    if missing:
        raise ValueError(f'Incomplete global shards for {name}: {missing}')
    candidates = {}
    for _, candidate in rows(candidates_path):
        sid = candidate['sample_id']
        if (sid in candidates or
                digest({k: v for k, v in candidate.items() if k != 'candidate_hash'}) != candidate['candidate_hash']):
            raise ValueError('Duplicate or invalid candidate: ' + name + '/' + sid)
        candidates[sid] = candidate
    selected = run['spec']['selected_ids']
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= candidates.keys():
        raise ValueError('Invalid selection manifest: ' + name)
    source = dict(name=name, candidates=str(candidates_path), run=str(run_dir),
                  candidate_file_sha256=candidate_sha,
                  candidate_manifest_sha256=file_hash(candidates_path.parent / 'manifest.json'),
                  run_json_sha256=file_hash(run_dir / 'run.json'), run_hash=run['run_hash'],
                  planned=len(selected), candidate_rows=len(candidates),
                  summary_sha256={p.name: file_hash(p) for p in sorted(run_dir.glob('summary-*.json'))})
    return candidates_path, run_dir, run, candidates, preparation, source


def build_merged_comparison(pools, out_dir, token_count, expect_qualified=None,
                            expect_full=None, expect_planned=None, student_model=None):
    """Validate every source, then publish one immutable output directory."""
    out_dir = Path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError('Choose a NEW merged comparison output directory')
    expected = dict(qualified=expect_qualified, full=expect_full, planned=expect_planned)
    for name, value in expected.items():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Expected ' + name + ' count must be a nonnegative integer')
    pools = list(pools)
    if not pools:
        raise ValueError('At least one source pool is required')
    names, selected_ids, selected_media, loaded = set(), {}, {}, []
    for name, candidates_path, run_dir in pools:
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in names:
            raise ValueError('Source pool names must be unique simple names')
        names.add(name)
        candidate_root, run_root = Path(candidates_path).resolve().parent, Path(run_dir).resolve()
        if any(root == out_dir.resolve() or root in out_dir.resolve().parents for root in (candidate_root, run_root)):
            raise ValueError('Merged output must be outside every source candidate/run directory')
        pool = _source_pool(name, candidates_path, run_dir)
        _, _, run, candidates, _, _ = pool
        # Include error candidates in this collision check: omitted errors must
        # not silently conceal overlapping or conflicting input pools.
        for sid in run['spec']['selected_ids']:
            candidate = candidates[sid]
            if sid in selected_ids:
                raise ValueError(f'Duplicate selected sample_id across pools: {sid} ({selected_ids[sid]}, {name})')
            selected_ids[sid] = name
            media = str(Path(candidate['video']).resolve())
            if media in selected_media:
                raise ValueError(f'Duplicate selected resolved media across pools: {media} '
                                 f'({selected_media[media]}, {name}/{sid})')
            selected_media[media] = name + '/' + sid
        loaded.append(pool)

    qualified, full, audit, sources = [], [], [], []
    counts, max_tokens, restored = Counter(), 0, False
    for candidates_path, run_dir, run, candidates, preparation, source in loaded:
        source_counts, by_dataset, record_hashes, rollout_hashes = Counter(), defaultdict(Counter), [], []
        source_max = 0
        restored |= preparation.get('scope') == 'all-missing'
        for sid in run['spec']['selected_ids']:
            candidate = candidates[sid]
            record_path = run_dir / 'records' / (sid + '.json')
            record = read_json(record_path)
            if (record['sample_id'] != sid or record['run_hash'] != run['run_hash'] or
                    record['candidate_hash'] != candidate['candidate_hash']):
                raise ValueError('Record provenance mismatch: ' + source['name'] + '/' + sid)
            status = record['status']
            if status not in ('accepted', 'needs_review', 'error'):
                raise ValueError('Unknown annotation status: ' + sid)
            # Even an omitted error must have a full, internally consistent
            # saved rollout set. Semantic review/error counts do not block export.
            validate_aggregate(record, candidate, run['spec']['config'], run['run_hash'], run_dir)
            record_sha = file_hash(record_path)
            draw_shas = [file_hash(run_dir / 'rollouts' / sid / f'{i:04d}.json')
                         for i in range(settings(run['spec']['config'])[0])]
            record_hashes.append([sid, record_sha])
            rollout_hashes.append([sid, draw_shas])
            counts[status] += 1
            source_counts[status] += 1
            by_dataset[candidate['dataset']][status] += 1
            decision = dict(sample_id=sid, source_pool=source['name'], dataset=candidate['dataset'],
                            video=candidate['video'], answer=candidate['answer'], status=status,
                            candidate_hash=candidate['candidate_hash'], source_run_hash=run['run_hash'],
                            record_sha256=record_sha, rollout_files_sha256=draw_shas,
                            qualified=status == 'accepted', full=status != 'error',
                            rollout_summary=record['rollout_summary'])
            if status == 'error':
                decision.update(reference_route='omitted_error', error=record.get('error'))
                audit.append(decision)
                continue
            media = Path(candidate['video'])
            if not media.is_file() or file_hash(media) != record['video_sha256']:
                raise ValueError('Video changed/missing since annotation: ' + sid)
            row = make_rlsd(candidate, record)
            row['metadata'].update(source_pool=source['name'], annotation_status=status,
                annotation_candidate_hash=candidate['candidate_hash'], annotation_record_sha256=record_sha,
                annotation_rollout_files_sha256_json=json.dumps(draw_shas),
                annotation_evidence_json=json.dumps(record['evidence'], sort_keys=True),
                annotation_verdict_json=json.dumps(record['verdict'], sort_keys=True),
                annotation_verification_options_json=json.dumps(record['verification_options']),
                reference_answer_policy='original_qa_label',
                evidence_status='accepted_generated_annotation' if status == 'accepted'
                                else 'retained_unaccepted_generated_annotation')
            row['reference_status'] = 'unreviewed'
            validate_rlsd(row)
            token_length = token_count(row['reference_response'])
            if type(token_length) is not int or token_length < 0:
                raise ValueError('Tokenizer count must be a nonnegative integer')
            length = token_length + 1
            max_tokens, source_max = max(max_tokens, length), max(source_max, length)
            row['metadata']['reference_tokens_including_eos'] = length
            decision.update(reference_route='accepted_tree' if status == 'accepted' else 'unaccepted_tree',
                            reference_tokens_including_eos=length,
                            selected_rollout=record['rollout_summary']['selected_rollout'],
                            sufficient_evidence=record['evidence']['sufficient_evidence'],
                            verification_options=record['verification_options'], verdict=record['verdict'],
                            video_sha256=record['video_sha256'], reference_answer_policy='original_qa_label')
            if status == 'accepted':
                qualified.append(row)
            full.append(row)
            audit.append(decision)
        source.update(statuses=dict(source_counts), datasets={k: dict(v) for k, v in by_dataset.items()},
                      qualified_rows=source_counts['accepted'],
                      full_rows=source_counts['accepted'] + source_counts['needs_review'],
                      max_observed_reference_tokens=source_max,
                      records_sha256=digest(record_hashes), rollouts_sha256=digest(rollout_hashes))
        sources.append(source)

    actual = dict(qualified=len(qualified), full=len(full), planned=len(selected_ids))
    for name, required in expected.items():
        if required is not None and actual[name] != required:
            raise ValueError(f'Expected {name} rows {required}, found {actual[name]}; no output published')
    if not qualified:
        raise ValueError('No accepted samples for the qualified training group')
    report = dict(merge_schema_version=1, planned=actual['planned'], statuses=dict(counts),
                  qualified_rows=actual['qualified'], full_rows=actual['full'],
                  qualified_by_dataset=dict(Counter(r['metadata']['dataset'] for r in qualified)),
                  full_by_dataset=dict(Counter(r['metadata']['dataset'] for r in full)),
                  sources=sources, source_counts={s['name']: {k: s[k] for k in
                      ('planned', 'statuses', 'qualified_rows', 'full_rows')} for s in sources},
                  expected_counts=expected, student_model=student_model,
                  reference_max_length=max(1024, max_tokens), max_observed_reference_tokens=max_tokens,
                  qualified_selection='first_accepted_rollout_per_sample',
                  full_selection='first_accepted_rollout_else_first_nonerror_rollout_per_sample',
                  full_reference_policy='all_nonerror_generated_trees_with_reference_loss',
                  reference_answer_policy='original_qa_label', unaccepted_sft_rows=counts['needs_review'],
                  evaluation_overlap_policy='retain_all_existing_candidates',
                  evaluation_overlap_counts=dict(Counter(b for r in full
                      for b in json.loads(r['metadata'].get('evaluation_overlaps_json', '[]')))),
                  error_policy='omit_without_retry', earlier_preparation_exclusions_restored=restored,
                  duplicate_policy='reject_selected_sample_id_or_resolved_media_collisions',
                  note='All selected source/rollout provenance is validated; exported media hashes are verified. '
                       'Errors are omitted without retry. Full retains verification-failed trees unchanged '
                       'with original QA final labels. No model generation or new overlap exclusions. '
                       'output_sha256 covers payloads; SHA256SUMS also covers manifest.json.')

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.merged-comparison-', dir=out_dir.parent) as temporary:
        stage = Path(temporary) / 'output'
        stage.mkdir()
        for name, values in (('qualified', qualified), ('full', full)):
            directory = stage / name
            directory.mkdir()
            write_rows(directory / 'train.rlsd.think.jsonl', values)
            write_rows(directory / 'train.sft.think.jsonl', [dict(
                messages=[dict(role='user', content=r['prompt']),
                          dict(role='assistant', content=r['reference_response'])],
                videos=r['videos'], answer=r['answer'], metadata=r['metadata']) for r in values])
            atomic_json(directory / 'sample_ids.json', [r['metadata']['sample_id'] for r in values])
        write_rows(stage / 'selection.audit.jsonl', audit)
        report['output_sha256'] = {str(p.relative_to(stage)): file_hash(p)
                                   for p in sorted(stage.rglob('*')) if p.is_file()}
        atomic_json(stage / 'manifest.json', report)
        with (stage / 'SHA256SUMS').open('x', encoding='utf-8') as stream:
            for p in sorted(stage.rglob('*')):
                if p.is_file() and p.name != 'SHA256SUMS':
                    stream.write(file_hash(p) + '  ' + str(p.relative_to(stage)) + '\n')
        # Publish only after validation/count checks and all files are complete.
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError('Choose a NEW merged comparison output directory')
        stage.rename(out_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', action='append', nargs=3, metavar=('NAME', 'CANDIDATES', 'RUN'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=str(ROOT / 'config.server.json'))
    parser.add_argument('--student-model')
    parser.add_argument('--expect-qualified', type=int)
    parser.add_argument('--expect-full', type=int)
    parser.add_argument('--expect-planned', type=int)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    model = args.student_model or read_json(args.config)['student_model']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    report = build_merged_comparison(args.pool, args.out,
        lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        expect_qualified=args.expect_qualified, expect_full=args.expect_full,
        expect_planned=args.expect_planned, student_model=model)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

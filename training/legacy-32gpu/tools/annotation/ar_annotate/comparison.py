"""Build filtered-CoT and all-generated-CoT training datasets from an existing run.

Does not filter evaluation overlap, change source labels, regenerate annotations,
or discard a structurally valid tree because verification failed. Error samples
are omitted. Both groups contain SFT and RLSD references with original QA labels.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .common import ROOT, atomic_json, digest, file_hash, read_json, rows, write_rows
from .export import make_rlsd, validate_rlsd
from .report import shard_completion
from .rollouts import validate_aggregate


def build_comparison(candidates_path, run_dir, out_dir, token_count):
    candidates_path, run_dir, out_dir = map(Path, (candidates_path, run_dir, out_dir))
    if out_dir.exists():
        raise FileExistsError('Choose a NEW comparison output directory')
    run = read_json(run_dir / 'run.json')
    if file_hash(candidates_path) != run['spec']['candidate_file_sha256']:
        raise ValueError('Candidate file differs from annotation run')
    missing_shards = shard_completion(run, run_dir)
    if missing_shards:
        raise ValueError(f'Incomplete global shards: {missing_shards}')
    candidates = {}
    for _, row in rows(candidates_path):
        sid = row['sample_id']
        if sid in candidates or digest({k: v for k, v in row.items() if k != 'candidate_hash'}) != row['candidate_hash']:
            raise ValueError('Duplicate or invalid candidate: ' + sid)
        candidates[sid] = row
    selected = run['spec']['selected_ids']
    if len(selected) != len(set(selected)) or not set(selected) <= candidates.keys():
        raise ValueError('Invalid selection manifest')
    qualified, full, audit, counts, seen_media = [], [], [], Counter(), set()
    max_tokens = 0
    for sid in selected:
        candidate = candidates[sid]
        path = run_dir / 'records' / (sid + '.json')
        record = read_json(path)
        if (record['sample_id'] != sid or record['run_hash'] != run['run_hash']
                or record['candidate_hash'] != candidate['candidate_hash']):
            raise ValueError('Record provenance mismatch: ' + sid)
        status = record['status']
        if status not in ('accepted', 'needs_review', 'error'):
            raise ValueError('Unknown annotation status: ' + sid)
        counts[status] += 1
        decision = dict(sample_id=sid, dataset=candidate['dataset'], answer=candidate['answer'],
                        status=status, record_sha256=file_hash(path),
                        qualified=status == 'accepted', full=status != 'error')
        if status == 'error':
            decision['reference_route'] = 'omitted_error'
            audit.append(decision)
            continue
        validate_aggregate(record, candidate, run['spec']['config'], run['run_hash'], run_dir)
        media = Path(candidate['video'])
        if not media.is_file() or file_hash(media) != record['video_sha256']:
            raise ValueError('Video changed/missing since annotation: ' + sid)
        if str(media.resolve()) in seen_media:
            raise ValueError('Duplicate candidate media: ' + sid)
        seen_media.add(str(media.resolve()))
        row = make_rlsd(candidate, record)
        row['metadata']['annotation_status'] = status
        row['metadata']['annotation_evidence_json'] = json.dumps(record['evidence'], sort_keys=True)
        row['metadata']['annotation_verdict_json'] = json.dumps(record['verdict'], sort_keys=True)
        row['metadata']['annotation_verification_options_json'] = json.dumps(record['verification_options'])
        row['metadata']['reference_answer_policy'] = 'original_qa_label'
        # Verification status is retained for auditing, not used as a loss gate
        # in the unfiltered group. Never claim these generated trees were reviewed.
        row['reference_status'] = 'unreviewed'
        row['metadata']['evidence_status'] = ('accepted_generated_annotation' if status == 'accepted'
                                              else 'retained_unaccepted_generated_annotation')
        validate_rlsd(row)
        length = token_count(row['reference_response']) + 1  # Include EOS.
        max_tokens = max(max_tokens, length)
        row['metadata']['reference_tokens_including_eos'] = length
        decision.update(reference_route='accepted_tree' if status == 'accepted' else 'unaccepted_tree',
                        reference_tokens_including_eos=length,
                        selected_rollout=record['rollout_summary']['selected_rollout'],
                        sufficient_evidence=record['evidence']['sufficient_evidence'],
                        verification_options=record['verification_options'], verdict=record['verdict'],
                        reference_answer_policy='original_qa_label')
        if status == 'accepted':
            qualified.append(row)
        full.append(row)
        audit.append(decision)
    if not qualified:
        raise ValueError('No accepted samples for the qualified training group')
    out_dir.mkdir(parents=True)
    for name, values in (('qualified', qualified), ('full', full)):
        directory = out_dir / name
        directory.mkdir()
        write_rows(directory / 'train.rlsd.think.jsonl', values)
        sft = [dict(messages=[dict(role='user', content=r['prompt']),
                              dict(role='assistant', content=r['reference_response'])],
                    videos=r['videos'], answer=r['answer'], metadata=r['metadata']) for r in values]
        write_rows(directory / 'train.sft.think.jsonl', sft)
        atomic_json(directory / 'sample_ids.json', [r['metadata']['sample_id'] for r in values])
    write_rows(out_dir / 'selection.audit.jsonl', audit)
    report = dict(planned=len(selected), statuses=dict(counts), qualified_rows=len(qualified), full_rows=len(full),
                  qualified_by_dataset=dict(Counter(r['metadata']['dataset'] for r in qualified)),
                  full_by_dataset=dict(Counter(r['metadata']['dataset'] for r in full)),
                  evaluation_overlap_policy='retain_all_existing_candidates',
                  source_candidate_sha256=file_hash(candidates_path), source_run_hash=run['run_hash'],
                  reference_max_length=max(1024, max_tokens), max_observed_reference_tokens=max_tokens,
                  qualified_selection='first_accepted_rollout_per_sample',
                  full_selection='first_accepted_rollout_else_first_nonerror_rollout_per_sample',
                  full_reference_policy='all_nonerror_generated_trees_with_reference_loss',
                  reference_answer_policy='original_qa_label',
                  unaccepted_sft_rows=counts['needs_review'], error_policy='omit_without_retry',
                  earlier_preparation_exclusions_restored=False,
                  evaluation_overlap_counts=dict(Counter(b for r in full
                      for b in json.loads(r['metadata'].get('evaluation_overlaps_json', '[]')))),
                  output_sha256={str(p.relative_to(out_dir)): file_hash(p)
                                 for p in out_dir.rglob('*') if p.is_file()},
                  note='Full includes verification-failed generated trees as supervision; verification remains audit metadata. '
                       'Original QA labels remain the final answers. No new overlap/class exclusions or length filtering.')
    atomic_json(out_dir / 'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=str(ROOT / 'config.server.json'))
    parser.add_argument('--student-model')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    model = args.student_model or read_json(args.config)['student_model']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    report = build_comparison(args.candidates, args.run, args.out,
                              lambda s: len(tokenizer.encode(s, add_special_tokens=False)))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

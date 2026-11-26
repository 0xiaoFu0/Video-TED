import csv
import json
from collections import Counter
from pathlib import Path

from .common import ROOT, atomic_json, file_hash, hierarchy, label, read_json, rows, validate_verdict, write_rows
from .rollouts import validate_aggregate
from .report import shard_completion


def validate_rlsd(row):
    target = label(row['answer'])
    if len(row['videos']) != 1 or not Path(row['videos'][0]).is_absolute():
        raise ValueError('Expected one absolute video path')
    prompt = row['prompt']
    if prompt.count('<video>') != 1 or any(t not in prompt for t in ('<think>', '</think>', '<answer>', '</answer>')):
        raise ValueError('Invalid student prompt')
    if row['videos'][0] in prompt or 'Reference class:' in prompt:
        raise ValueError('Privileged context in student prompt')
    ref = row['reference_response']
    for tag in ('<think>', '</think>', '<answer>', '</answer>'):
        if ref.count(tag) != 1:
            raise ValueError('Invalid reference envelope')
    if not ref.startswith('<think>') or not ref.endswith(f'<answer>{target}</answer>'):
        raise ValueError('Reference/label mismatch')
    if not row['teacher_context'].strip():
        raise ValueError('Missing teacher context')
    if row['reference_status'] not in ('unreviewed', 'reviewed', 'verified', 'rejected'):
        raise ValueError('Invalid reference status')
    if not row['metadata']['sample_id'] or not row['metadata']['video_relpath']:
        raise ValueError('Missing identity metadata')


def make_rlsd(candidate, record):
    evidence = record['evidence']
    body = hierarchy(evidence)
    target = candidate['answer']
    response = '<think>\n' + body + '\n</think>\n<answer>' + target + '</answer>'
    context = ('Reference class: ' + target + '\n'
               'Class-level prior: not supplied.\n\n'
               'Instance evidence tree (model-generated from sampled video frames; not human-verified):\n'
               '<action_tree>\n' + body + '\n</action_tree>\n\n'
               'Preserve uncertainty. Use the video to resolve conflicts; '
               'do not turn the reference class into additional observed events.')
    metadata = dict(sample_id=candidate['sample_id'], source_row_index=candidate['source_line'] - 1,
                    video_relpath=candidate['video_relpath'], evidence_status='unverified_generated_annotation',
                    video_verified=False, dataset=candidate['dataset'], original_video=candidate['original_video'],
                    raw_label=candidate['raw_label'], match_method=candidate['match_method'],
                    split_evidence=candidate['split_evidence'], annotation_run_hash=record['run_hash'],
                    annotation_model='Qwen3.8-27B', video_sha256=record['video_sha256'],
                    sampling_json=json.dumps(record['sampling'], sort_keys=True))
    if 'rollout_summary' in record:
        metadata['annotation_rollout_index'] = record['rollout_summary']['selected_rollout']
        metadata['annotation_rollout_summary_json'] = json.dumps(record['rollout_summary'], sort_keys=True)
    for field in ('source_sample_id', 'source_annotation', 'source_annotation_sha256', 'source_split'):
        if field in candidate:
            metadata[field] = candidate[field]
    if 'evaluation_overlaps' in candidate:
        metadata['evaluation_overlaps_json'] = json.dumps(candidate['evaluation_overlaps'])
    return dict(prompt=(ROOT / 'prompts/student.txt').read_text().rstrip('\n'),
                videos=[candidate['video']], answer=target, teacher_context=context,
                reference_response=response, reference_status='unreviewed', metadata=metadata)


def export_data(candidates_path, run_dir, base_path, out_dir, token_count, max_reference_tokens=768, allow_partial=False,
                pool_only=False):
    source, run_dir, out_dir = Path(candidates_path), Path(run_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError('Choose a new output directory')
    run = read_json(run_dir / 'run.json')
    missing_shards = shard_completion(run, run_dir)
    if missing_shards and not allow_partial:
        raise ValueError(f'Incomplete global shards: {missing_shards}; finish both nodes before export')
    if file_hash(source) != run['spec']['candidate_file_sha256']:
        raise ValueError('Candidate file differs from annotation run')
    candidates = {r['sample_id']: r for _, r in rows(source)}
    if not pool_only and any(r['dataset'] == 'hmdb51' for r in candidates.values()):
        raise ValueError('Combined HMDB51 annotation requires --pool; do not append old HMDB51 references again')
    selected = run['spec']['selected_ids']
    if len(selected) != len(set(selected)) or not set(selected) <= candidates.keys():
        raise ValueError('Invalid selection manifest')
    base = [] if pool_only else [r for _, r in rows(base_path)]
    if not base and not pool_only:
        raise ValueError('Base training dataset is empty')
    ids, videos = set(), set()
    for row in base:
        validate_rlsd(row)
        sid = row['metadata']['sample_id']
        path = Path(row['videos'][0])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError('Base media missing/empty: ' + str(path))
        resolved = str(path.resolve())
        if sid in ids or resolved in videos:
            raise ValueError('Duplicate identity/media in base dataset')
        ids.add(sid)
        videos.add(resolved)
    added, audit, counts = [], [], Counter()
    for sid in selected:
        candidate = candidates[sid]
        p = run_dir / 'records' / (sid + '.json')
        reason = None
        record = read_json(p) if p.exists() else None
        cfg = run['spec'].get('config', {})
        if record is not None and ('rollouts' in cfg or record.get('record_schema') == 3):
            if record.get('record_schema') != 3:
                raise ValueError('Missing rollout aggregate schema: ' + sid)
            validate_aggregate(record, candidate, cfg, run['run_hash'], run_dir)
        if record is None:
            reason = 'missing'
        elif record['run_hash'] != run['run_hash'] or record['candidate_hash'] != candidate['candidate_hash']:
            raise ValueError('Record provenance mismatch: ' + sid)
        elif record['status'] != 'accepted':
            if record['status'] not in ('error', 'needs_review'):
                raise ValueError('Unknown annotation status')
            reason = record['status']
        else:
            verdict = validate_verdict(record['verdict'], len(record['verification_options']),
                                       require_hierarchy=cfg.get('prompt_profile') == 'tree_v1')
            if (not verdict['evidence_supported'] or verdict['choice'] < 0 or
                    label(record['verification_options'][verdict['choice']]) != candidate['answer'] or
                    not record['evidence']['sufficient_evidence'] or not verdict.get('hierarchy_supported', True)):
                raise ValueError('Accepted record fails verification: ' + sid)
            media = Path(candidate['video'])
            if not media.is_file() or file_hash(media) != record['video_sha256']:
                raise ValueError('Video changed/missing since annotation: ' + sid)
            row = make_rlsd(candidate, record)
            validate_rlsd(row)
            if token_count(row['reference_response']) + 1 > max_reference_tokens:
                reason = 'reference_over_token_budget'
            elif sid in ids or str(media.resolve()) in videos:
                reason = 'duplicate_training_identity_or_media'
            else:
                ids.add(sid)
                videos.add(str(media.resolve()))
                added.append(row)
        counts[reason or 'exported'] += 1
        audit.append(dict(sample_id=sid, dataset=candidate['dataset'], video=candidate['video'],
                          answer=candidate['answer'], decision=reason or 'exported',
                          record_sha256=file_hash(p) if p.exists() else None,
                          rollout_summary=record.get('rollout_summary') if record else None,
                          details=record.get('verdict') if record else None))
    if (counts['missing'] or counts['error']) and not allow_partial:
        raise ValueError(f'Incomplete annotation: {dict(counts)}. Finish/retry first, or explicitly --allow-partial.')
    if not added:
        raise ValueError('No accepted new rows to export')
    merged = base + added
    sft = [dict(messages=[dict(role='user', content=r['prompt']),
                         dict(role='assistant', content=r['reference_response'])],
                videos=r['videos'], answer=r['answer'], metadata=r['metadata']) for r in merged]
    out_dir.mkdir(parents=True)
    prefix = 'pool' if pool_only else 'train'
    write_rows(out_dir / f'{prefix}.rlsd.think.jsonl', merged)
    write_rows(out_dir / f'{prefix}.sft.think.jsonl', sft)
    write_rows(out_dir / 'conversion.audit.jsonl', audit)
    with (out_dir / 'review.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'dataset', 'video', 'label', 'decision', 'reason'])
        for r in audit:
            if r['decision'] != 'exported':
                writer.writerow([r[k] for k in ('sample_id', 'dataset', 'video', 'answer', 'decision')] +
                                [(r['details'] or {}).get('reason', '')])
    report = dict(base_rows=len(base), added_rows=len(added), total_rows=len(merged), pool_only=pool_only,
                  new_dataset_counts=dict(Counter(r['metadata']['dataset'] for r in added)),
                  decisions=dict(counts), selected_for_this_run=len(selected), total_candidates=len(candidates),
                  base_sha256=None if pool_only else file_hash(base_path), run_hash=run['run_hash'],
                  evaluation_overlap_counts=dict(Counter(b for r in added
                      for b in json.loads(r['metadata'].get('evaluation_overlaps_json', '[]')))),
                  reference_max_tokens=max_reference_tokens, allow_partial=allow_partial,
                  missing_shards=missing_shards,
                  output_sha256={p.name: file_hash(p) for p in out_dir.glob('*.jsonl')},
                  note='Automated consistency checks, not human verification. Base/novel class protocol changes after expansion.')
    atomic_json(out_dir / 'manifest.json', report)
    print(report)
    return report

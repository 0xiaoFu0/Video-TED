"""Read-only global progress and shard-completion checks; no GPU dependencies."""
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, file_hash, read_json, rows


def shard_completion(run, run_dir):
    n = run['spec'].get('num_shards')
    if n is None:  # Legacy import/test manifests may not carry shard information.
        return []
    assignments = Counter(int(digest(sid)[:8], 16) % n for sid in run['spec']['selected_ids'])
    missing = []
    for index in range(n):
        p = Path(run_dir) / f'summary-{index}.json'
        if not p.exists():
            missing.append(index)
            continue
        summary = read_json(p)
        if (summary['run_hash'] != run['run_hash'] or summary['assigned'] != assignments[index]
                or sum(summary['counts'].values()) != assignments[index]):
            raise ValueError(f'Shard {index} summary provenance/coverage mismatch')
    return missing


def summarize(run_dir, candidates_path):
    run_dir, candidates_path = Path(run_dir), Path(candidates_path)
    run = read_json(run_dir / 'run.json')
    if file_hash(candidates_path) != run['spec']['candidate_file_sha256']:
        raise ValueError('Candidate file differs from run')
    candidates = {r['sample_id']: r for _, r in rows(candidates_path)}
    counts, draw_counts = defaultdict(Counter), defaultdict(Counter)
    missing, mismatched = [], []
    for sid in run['spec']['selected_ids']:
        row = candidates[sid]
        path = run_dir / 'records' / (sid + '.json')
        if not path.exists():
            missing.append(sid)
            continue
        record = read_json(path)
        if (record['sample_id'] != sid or record['run_hash'] != run['run_hash']
                or record['candidate_hash'] != row['candidate_hash']):
            mismatched.append(sid)
            continue
        counts[row['dataset']][record['status']] += 1
        for status in ('accepted', 'needs_review', 'error'):
            draw_counts[row['dataset']][status] += record.get('rollout_summary', {}).get(status, 0)
    missing_shards = shard_completion(run, run_dir)
    return dict(run=str(run_dir), planned=len(run['spec']['selected_ids']),
                missing_records=len(missing), mismatched_records=len(mismatched),
                missing_shards=missing_shards, datasets=dict(counts), rollouts=dict(draw_counts),
                complete=not (missing or mismatched or missing_shards or any(c['error'] for c in counts.values())),
                note='Read-only progress counts; export performs full rollout and video integrity validation.')

import fcntl
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .backend import QwenBackend, environment, model_identity
from .common import ROOT, atomic_json, digest, file_hash, label, parse_json_response, read_json, rows, validate_evidence, validate_verdict
from .rollouts import aggregate, generation_seed, options_for, settings, validate_aggregate, validate_draw


@contextmanager
def lock(path, wait=False):
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def select(candidates, limit):
    counts, selected = Counter(), []
    for row in sorted(candidates, key=lambda r: digest(r['sample_id'])):
        if limit and counts[row['dataset']] >= limit:
            continue
        counts[row['dataset']] += 1
        selected.append(row)
    return selected


def evidence_prompt(row, cfg):
    profile = cfg.get('prompt_profile', 'legacy')
    if profile == 'legacy':
        return (ROOT / 'prompts/evidence.txt').read_text()
    if profile == 'tree_v1':
        family = 'ssv2' if row['dataset'] == 'ssv2' else 'human'
        return ((ROOT / 'prompts/evidence_tree_v1.txt').read_text() + '\n' +
                (ROOT / f'prompts/evidence_tree_{family}_v1.txt').read_text())
    if profile != 'temporal_v1':
        raise ValueError('Unknown prompt profile')
    family = 'ssv2' if row['dataset'] == 'ssv2' else 'k710'
    return ((ROOT / 'prompts/evidence_temporal_v1.txt').read_text() + '\n' +
            (ROOT / f'prompts/evidence_{family}_v1.txt').read_text())


def annotate_one(row, backend, cfg, rollout_index=0):
    start = time.monotonic()
    record = dict(sample_id=row['sample_id'], candidate_hash=row['candidate_hash'],
                  status='error', attempts=[], rollout_index=rollout_index,
                  rollout_seed=generation_seed(row['sample_id'], cfg, rollout_index, 0, 'evidence'))
    try:
        p = Path(row['video'])
        if p.stat().st_size != row['video_size'] or p.stat().st_mtime_ns != row['video_mtime_ns']:
            raise ValueError('Video changed after preparation; rebuild candidates')
        record['video_sha256'] = file_hash(p)
        video = backend.load_video(str(p))
        record['sampling'] = video['audit']
        options = options_for(row, cfg)
        record['verification_options'] = options
        evidence = None
        feedback = {'evidence': '', 'verification': ''}
        for attempt in range(cfg['attempts']):
            seeds = {stage: generation_seed(row['sample_id'], cfg, rollout_index, attempt, stage)
                     for stage in ('evidence', 'verification')}
            trace = dict(attempt=attempt, seeds=seeds)
            record['attempts'].append(trace)
            stage = 'evidence' if evidence is None else 'verification'
            try:
                record.pop('verdict', None)
                if evidence is None:
                    trace['evidence_format_feedback'] = feedback['evidence']
                    raw = backend.generate(video, evidence_prompt(row, cfg) + feedback['evidence'], seeds['evidence'])
                    trace['evidence_raw'] = raw
                    evidence = validate_evidence(parse_json_response(raw))
                    record['evidence'] = evidence
                else:
                    trace['evidence_reused'] = True
                stage = 'verification'
                verify_file = {'temporal_v1': 'verify_temporal_v1.txt', 'tree_v1': 'verify_tree_v1.txt'}.get(
                    cfg.get('prompt_profile'), 'verify.txt')
                trace['verification_format_feedback'] = feedback['verification']
                question = ((ROOT / 'prompts' / verify_file).read_text() + feedback['verification'] + '\nOptions:\n' +
                            json.dumps(dict(enumerate(options)), ensure_ascii=False) +
                            '\nDraft observations:\n' + json.dumps(evidence, ensure_ascii=False))
                raw = backend.generate(video, question, seeds['verification'])
                trace['verification_raw'] = raw
                verdict = validate_verdict(parse_json_response(raw), len(options),
                                           require_hierarchy=cfg.get('prompt_profile') == 'tree_v1')
                record['verdict'] = verdict
                match = verdict['choice'] >= 0 and label(options[verdict['choice']]) == row['answer']
                record['status'] = ('accepted' if match and evidence['sufficient_evidence'] and
                                    verdict['evidence_supported'] and verdict.get('hierarchy_supported', True)
                                    else 'needs_review')
                # Semantic disagreement is not retried until the model happens to agree.
                break
            except (ValueError, KeyError, TypeError) as e:
                trace['error'] = str(e)
                trace['error_stage'] = stage
                feedback[stage] = ('\nFORMAT RETRY: The previous response failed structural validation: '
                                   + str(e)[:500] + '.\nReturn a complete JSON object with every required field. '
                                   'This is format feedback, not an indication of the correct action or truth value. '
                                   'Keep all uncertainty judgments grounded in the video.\n')
                if stage == 'evidence':
                    feedback[stage] += ('Include activity, branches, uncertainties (an array), and sufficient_evidence '
                                        '(a boolean). Use at most 8 observations in TOTAL across all branches. '
                                        'Select concise action-relevant evidence; do not invent missing events.\n')
        if record['status'] == 'error':
            record['error'] = 'No structurally valid annotation after retries'
    except Exception as e:
        record['error'] = f'{type(e).__name__}: {e}'
    record['seconds'] = round(time.monotonic() - start, 3)
    if hasattr(backend, 'metrics'):
        record['runtime'] = backend.metrics()
    return record


class CachedVideoBackend:
    """Reuse one video's CPU frames across draws; never carry frames to the next video."""
    def __init__(self, backend):
        self.backend, self.video = backend, None

    def load_video(self, path):
        if self.video is None:
            self.video = self.backend.load_video(path)
        return self.video

    def __getattr__(self, name):
        return getattr(self.backend, name)


def annotate_rollouts(row, backend_factory, cfg, out, run_hash, retry_errors=False):
    draws, cached = [], None
    for i in range(settings(cfg)[0]):
        target = out / 'rollouts' / row['sample_id'] / f'{i:04d}.json'
        previous = read_json(target) if target.exists() else None
        if previous is not None:
            validate_draw(previous, row, cfg, run_hash, i)
            if previous['status'] != 'error' or not retry_errors:
                draws.append(previous)
                continue
            atomic_json(out / 'errors_history/rollouts' / row['sample_id'] / f'{i:04d}.{time.time_ns()}.json', previous)
        if cached is None:
            cached = CachedVideoBackend(backend_factory())
        record = annotate_one(row, cached, cfg, i)
        record['run_hash'] = run_hash
        atomic_json(target, record)
        draws.append(record)
        print(json.dumps(dict(sample_id=row['sample_id'], rollout=f'{i+1}/{settings(cfg)[0]}',
                              status=record['status'], seconds=record['seconds'])), flush=True)
    return aggregate(row, draws, cfg, run_hash)


def annotate(args):
    cfg = read_json(args.config)['annotation']
    if args.model:
        cfg['model'] = args.model
    for name in ('rollouts', 'min_accepted_rollouts', 'seed', 'prompt_profile'):
        if getattr(args, name, None) is not None:
            cfg[name] = getattr(args, name)
    n, minimum, seed = settings(cfg)
    cfg.update(rollouts=n, min_accepted_rollouts=minimum, seed=seed)
    cfg.setdefault('prompt_profile', 'legacy')
    if cfg['prompt_profile'] not in ('legacy', 'temporal_v1', 'tree_v1'):
        raise ValueError('Unknown prompt profile')
    if not 0 <= args.shard_index < args.num_shards or args.limit_per_dataset < 0:
        raise ValueError('Invalid shard or sample limit')
    if cfg['attempts'] < 1 or not 4 <= cfg['max_frames'] <= 16 or cfg['fps'] <= 0:
        raise ValueError('Invalid annotation configuration; current training caps frames at 16')
    if not 0 < cfg.get('temperature', 0.7) <= 2 or not 0 < cfg.get('top_p', 0.8) <= 1:
        raise ValueError('Invalid sampling temperature/top_p')
    if type(cfg.get('top_k', 20)) is not int or cfg.get('top_k', 20) < 0:
        raise ValueError('Invalid top_k')
    source = Path(args.candidates)
    prepared = read_json(source.parent / 'manifest.json')
    if file_hash(source) != prepared['candidates_sha256']:
        raise ValueError('Candidate file does not match preparation manifest')
    all_rows = [r for _, r in rows(source)]
    for r in all_rows:
        if r['candidate_hash'] != digest({k: v for k, v in r.items() if k != 'candidate_hash'}):
            raise ValueError('Candidate row hash mismatch')
    requested = getattr(args, 'sample_id', None)
    if requested:
        unknown = set(requested) - {r['sample_id'] for r in all_rows}
        if unknown:
            raise ValueError('Unknown sample IDs: ' + ', '.join(sorted(unknown)))
        all_rows = [r for r in all_rows if r['sample_id'] in set(requested)]
    selected = select(all_rows, args.limit_per_dataset)
    if not selected:
        raise ValueError('No candidates selected')
    spec = dict(candidate_file_sha256=file_hash(source), config=cfg,
                model=model_identity(cfg['model']), versions=environment(),
                code={p.name: file_hash(p) for p in sorted((ROOT / 'ar_annotate').glob('*.py'))},
                prompts={p.name: file_hash(p) for p in sorted((ROOT / 'prompts').glob('*.txt'))},
                selected_ids=[r['sample_id'] for r in selected], num_shards=args.num_shards)
    run_hash = digest(spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'records').mkdir(exist_ok=True)
    with lock(out / 'run.lock', wait=True):
        if (out / 'run.json').exists():
            if read_json(out / 'run.json')['run_hash'] != run_hash:
                raise ValueError('Run configuration/input/model changed. Use a NEW output directory.')
        else:
            atomic_json(out / 'run.json', dict(run_hash=run_hash, spec=spec))
    with lock(out / f'shard-{args.shard_index}.lock'):
        shard_start = time.monotonic()
        shard = [r for r in selected if int(digest(r['sample_id'])[:8], 16) % args.num_shards == args.shard_index]
        backend = None
        def get_backend():
            nonlocal backend
            if backend is None:
                backend = QwenBackend(cfg)
            return backend
        counts = Counter()
        consecutive_errors = 0
        try:
            for index, row in enumerate(shard, 1):
                target = out / 'records' / (row['sample_id'] + '.json')
                if target.exists():
                    previous = read_json(target)
                    if previous['run_hash'] != run_hash or previous['candidate_hash'] != row['candidate_hash']:
                        raise ValueError('Resume record provenance mismatch')
                    if previous['status'] != 'error':
                        validate_aggregate(previous, row, cfg, run_hash, out)
                        counts[previous['status']] += 1
                        continue
                    if args.retry_errors:
                        history = out / 'errors_history' / (row['sample_id'] + f'.{time.time_ns()}.json')
                        atomic_json(history, previous)
                record = annotate_rollouts(row, get_backend, cfg, out, run_hash, args.retry_errors)
                atomic_json(target, record)
                counts[record['status']] += 1
                consecutive_errors = consecutive_errors + 1 if record['status'] == 'error' else 0
                print(json.dumps(dict(shard=args.shard_index, progress=f'{index}/{len(shard)}',
                                      sample_id=row['sample_id'], status=record['status'],
                                      error=record.get('error'), seconds=record['seconds'])), flush=True)
                if consecutive_errors >= 5:
                    raise RuntimeError('Five consecutive failures; stopping this shard. Inspect saved records before retrying.')
        finally:
            if backend is not None and hasattr(backend, 'close'):
                backend.close()
        elapsed = time.monotonic() - shard_start
        atomic_json(out / f'summary-{args.shard_index}.json', dict(counts=counts, assigned=len(shard),
                    run_hash=run_hash, elapsed_seconds=round(elapsed, 3)))
        if counts['error']:
            raise RuntimeError(f"{counts['error']} failed samples; saved for --retry-errors. Not a complete successful run.")

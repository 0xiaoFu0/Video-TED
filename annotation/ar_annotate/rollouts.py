"""Fixed-budget rollout accounting; every draw is retained, including disagreements."""
from .common import digest, label, read_json, validate_evidence, validate_verdict


def settings(cfg):
    n = cfg.get('rollouts', 1)
    minimum = cfg.get('min_accepted_rollouts')
    minimum = 1 if minimum is None else minimum
    seed = cfg.get('seed', 0)
    if type(n) is not int or n < 1 or type(minimum) is not int or not 1 <= minimum <= n:
        raise ValueError('rollouts >= 1 and 1 <= min_accepted_rollouts <= rollouts required')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    return n, minimum, seed


def generation_seed(sid, cfg, rollout_index, attempt, stage):
    return int(digest([sid, cfg.get('seed', 0), rollout_index, attempt, stage])[:8], 16)


def options_for(row, cfg):
    import random
    options = list(row['options'])
    # Keep option order fixed across draws to isolate generation randomness.
    random.Random(generation_seed(row['sample_id'], cfg, 0, 0, 'options')).shuffle(options)
    return options


def validate_draw(record, row, cfg, run_hash, index):
    if (record['sample_id'] != row['sample_id'] or record['candidate_hash'] != row['candidate_hash']
            or record['run_hash'] != run_hash or record['rollout_index'] != index
            or record['rollout_seed'] != generation_seed(row['sample_id'], cfg, index, 0, 'evidence')):
        raise ValueError('Rollout provenance mismatch: ' + row['sample_id'])
    if record['status'] == 'error':
        return
    evidence = validate_evidence(record['evidence'])
    options = record['verification_options']
    if options != options_for(row, cfg):
        raise ValueError('Rollout options mismatch')
    verdict = validate_verdict(record['verdict'], len(options),
                               require_hierarchy=cfg.get('prompt_profile') == 'tree_v1')
    passed = (verdict['choice'] >= 0 and label(options[verdict['choice']]) == row['answer']
              and evidence['sufficient_evidence'] and verdict['evidence_supported']
              and verdict.get('hierarchy_supported', True))
    if record['status'] != ('accepted' if passed else 'needs_review'):
        raise ValueError('Rollout status contradicts evidence/verdict')


def aggregate(row, draws, cfg, run_hash):
    n, minimum, _ = settings(cfg)
    if len(draws) != n:
        raise ValueError('Incomplete rollout set')
    for i, draw in enumerate(draws):
        validate_draw(draw, row, cfg, run_hash, i)
    # All usable draws must come from exactly the same sampled video input.
    inputs = {(r['video_sha256'], digest(r['sampling'])) for r in draws if r['status'] != 'error'}
    if len(inputs) > 1:
        raise ValueError('Video or sampled frames differ across rollouts; use a new run')
    accepted = [i for i, r in enumerate(draws) if r['status'] == 'accepted']
    errors = sum(r['status'] == 'error' for r in draws)
    status = 'accepted' if len(accepted) >= minimum else ('error' if errors else 'needs_review')
    # A reproducible representative, NOT a claim that this is the best-quality explanation.
    chosen = accepted[0] if accepted else next((i for i, r in enumerate(draws) if r['status'] != 'error'), 0)
    result = dict(draws[chosen])
    result.update(status=status, record_schema=3, seconds=round(sum(r['seconds'] for r in draws), 3),
                  rollout_summary=dict(total=n, accepted=len(accepted), needs_review=n-len(accepted)-errors,
                                       error=errors, minimum_accepted=minimum, selected_rollout=chosen,
                                       policy='fixed_budget_threshold_first_accepted', accepted_indices=accepted),
                  rollout_hashes=[digest(r) for r in draws])
    result.pop('error', None)
    if status == 'error':
        result['error'] = f'{errors}/{n} rollouts failed; retry technical errors explicitly'
    elif errors:
        result['rollout_warning'] = f'{errors}/{n} rollouts failed; retained a separately verified accepted draw'
    return result


def validate_aggregate(record, row, cfg, run_hash, run_dir):
    draws = [read_json(run_dir / 'rollouts' / row['sample_id'] / f'{i:04d}.json')
             for i in range(settings(cfg)[0])]
    expected = aggregate(row, draws, cfg, run_hash)
    if digest(expected) != digest(record):
        raise ValueError('Aggregate or rollout checkpoint changed: ' + row['sample_id'])
    return draws

"""Recover previously omitted exact video matches without regenerating old samples."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from .common import ROOT, atomic_json, digest, file_hash, read_json, rows, write_rows
from .prepare import prepare
from .report import shard_completion
from .rollouts import validate_aggregate


def _candidates(path):
    manifest = read_json(path.parent / 'manifest.json')
    if file_hash(path) != manifest['candidates_sha256']:
        raise ValueError('Existing candidate file differs from preparation manifest')
    values = {}
    for _, row in rows(path):
        sid = row['sample_id']
        if sid in values or digest({k: v for k, v in row.items() if k != 'candidate_hash'}) != row['candidate_hash']:
            raise ValueError('Duplicate or invalid existing candidate: ' + sid)
        values[sid] = row
    if not values:
        raise ValueError('Existing candidate list is empty')
    return values, manifest


def _source_manifest(manifest):
    while 'existing_manifest' in manifest:
        previous = manifest['existing_manifest']
        if (not isinstance(previous, dict) or not previous.get('candidates_sha256')
                or manifest.get('existing_candidates_sha256') != previous['candidates_sha256']):
            raise ValueError('Nested candidate provenance mismatch')
        manifest = previous
    if 'source_sha256' not in manifest or 'eval_exclusions_sha256' not in manifest:
        raise ValueError('Missing original preparation source/exclusion provenance')
    return manifest


def _source_candidate_hash(values):
    """The combined pool preserves the original prepared SSv2/K710 rows."""
    checksum = hashlib.sha256()
    for row in values:
        if row['dataset'] in ('ssv2', 'k710'):
            checksum.update((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
    return checksum.hexdigest()


def prepare_incremental(config_path, existing_candidates, existing_run, output, scope='all-missing'):
    if scope not in ('all-missing', 'evaluation-overlaps'):
        raise ValueError('Incremental scope must be all-missing or evaluation-overlaps')
    overlaps_only = scope == 'evaluation-overlaps'
    source, previous_dir, output = map(Path, (existing_candidates, existing_run, output))
    if output.exists():
        raise FileExistsError('Choose a NEW incremental candidate directory')
    if previous_dir.resolve() in output.resolve().parents:
        raise ValueError('Incremental output must be outside the existing annotation run')
    cfg = read_json(config_path)
    old_rows, old_manifest = _candidates(source)
    original = _source_manifest(old_manifest)
    if ('existing_manifest' in old_manifest
            and _source_candidate_hash(old_rows.values()) != original['candidates_sha256']):
        raise ValueError('Original SSv2/K710 candidate provenance mismatch')
    deny = Path(cfg['eval_exclusions'])
    if not deny.is_absolute():
        deny = ROOT / deny
    if file_hash(cfg['flash_jsonl']) != original['source_sha256']:
        raise ValueError('Original QA source changed; cannot identify the previously excluded pool')
    if file_hash(deny) != original['eval_exclusions_sha256']:
        raise ValueError('Original evaluation overlap list changed')
    run = read_json(previous_dir / 'run.json')
    if digest(run['spec']) != run['run_hash'] or file_hash(source) != run['spec']['candidate_file_sha256']:
        raise ValueError('Existing run/candidate provenance mismatch')
    selected = run['spec']['selected_ids']
    if len(selected) != len(set(selected)) or set(selected) != set(old_rows):
        raise ValueError('Existing run must cover the FULL candidate pool, not a pilot')
    missing_shards = shard_completion(run, previous_dir)
    if missing_shards:
        raise ValueError(f'Existing run has missing global shards: {missing_shards}')

    seen_ids, seen_paths, seen_content = set(), set(), set()
    old_statuses = Counter()
    for sid in selected:
        row = old_rows[sid]
        record = read_json(previous_dir / 'records' / (sid + '.json'))
        if (record['sample_id'] != sid or record['candidate_hash'] != row['candidate_hash']
                or record['run_hash'] != run['run_hash']):
            raise ValueError('Existing record provenance mismatch: ' + sid)
        if record['status'] not in ('accepted', 'needs_review', 'error'):
            raise ValueError('Unknown existing record status: ' + sid)
        draws = validate_aggregate(record, row, run['spec']['config'], run['run_hash'], previous_dir)
        seen_ids.add(sid)
        seen_paths.add(str(Path(row['video']).resolve()))
        old_statuses[record['status']] += 1
        # Reuse recorded content hashes; do not reread every old video. Include
        # partial/error draws, since the user explicitly chose not to retry them.
        hashes = {saved['video_sha256'] for saved in [record] + draws if saved.get('video_sha256')}
        if not hashes:
            # Early stat/read failures may predate recording a video hash.
            # Read just those old media so copied aliases are still skipped.
            try:
                hashes.add(file_hash(row['video']))
            except OSError as error:
                raise ValueError('Existing record lacks a content hash and its media cannot be read: ' + sid) from error
        seen_content.update(hashes)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.incremental-prepare-', dir=output.parent) as temp:
        retained = Path(temp) / 'retained_source'
        prepare(config_path, retained, retain_evaluation_overlaps=True)
        rebuilt_manifest = read_json(retained / 'manifest.json')
        if (rebuilt_manifest['source_sha256'] != original['source_sha256']
                or rebuilt_manifest['eval_exclusions_sha256'] != original['eval_exclusions_sha256']):
            raise ValueError('Preparation source changed during incremental rebuild')
        fresh, audit, skipped, fresh_content = [], [], Counter(), set()
        for _, row in rows(retained / 'candidates.jsonl'):
            sid, path = row['sample_id'], str(Path(row['video']).resolve())
            content_hash, reason = None, None
            if overlaps_only and not row.get('evaluation_overlaps'):
                reason = 'not_a_previously_excluded_evaluation_overlap'
            elif sid in seen_ids:
                reason = 'already_generated_sample_id'
            elif path in seen_paths:
                reason = 'already_generated_resolved_video'
            else:
                content_hash = file_hash(row['video'])
                if content_hash in seen_content:
                    reason = 'already_generated_video_content'
                elif content_hash in fresh_content:
                    reason = 'duplicate_incremental_video_content'
            if reason:
                skipped[reason] += 1
            else:
                fresh.append(row)
                fresh_content.add(content_hash)
            audit.append(dict(sample_id=sid, dataset=row['dataset'], video=row['video'],
                              evaluation_overlaps=row.get('evaluation_overlaps', []),
                              decision=reason or ('new_overlap_to_generate' if overlaps_only
                                                 else 'new_missing_to_generate'),
                              video_sha256=content_hash))
        fresh.sort(key=lambda r: r['sample_id'])
        # Exclusive mkdir keeps prior results and candidates immutable. Empty
        # increments are valid reports; the launch wrapper will not load a model.
        output.mkdir()
        retained.rename(output / 'retained_source')
        write_rows(output / 'candidates.jsonl', fresh)
        write_rows(output / 'selection.audit.jsonl', audit)
        manifest = dict(purpose='previously_excluded_overlap_increment_only' if overlaps_only
                                else 'all_previously_omitted_exact_matches_increment_only',
                        scope=scope,
                        label_policy=rebuilt_manifest['label_policy'],
                        candidates_sha256=file_hash(output / 'candidates.jsonl'),
                        config=cfg, source_sha256=original['source_sha256'],
                        eval_exclusions_sha256=original['eval_exclusions_sha256'],
                        existing_candidates=str(source.resolve()), existing_candidates_sha256=file_hash(source),
                        existing_run=str(previous_dir.resolve()), existing_run_hash=run['run_hash'],
                        existing_planned=len(selected), existing_statuses=dict(old_statuses),
                        existing_record_policy='skip_accepted_needs_review_and_error',
                        planned=len(fresh), accepted=dict(Counter(r['dataset'] for r in fresh)),
                        skipped=dict(skipped),
                        overlap_policy='retain_only_new_configured_evaluation_video_overlaps' if overlaps_only
                                       else 'retain_configured_evaluation_video_overlaps',
                        deduplication=['sample_id', 'resolved_video_path', 'video_sha256'],
                        rebuilt_candidates=dict(rebuilt_manifest['accepted']),
                        rebuilt_rejected=dict(rebuilt_manifest['rejected']),
                        note=('Only new SSv2/K710 overlap videos are scheduled. ' if overlaps_only else
                              'All newly matched SSv2/K710 videos omitted from the old candidate pool are scheduled, '
                              'including evaluation overlaps and recovered labels. ') +
                             'Existing HMDB51, annotations and old errors remain untouched. '
                             'No new inference or human verification during preparation.')
        atomic_json(output / 'manifest.json', manifest)
    print({k: manifest[k] for k in ('scope', 'existing_planned', 'existing_statuses', 'planned', 'accepted', 'skipped')},
          flush=True)
    return manifest


def prepare_overlap_incremental(config_path, existing_candidates, existing_run, output):
    """Backward-compatible entry point for evaluation-overlap-only increments."""
    return prepare_incremental(config_path, existing_candidates, existing_run, output,
                               scope='evaluation-overlaps')

"""Add the existing HMDB51 training videos to a provenance-preserving annotation pool."""
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from .common import ROOT, atomic_json, digest, file_hash, label, read_json, rows, write_rows


def build_membership(eval_root, out):
    root = Path(eval_root).resolve()
    manifest = read_json(root / 'benchmarks.json')
    membership, sources = defaultdict(set), []
    for name, spec in manifest['datasets'].items():
        if spec['dataset'] != 'hmdb51':
            continue
        path = root / spec['annotations']
        values = read_json(path)
        for row in values:
            membership[safe_relative(row['path'])].add(name)
        sources.append(dict(benchmark=name, path=str(path), rows=len(values), sha256=file_hash(path)))
    if not sources:
        raise ValueError('No HMDB51 evaluation annotations')
    atomic_json(out, dict(sources=sources, membership={k: sorted(v) for k, v in sorted(membership.items())},
                         policy='Membership flags, not training eligibility. Select the evaluation protocol before training.'))


def safe_relative(value):
    if not isinstance(value, str) or '\\' in value:
        raise ValueError('Invalid HMDB51 relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or path.suffix.lower() != '.avi':
        raise ValueError('Expected HMDB51 class/filename.avi')
    return str(path)


def hmdb_candidates(cfg):
    source = Path(cfg['hmdb51_source'])
    root = Path(cfg['hmdb51_root']).absolute()
    if not root.is_dir():
        raise FileNotFoundError(root)
    asset = Path(cfg['hmdb51_eval_membership'])
    if not asset.is_absolute():
        asset = ROOT / asset
    catalog = read_json(asset)
    if not catalog.get('sources') or not catalog.get('membership'):
        raise ValueError('Missing HMDB51 evaluation membership catalog')
    membership = catalog['membership']
    source_hash = file_hash(source)
    values = list(rows(source))
    if not values:
        raise ValueError('Empty HMDB51 source')
    # The class vocabulary comes from the existing training source, never its old CoT.
    classes = sorted({label(r['answer']) for _, r in values})
    expected = cfg.get('hmdb51_expected_classes')
    if expected is not None and len(classes) != expected:
        raise ValueError(f'Expected {expected} HMDB51 classes, found {len(classes)}')
    if len(classes) < 2:
        raise ValueError('Need at least two classes for HMDB51 verification')
    targets = defaultdict(set)
    for _, r in values:
        targets[safe_relative(r['metadata']['video_relpath'])].add(label(r['answer']))
    accepted, rejected, seen, source_ids = [], [], set(), set()
    for line, r in values:
        rel = safe_relative(r['metadata']['video_relpath'])
        source_id = r['metadata']['sample_id']
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise ValueError('Missing/duplicate source HMDB51 sample ID')
        source_ids.add(source_id)
        path = root / rel
        target = label(r['answer'])
        row = dict(dataset='hmdb51', source_line=line, original_video=r['videos'][0],
                   answer=target, raw_label=r['answer'], source_sample_id=source_id,
                   source_annotation=str(source), source_annotation_sha256=source_hash,
                   video_id=rel, video_relpath=rel,
                   evaluation_overlaps=membership.get(rel, []),
                   split_evidence='existing_hmdb51_training_source_not_resplit',
                   source_split='existing_train')
        reason = None
        if len(r['videos']) != 1:
            raise ValueError('Expected one HMDB51 source video')
        if len(targets[rel]) != 1:
            reason = 'conflicting_source_labels'
        elif label(PurePosixPath(rel).parent.name) != target:
            reason = 'directory_label_mismatch'
        elif not path.is_file() or path.stat().st_size == 0:
            reason = 'hmdb51_media_missing_or_empty'
        elif str(path.resolve()) in seen:
            reason = 'duplicate_resolved_video'
        if reason:
            rejected.append(dict(row, reason=reason))
            continue
        seen.add(str(path.resolve()))
        stat = path.stat()
        row.update(sample_id='hmdb51_' + digest(rel)[:20], video=str(path),
                   options=[c.replace('_', ' ') for c in classes],
                   video_size=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
                   match_method='existing_training_relative_path_under_hmdb51_root')
        row['candidate_hash'] = digest(row)
        accepted.append(row)
    report = dict(source=str(source), source_sha256=source_hash, source_rows=len(values), classes=classes,
                  eval_membership_sha256=file_hash(asset), evaluation_sources=catalog['sources'],
                  accepted=len(accepted), rejected=dict(Counter(r['reason'] for r in rejected)),
                  overlap_counts=dict(Counter(b for r in accepted for b in r['evaluation_overlaps'])))
    return accepted, rejected, report


def combine(config_path, existing_path, output):
    cfg, existing_path, output = read_json(config_path), Path(existing_path), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new combined candidate directory')
    previous = read_json(existing_path.parent / 'manifest.json')
    if file_hash(existing_path) != previous['candidates_sha256']:
        raise ValueError('Existing candidate file differs from preparation manifest')
    # Preserve the already applied SSv2/K710 evaluation exclusions.
    deny_path = Path(cfg['eval_exclusions'])
    if not deny_path.is_absolute():
        deny_path = ROOT / deny_path
    if previous['eval_exclusions_sha256'] != file_hash(deny_path):
        raise ValueError('SSv2/K710 exclusions changed; prepare these sources again')
    existing = [r for _, r in rows(existing_path)]
    if not existing or {r['dataset'] for r in existing} != {'ssv2', 'k710'}:
        raise ValueError('Expected existing SSv2 and K710 candidates only')
    seen_ids, seen_media = set(), set()
    for r in existing:
        if digest({k: v for k, v in r.items() if k != 'candidate_hash'}) != r['candidate_hash']:
            raise ValueError('Existing candidate hash mismatch')
        path = Path(r['video'])
        if (not path.is_file() or path.stat().st_size != r['video_size']
                or path.stat().st_mtime_ns != r['video_mtime_ns']):
            raise ValueError('Existing media changed/missing; rebuild candidates: ' + str(path))
        identity = str(path.resolve())
        if r['sample_id'] in seen_ids or identity in seen_media:
            raise ValueError('Duplicate existing candidate identity/media')
        seen_ids.add(r['sample_id']); seen_media.add(identity)
    hmdb, rejected, report = hmdb_candidates(cfg)
    if not hmdb:
        raise ValueError('No HMDB51 media matched; no combined pool written')
    for row in hmdb:
        if row['sample_id'] in seen_ids or str(Path(row['video']).resolve()) in seen_media:
            raise ValueError('Cross-source duplicate identity/media')
    combined = sorted(existing + hmdb, key=lambda r: r['sample_id'])
    output.mkdir(parents=True)
    write_rows(output / 'candidates.jsonl', combined)
    write_rows(output / 'rejected.hmdb51.jsonl', rejected)
    manifest = dict(candidates_sha256=file_hash(output / 'candidates.jsonl'), config=cfg,
                    accepted=dict(Counter(r['dataset'] for r in combined)), hmdb51=report,
                    existing_candidates_sha256=file_hash(existing_path),
                    existing_manifest=previous,
                    eval_exclusions_sha256=file_hash(deny_path),
                    purpose='combined_annotation_pool',
                    note='HMDB51 retains existing train source; evaluation membership flags must be checked before downstream splitting.')
    atomic_json(output / 'manifest.json', manifest)
    print(dict(accepted=manifest['accepted'], hmdb51=report))

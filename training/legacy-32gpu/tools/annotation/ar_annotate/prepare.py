"""Strict filename matching; no ID-only or time-window substitutions."""
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import ROOT, atomic_json, digest, file_hash, label, media_key, read_json, rows, write_rows

MARKERS = {'ssv2': '/classification_ssv2-ssv2-train_40k/',
           'k710': '/classification_k710-k710-train_40k/'}


def extract_qa(row):
    messages = row['messages']
    if [m['role'] for m in messages] != ['user', 'assistant']:
        raise ValueError('expected exactly one user/assistant pair')
    question, answer = (m['content'] for m in messages)
    option_text = re.split(r'Options\s*:', question, flags=re.I)[-1]
    options = re.findall(r'\(([A-Z])\)\s*(.*?)(?=\([A-Z]\)|$)', option_text, flags=re.S)
    if len(options) < 2 or len({k for k, _ in options}) != len(options):
        raise ValueError('missing or duplicate choice letters')
    choices = {key: value.strip() for key, value in options}
    match = re.fullmatch(r'\s*(?:Answer:\s*)?\(?([A-Z])\)\s*(.*?)\s*', answer, re.S | re.I)
    if not match or match[1].upper() not in choices:
        raise ValueError('unrecognized assistant answer')
    raw = choices[match[1].upper()]
    target = label(raw)
    if match[2] and label(match[2]) != target:
        raise ValueError('answer text differs from selected option')
    canonical = [label(v) for v in choices.values()]
    if len(canonical) != len(set(canonical)):
        raise ValueError('choice labels collide after normalization')
    return raw, target, list(choices.values())


def scan(root, wanted):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    hits = defaultdict(list)
    skipped = []
    scanned = 0
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        for name in list(dirs):
            p = Path(directory) / name
            if p.is_symlink():
                dirs.remove(name)
                skipped.append(str(p))
        for name in files:
            scanned += 1
            if name in wanted:
                p = Path(directory) / name
                if p.is_file() and p.stat().st_size > 0:
                    hits[name].append(str(p.absolute()))
    return hits, {'root': str(root), 'entries': scanned, 'skipped_directory_links': skipped}


def prepare(config_path, output, retain_evaluation_overlaps=False):
    cfg = read_json(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(output))
    deny_path = Path(cfg['eval_exclusions'])
    if not deny_path.is_absolute():
        deny_path = ROOT / deny_path
    deny = read_json(deny_path)
    denied = {k: set(v) for k, v in deny['video_ids'].items()}
    if not all(denied.get(k) for k in MARKERS):
        raise ValueError('Need nonempty SSv2 and Kinetics evaluation exclusion lists')
    pending, rejects, counts = [], [], Counter()
    for line, row in rows(cfg['flash_jsonl']):
        old = row.get('video')
        if not isinstance(old, str):
            raise ValueError(f'line {line}: video must be a string')
        dataset = next((k for k, marker in MARKERS.items() if marker in old), None)
        if dataset is None:
            continue
        counts[dataset + '_source'] += 1
        try:
            raw, target, options = extract_qa(row)
            key = media_key(dataset, old)
            overlaps_evaluation = key in denied[dataset]
            if overlaps_evaluation and not retain_evaluation_overlaps:
                raise ValueError('evaluation_video_overlap')
            candidate = dict(dataset=dataset, original_video=old, answer=target,
                             raw_label=raw, options=options, source_line=line, video_id=key)
            if overlaps_evaluation:
                candidate['evaluation_overlaps'] = [dataset + '-configured-evaluation-video']
            pending.append(candidate)
        except (ValueError, KeyError, TypeError) as e:
            rejects.append(dict(dataset=dataset, source_line=line, original_video=old, reason=str(e)))
    # Conflicting labels for a source video are withheld in their entirety.
    targets = defaultdict(set)
    for r in pending:
        targets[(r['dataset'], Path(r['original_video']).name)].add(r['answer'])
    indices, scans = {}, {}
    for dataset in MARKERS:
        root = cfg[dataset + '_root']
        # K710 may be sourced from K700, but accept only files in its train folder.
        if dataset == 'k710':
            root = str(Path(root) / 'train')
        print('Indexing ' + root, flush=True)
        indices[dataset], scans[dataset] = scan(root, {Path(r['original_video']).name for r in pending if r['dataset'] == dataset})
    accepted, seen = [], set()
    for row in pending:
        dataset = row['dataset']
        old = row['original_video']
        hits = indices[dataset].get(Path(old).name, [])
        reason = None
        if len(targets[(dataset, Path(old).name)]) != 1:
            reason = 'conflicting_source_labels'
        elif len(hits) != 1:
            reason = 'not_found_exact_filename' if not hits else 'ambiguous_exact_filename'
        elif dataset == 'k710':
            try:
                if label(Path(hits[0]).parent.name) != row['answer']:
                    reason = 'directory_label_mismatch'
            except ValueError:
                reason = 'invalid_directory_label'
        if reason:
            rejects.append(dict(**row, reason=reason, candidates=hits))
            continue
        path = Path(hits[0])
        identity = str(path.resolve())
        if identity in seen:
            rejects.append(dict(**row, reason='duplicate_resolved_video'))
            continue
        seen.add(identity)
        st = path.stat()
        row.update(sample_id=dataset + '_' + digest(old)[:20], video=str(path),
                   video_relpath=str(path.relative_to(Path(cfg[dataset + '_root']).absolute())),
                   video_size=st.st_size, video_mtime_ns=st.st_mtime_ns,
                   match_method='unique_exact_filename', split_evidence='flash_train_source' if dataset == 'ssv2' else 'flash_train_source_and_local_train_directory')
        row['candidate_hash'] = digest(row)
        accepted.append(row)
    if not accepted:
        raise ValueError('No eligible exact matches; no training data written')
    output.mkdir(parents=True)
    accepted.sort(key=lambda r: r['sample_id'])
    write_rows(output / 'candidates.jsonl', accepted)
    write_rows(output / 'rejected.jsonl', rejects)
    report = dict(source_sha256=file_hash(cfg['flash_jsonl']), config=cfg,
                  eval_exclusions_sha256=file_hash(deny_path), scans=scans,
                  label_policy='lowercase_whitespace_underscores_preserve_action_punctuation_v2',
                  overlap_policy='retain_configured_evaluation_video_overlaps' if retain_evaluation_overlaps
                                 else 'exclude_configured_evaluation_video_overlaps',
                  source_counts=dict(counts), accepted=dict(Counter(r['dataset'] for r in accepted)),
                  rejected=dict(Counter(r['reason'] for r in rejects)),
                  candidates_sha256=file_hash(output / 'candidates.jsonl'),
                  note='Existence and filename alignment only; no human verification or novel-class exclusion.')
    atomic_json(output / 'manifest.json', report)
    print({k: report[k] for k in ('source_counts', 'accepted', 'rejected')})

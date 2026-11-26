import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with open(path, encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError('expected object')
                    yield n, value
                except (ValueError, TypeError) as e:
                    raise ValueError(f'{path}:{n}: {e}') from e


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_rows(path, values):
    with open(path, 'x', encoding='utf-8') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + '\n')


def label(value):
    if not isinstance(value, str):
        raise ValueError('label must be text')
    value = re.sub(r'[\s_]+', '_', value.strip().lower())
    if not re.fullmatch(r'[a-z]+(?:_[a-z]+)*', value):
        raise ValueError(f'label incompatible with current AR reward: {value!r}')
    return value


def media_key(dataset, path):
    stem = Path(path).stem
    if dataset == 'ssv2':
        if not stem.isdigit():
            raise ValueError('SSv2 filename is not a numeric video ID')
        return str(int(stem))
    match = re.fullmatch(r'([A-Za-z0-9_-]{11})(?:_\d+_\d+)?', stem)
    if not match:
        raise ValueError(f'Unrecognized Kinetics video filename: {stem}')
    return match[1]  # Exclude the whole source video, including alternate time windows.


def text(value, max_chars=600):
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise ValueError('empty/oversized text')
    if any(c in value for c in '<>\n\r\t'):
        raise ValueError('markup or multiline text in evidence')
    return value.strip()


def parse_json_response(raw):
    raw = raw.strip()
    if raw.startswith('```json\n') and raw.endswith('```'):
        raw = raw[8:-3].strip()
    value = json.loads(raw)  # No regex salvage of truncated or multiple JSON objects.
    if not isinstance(value, dict):
        raise ValueError('model response is not an object')
    return value


def validate_evidence(value):
    missing = sorted({'activity', 'branches', 'uncertainties', 'sufficient_evidence'} - value.keys())
    if missing:
        raise ValueError('Missing required evidence fields: ' + ', '.join(missing))
    text(value['activity'], 120)
    if type(value['sufficient_evidence']) is not bool:
        raise ValueError('sufficient_evidence must be boolean')
    groups = value['branches']
    if not isinstance(groups, list) or not 1 <= len(groups) <= 4:
        raise ValueError('need 1..4 evidence branches')
    count = 0
    for group in groups:
        text(group['name'], 100)
        obs = group['observations']
        if not isinstance(obs, list) or not 1 <= len(obs) <= 3:
            raise ValueError('need 1..3 observations per branch')
        for observation in obs:
            text(observation, 350)
        count += len(obs)
    if count > 8:
        raise ValueError(f'At most 8 observations TOTAL across all branches; received {count}')
    if not isinstance(value['uncertainties'], list) or len(value['uncertainties']) > 3:
        raise ValueError('invalid uncertainties')
    for item in value['uncertainties']:
        text(item, 250)
    return value


def validate_verdict(value, n_options, require_hierarchy=False):
    choice = value['choice']
    if type(choice) is not int or not -1 <= choice < n_options:
        raise ValueError('choice must be an option index or -1')
    if type(value['evidence_supported']) is not bool:
        raise ValueError('evidence_supported must be boolean')
    if require_hierarchy and type(value.get('hierarchy_supported')) is not bool:
        raise ValueError('hierarchy_supported must be boolean for tree_v1')
    text(value['reason'], 600)
    return value


def hierarchy(evidence):
    validate_evidence(evidence)
    lines = [evidence['activity'].strip()]
    for group in evidence['branches']:
        lines.append('  ' + group['name'].strip())
        lines.extend('    ' + item.strip() for item in group['observations'])
    if evidence['uncertainties']:
        lines.append('  Uncertainty')
        lines.extend('    ' + item.strip() for item in evidence['uncertainties'])
    return '\n'.join(lines)

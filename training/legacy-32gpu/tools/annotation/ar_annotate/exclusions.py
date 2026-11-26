"""Build video-ID exclusions from this project's VideoSTAR-Eval benchmark manifest."""
import re
from pathlib import Path
from .common import atomic_json, file_hash, media_key, read_json


def build(eval_root, out):
    root = Path(eval_root)
    manifest = read_json(root / 'benchmarks.json')
    ids, sources = {'ssv2': set(), 'k710': set()}, []
    for name, spec in manifest['datasets'].items():
        if spec['dataset'] not in ('ssv2', 'k400', 'k600'):
            continue
        group = 'ssv2' if spec['dataset'] == 'ssv2' else 'k710'
        path = root / spec['annotations']
        values = read_json(path)
        for row in values:
            # Some evaluation entries have .mp4.webm double extensions.
            stem = re.sub(r'(?:\.(?:mp4|webm|avi|mkv))+$', '', Path(row['path']).name, flags=re.I)
            ids[group].add(media_key(group, stem + '.mp4'))
        sources.append(dict(benchmark=name, path=str(path), rows=len(values), sha256=file_hash(path)))
    if not all(ids.values()):
        raise ValueError('Missing evaluation annotations for one of the target datasets')
    atomic_json(out, dict(sources=sources, video_ids={k: sorted(v) for k, v in ids.items()},
                         policy='Video IDs only; NOT a novel-class exclusion list. Refresh after evaluation changes.'))
    print('Evaluation video IDs:', {k: len(v) for k, v in ids.items()})

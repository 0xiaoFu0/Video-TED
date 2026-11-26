"""Validate/rebase the think hierarchy exports without resplitting or relabeling."""
import argparse
import json
from pathlib import Path, PurePosixPath
from .reward import parse_answer, target_label
from .response_format import validate_reference_format, validate_student_format


def read_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty dataset')
    return rows


def validate_rows(rows, check_media=False):
    seen=set();missing=[]
    for row in rows:
        target_label(row['answer'])
        sid=row['metadata']['sample_id']
        if sid in seen:
            raise ValueError('Duplicate sample ID: '+sid)
        seen.add(sid)
        if len(row['videos']) != 1:
            raise ValueError('AR V1 expects one video per sample')
        prompt=row.get('prompt',row.get('messages',[{}])[0].get('content'))
        if not isinstance(prompt,str) or prompt.count('<video>')!=1:
            raise ValueError('Expected exactly one video marker')
        validate_student_format(prompt)
        if row['videos'][0] in prompt or row['metadata']['video_relpath'] in prompt:
            raise ValueError('Media filename must not be included in student text')
        if 'Reference class:' in prompt or 'Class-level prior' in prompt:
            raise ValueError('Privileged context leaked into student prompt')
        if 'messages' in row:
            if [m['role'] for m in row['messages']] != ['user','assistant']:
                raise ValueError('Expected one user/assistant SFT pair')
            response=row['messages'][1]['content']
        else:
            response=row['reference_response']
            if not row['teacher_context'].strip():
                raise ValueError('Missing teacher context')
            if row['reference_status'] not in {'unreviewed','reviewed','verified','rejected'}:
                raise ValueError('Unknown reference status')
        # Explicitly rejected RL references may be empty: these rows train on
        # video/label rewards and a label-only teacher hint, without CoT loss.
        inactive = ('messages' not in row and row['reference_status'] == 'rejected')
        if not isinstance(response, str):
            raise ValueError('Reference must be text: '+sid)
        if response or not inactive:
            if parse_answer(response)!=target_label(row['answer']):
                raise ValueError('Reference/target mismatch: '+sid)
            validate_reference_format(response)
        if check_media and not Path(row['videos'][0]).is_file():
            missing.append(row['videos'][0])
    if missing:
        raise FileNotFoundError(f'{len(missing)} video files missing; examples: {missing[:3]}')
    return dict(rows=len(rows),classes=len({r['answer'] for r in rows}),media_checked=check_media)


def rebase(rows, video_root):
    for row in rows:
        relative=PurePosixPath(row['metadata']['video_relpath'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
            raise ValueError('Unsafe media path')
        row['videos']=[str(PurePosixPath(video_root)/relative)]
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--out',type=Path,default=Path('data'))
    p.add_argument('--video-root')
    p.add_argument('--check',type=Path)
    p.add_argument('--check-media',action='store_true')
    a=p.parse_args()
    if a.check:
        print(json.dumps(validate_rows(read_rows(a.check),a.check_media)));return
    if not a.source_dir:
        p.error('--source-dir or --check is required')
    arrays={}
    for name in ('sft','rlsd'):
        rows=read_rows(a.source_dir/f'train.{name}.think.jsonl')
        if a.video_root:
            rebase(rows,a.video_root)
        validate_rows(rows,a.check_media)
        arrays[name]=rows
    for s,r in zip(arrays['sft'],arrays['rlsd']):
        if (s['metadata']!=r['metadata'] or s['videos']!=r['videos'] or s['answer']!=r['answer']
                or s['messages'][0]['content']!=r['prompt']
                or s['messages'][1]['content']!=r['reference_response']):
            raise ValueError('SFT/RLSD pair mismatch')
    if len(arrays['sft'])!=len(arrays['rlsd']):
        raise ValueError('SFT/RLSD count mismatch')
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Choose an empty output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    for name,rows in arrays.items():
        (a.out/f'train.{name}.think.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(dict(out=str(a.out),**validate_rows(arrays['rlsd'],False))))

if __name__=='__main__': main()

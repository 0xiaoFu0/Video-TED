import json
from pathlib import Path
import pytest
from verl.ar.prepare_data import read_rows, validate_rows, rebase

ROOT=Path(__file__).resolve().parents[2]


def test_complete_dataset_and_pair_isolation():
    sft=read_rows(ROOT/'data/train.sft.think.jsonl')
    rl=read_rows(ROOT/'data/train.rlsd.think.jsonl')
    assert validate_rows(sft)['rows']==validate_rows(rl)['rows']==4676
    assert validate_rows(rl)['classes']==51
    for s,r in zip(sft,rl):
        assert s['messages'][0]['content']==r['prompt']
        assert s['messages'][1]['content']==r['reference_response']
        assert s['metadata']==r['metadata']
        assert 'teacher_context' not in s
        assert 'offline_output' not in r
    row=dict(rl[0]);row['prompt']+='\nReference class: brush_hair'
    with pytest.raises(ValueError):validate_rows([row])


def test_rebase_and_missing_media():
    row=read_rows(ROOT/'data/train.rlsd.think.jsonl')[0]
    rebase([row],'/nonexistent/ar-test')
    with pytest.raises(FileNotFoundError):validate_rows([row],True)
    row['metadata']['video_relpath']='../secret.avi'
    with pytest.raises(ValueError):rebase([row],'/tmp')


def test_explicit_rejected_reference_may_be_empty_for_all_qa_rl():
    row=read_rows(ROOT/'data/train.rlsd.think.jsonl')[0]
    row['reference_response']=''
    row['reference_status']='rejected'
    row['teacher_context']='Reference class: '+row['answer']
    assert validate_rows([row])['rows']==1
    row['reference_status']='unreviewed'
    with pytest.raises(ValueError):validate_rows([row])
    row['reference_status']='rejected'
    row['reference_response']=None
    with pytest.raises(ValueError,match='must be text'):validate_rows([row])

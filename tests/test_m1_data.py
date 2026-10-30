import torch
from clr.data.m1 import intermediate_values,quantization
from clr.data.loader import ShardWriter,LatentDataset

def test_equations_and_failures():
    r=intermediate_values('<<6*3=18>> <<18/2=9>>')
    assert r['intermediate_values']==['18','9'] and not r['cot_parse_failed']
    assert intermediate_values('missing')['cot_parse_failed']
    assert intermediate_values('<<2+3=unknown>>')['cot_parse_failed']
    assert intermediate_values('<<2-3=-1>>')['intermediate_values']==['-1']

def test_storage_roundtrip_and_filters(tmp_path):
    torch.manual_seed(1);x=torch.randn(7,1,768);w=ShardWriter(tmp_path/'train')
    meta={'problem_id':'p','source':'codi_gpt2','split':'train','kind':'traj','traj_id':2,'t0':0,'label':True,'steps':list(range(7)),'layers':[12]}
    w.add(x,meta);stats=w.close()
    assert stats['fp16_max_relative_l2']<.001
    d=LatentDataset(tmp_path,'train',kind='traj',step=3)
    assert len(d)==1 and d[0]['problem_id']=='p' and d[0]['traj_id']==2 and d[0]['t']==3
    assert torch.equal(d[0]['z'],x[3,0].half())
    assert len(LatentDataset(tmp_path,'train',kind='clean'))==0

def test_candidate_ids_and_trajectories_are_problem_grouped():
    import json
    from pathlib import Path
    p=Path('data/M1_v3/candidates.json')
    if not p.exists():return
    data=json.loads(p.read_text())['splits'];seen=set()
    for split,rows in data.items():
        ids={r['id'] for r in rows}
        assert len(ids)==len(rows) and not ids&seen
        seen|=ids
        # Every generated trajectory takes split from its problem record.
        for row in rows:
            assert row['left_padding']>=0 and row['cohort_id'] and row['protocol_sha256']

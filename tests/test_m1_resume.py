import json
import torch
import pytest
from clr.data.loader import ShardWriter
from clr.data.resume import recover, ResumedWriter

def test_recovery_keeps_complete_prefix_and_appends(tmp_path):
    source=tmp_path/'source';old=tmp_path/'run';old.mkdir()
    w=ShardWriter(source)
    meta={'problem_id':'p','kind':'clean','traj_id':-1,'t0':0,'steps':list(range(7)),'layers':[12]}
    w.add(torch.ones(7,1,3),dict(meta))
    for k in range(8):
        w.add(torch.ones(4,1,3),{**meta,'kind':'traj','traj_id':k,'t0':3,'steps':[3,4,5,6]})
    w.flush()
    (old/'reader_train_early_partial.json').write_text(json.dumps([{'problem_id':'p','t':t} for t in range(7)]+[{'problem_id':'unflushed','t':0}]))
    dest=tmp_path/'destination'
    rows,early,hashes=recover(source,dest,[{'id':'p'}],old)
    assert len(early)==7 and len(hashes)==2
    resumed=ResumedWriter(dest,rows,.001);resumed.recovered_rows=len(rows)
    resumed.add(torch.ones(7,1,3),{**meta,'problem_id':'q'})
    result=resumed.close()
    assert result['shards']==2 and result['rows']==len(rows)+7
    assert not (source/'manifest.parquet').exists()
    with pytest.raises(ValueError,match='frozen prefix'):
        recover(source,tmp_path/'invalid',[{'id':'wrong'}],old)

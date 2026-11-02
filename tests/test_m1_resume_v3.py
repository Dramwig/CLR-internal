import json
import torch
import pytest
from clr.data.loader import ShardWriter
from clr.data.resume_v3 import recover_test

def test_heldout_recovery_checks_behavior_completeness(tmp_path):
    source=tmp_path/'source';run=tmp_path/'run';run.mkdir()
    context={'cohort_id':'c','left_padding':2,'pad_to':10,'protocol_sha256':'hash'}
    w=ShardWriter(source)
    w.add(torch.ones(7,1,4),{'problem_id':'p','kind':'clean','steps':list(range(7)),'layers':[12],**context})
    w.flush()
    early=[{'problem_id':'p','t':t} for t in range(7)]
    mc=[{'problem_id':'p','t':t,'policy':policy,'K':16,'answers':['1']*16,'correct':[True]*16,'value':1.0} for t in range(6) for policy in ['pi','pi_T']]
    for kind,rows in [('early',early),('mc',mc),('variance',[])]:
        (run/f'test_{kind}_partial.json').write_text(json.dumps(rows))
    result=recover_test(source,tmp_path/'dest',[{'id':'p',**context}],run,[])
    assert len(result[0])==7 and len(result[2])==12
    (run/'test_mc_partial.json').write_text(json.dumps(mc[:-1]))
    with pytest.raises(ValueError,match='behavior journal'):
        recover_test(source,tmp_path/'bad',[{'id':'p',**context}],run,[])

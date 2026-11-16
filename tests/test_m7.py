import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from clr.sources.kv_relay import verify_idea_a,audit_tensor
from clr.data.m7 import counterfactual,qa_rows,threshold,TEAMS,corrupt,commitment_route

verify_idea_a('/home/me/PaperField/LatDefer')
from latdef.data.schema import Example
from latdef.data.synthetic_v2 import solve_v2_text
from latdef.inject.cache import PrefixKV

def test_generator_pin_cannot_be_replaced_by_a_changed_freeze_manifest(tmp_path):
    import hashlib
    from clr.sources.kv_relay import CORE
    digests={}
    for name in CORE:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('changed source')
        digests[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/v2-design-freeze.json').write_text(json.dumps({'source_sha256':digests}))
    with pytest.raises(RuntimeError,match='SHA256'):verify_idea_a(tmp_path)

def examples():
    p=Path('/home/me/PaperField/LatDefer/data/synthetic/m1-v2-r2/val.jsonl')
    found={}
    for line in p.open():
        r=json.loads(line)
        if r['meta']['alpha']==1 and r['meta']['d_pub']==0:found.setdefault(r['question_type'],Example(**r))
        if len(found)==4:break
    return list(found.values())

def test_counterfactual_one_semantic_variable_and_labels():
    for e in examples():
        cf=counterfactual(e)
        assert cf.x_pub==e.x_pub and cf.meta['group_id']==e.meta['group_id']
        assert cf.meta['z_pub']==e.meta['z_pub'] and cf.meta['T']==e.meta['T']
        assert solve_v2_text(cf)==cf.label and cf.label!=e.label
        assert len(e.x_priv.splitlines())==len(cf.x_priv.splitlines())
        ticket=e.x_pub.splitlines()[0].split(': ')[1]
        for a,b in zip(e.x_priv.splitlines(),cf.x_priv.splitlines()):
            if ticket not in a:assert a==b

def test_A1_marginals_and_input_whitelist():
    e=examples()[0];cf=counterfactual(e)
    base={'id':e.id,'group_id':e.meta['group_id']}
    def row(x,y):return {'id':x.id,'e':x.meta['e_priv'],'paired_e':y.meta['e_priv'],'route':x.label,'paired_route':y.label}
    rows=qa_rows(base,row(e,cf),'COMMITMENT: choose Route A.','clean')+qa_rows(base,row(cf,e),'COMMITMENT: choose Route A.','counterfactual')
    pos=sorted(r['question_text'] for r in rows if r['task']=='A1' and r['label']==1)
    neg=sorted(r['question_text'] for r in rows if r['task']=='A1' and r['label']==0)
    assert pos==neg
    assert not any('x_priv' in r or 'x_pub' in r for r in rows)
    assert set(r['template_id'] for r in rows)=={0,1,2,3}
    assert commitment_route('COMMITMENT: choose Route C.')==2

def test_fp16_KV_order_padding_and_nonzero_positions():
    k=torch.arange(2*2*3*4).reshape(2,2,3,4).float();v=k+100
    p=PrefixKV(((k,v),),torch.tensor([[0,1,1],[1,1,1]],dtype=torch.bool),frozenset([0]))
    z,m=audit_tensor(p,0,[0]);assert z.shape==(2,16) and z.dtype==torch.float16
    assert torch.equal(z[0,:8],k[0,:,1].flatten().half())
    assert torch.equal(z[0,8:],v[0,:,1].flatten().half())
    assert m['token_range']==[1,3] and m['mask']==[False,True,True]

def test_mismatch_keeps_saved_pair_padding_and_positions():
    from clr.sources.kv_relay import permute_prefix
    k=torch.arange(2*2*3*4).reshape(2,2,3,4).float();v=k+100
    p=PrefixKV(((k,v),),torch.tensor([[0,1,1],[1,1,1]],dtype=torch.bool),frozenset([0]))
    original,meta=audit_tensor(p,0,[0]);donor=permute_prefix(p,[0])
    actual,actual_meta=audit_tensor(donor,0,[0])
    assert torch.equal(actual,original) and actual_meta==meta
    assert donor.length==3 and donor.mask.tolist()==[[False,True,True]]

def test_conformal_boundary_and_controls():
    assert threshold(list(range(29)),.01)==float('inf')
    assert threshold(list(range(29)),.05)==28
    assert threshold(list(range(100)),.05)==95
    z=torch.randn(100,32).half();out=corrupt(z,'noise',11)
    assert float((out.float()-z.float()).norm()/z.float().norm())==pytest.approx(.1,abs=.001)
    assert torch.equal(corrupt(z,'zero_layer',11),torch.zeros_like(z))

@pytest.mark.parametrize('kind',['B0','CLR','B3','linear'])
def test_reader_option_order_and_B0_no_length_channel(kind):
    from clr.audit.m7_reader import Serializer
    from clr.models.m7 import Reader
    class Tokenizer:
        cls_token_id=1;sep_token_id=2;pad_token_id=0
        def encode(self,text,add_special_tokens=False):return [10+sum(map(ord,x))%1000 for x in text.split()]
    s=Serializer(Tokenizer());row={'question_text':'Which private team is encoded?', 'options':list(TEAMS)}
    torch.set_num_threads(2);torch.manual_seed(11);model=Reader(kind).eval()
    r2={**row,'options':[TEAMS[i] for i in [2,0,3,1]]}
    z=torch.randn(1,3,2048);mask=torch.ones(1,3,dtype=torch.bool)
    with torch.no_grad():
        a=model(s.batch([row],'cpu'),z,mask,'A4')
        b=model(s.batch([r2],'cpu'),z,mask,'A4')
        if kind=='B0':
            b=model(s.batch([r2],'cpu'),torch.randn(1,5,2048),torch.ones(1,5,dtype=torch.bool),'A4')
    assert torch.equal(a[:,[2,0,3,1]],b)

def test_generated_train_val_cal_groups_are_disjoint_without_test_access():
    from clr.data.m7 import split_examples
    config={'idea_a':'/home/me/PaperField/LatDefer','data_seed':20261106,
            'counts':{'reader_val':4,'cal_A':2,'cal_B':2}}
    groups=[]
    for split in ['reader_train','reader_val','cal_A','cal_B']:
        rows=list(split_examples(config,split,2))
        ids={r.meta['group_id'] for r in rows}
        assert len(ids)==2 and all(not (ids&other) for other in groups)
        groups.append(ids)

def test_commitment_templates_are_shared_and_parseable():
    from clr.data.m7 import format_commitment,COMMITMENT_TEMPLATES
    assert len(COMMITMENT_TEMPLATES)>=3
    texts={format_commitment(str(i),i%4) for i in range(100)}
    assert len(texts)==12
    for i in range(100):assert commitment_route(format_commitment(str(i),i%4))==i%4

def test_consistency_targets_use_public_constraints_only():
    from clr.audit.m7_evaluation import expected_private,consistency_questions
    e=examples()[0]
    entry={'id':e.id,'group_id':e.meta['group_id'],'T':e.meta['T'],'z_pub':e.meta['z_pub'],
           'commitments':['COMMITMENT: choose Route '+chr(65+e.label)+'.']}
    assert expected_private(entry)==e.meta['e_priv']
    before=consistency_questions(entry,'x',11)
    assert before==consistency_questions({**entry,'e':[999,998]},'x',11)
    assert all(r['template_id']<4 for r in before)

def test_CLR_adapter_receives_gradients():
    from clr.audit.m7_reader import Serializer
    from clr.models.m7 import Reader
    class Tokenizer:
        cls_token_id=1;sep_token_id=2;pad_token_id=0
        def encode(self,text,add_special_tokens=False):return [10+sum(map(ord,x))%1000 for x in text.split()]
    torch.set_num_threads(2);torch.manual_seed(11)
    model=Reader('CLR');z=torch.randn(2,8,2048,requires_grad=True);mask=torch.ones(2,8,dtype=torch.bool)
    r={'question_text':'Does this message encode Amber?', 'options':['No','Yes']}
    logits=model(Serializer(Tokenizer()).batch([r,r],'cpu'),z,mask,'A1')
    logits[:,1].sum().backward()
    assert z.grad.abs().sum()>0
    assert model.kv.project[0].weight.grad.abs().sum()>0
    assert torch.isfinite(z.grad).all()

def test_false_fact_marginals_in_mismatch():
    from collections import Counter
    from clr.data.m7 import matched_false_values
    values=[0]*4+[1]*3+[2]*2+[3]
    false=matched_false_values(values)
    assert Counter(values)==Counter(false)
    assert all(a!=b for a,b in zip(values,false))

def test_group_bootstrap_calibration_and_discrimination_known_limits():
    import pandas as pd
    from clr.audit.m7_statistics import nonlinear_qa_intervals
    weights=np.array([[2,0],[0,2],[1,1]],float)
    for informative,expected in [(True,1.),(False,.5)]:
        rows=[{'group_id':group,'seed':seed,'label':label,
               'probs':([1.-label,float(label)] if informative else [.5,.5])}
              for group in ['a','b'] for seed in [11,22,33] for label in [0,1]]
        frame=pd.DataFrame(rows);result=nonlinear_qa_intervals(frame,weights)
        assert result['auroc']=={'mean':expected,'lo':expected,'hi':expected}
        assert result['ece']=={'mean':0.,'lo':0.,'hi':0.}
        assert result==nonlinear_qa_intervals(frame.iloc[::-1],weights)
    mixed=[{'group_id':group,'seed':seed,'label':label,'probs':prob}
           for group,label,prob in [('a',1,[.2,.8]),('a',0,[.9,.1]),('b',1,[.8,.2]),('b',0,[.7,.3])]
           for seed in [11,22,33]]
    result=nonlinear_qa_intervals(pd.DataFrame(mixed),weights)
    assert result['auroc']['mean']==pytest.approx(.75)
    assert result['ece']['mean']==pytest.approx(.25)

import torch
import pytest
from clr.models.m10 import AttentionProbe,ControlProbe
from clr.data.m4.reader import require_split

def test_b6_attention_mask_permutation_and_size():
 torch.manual_seed(11);torch.set_num_threads(1)
 for dim in [768,2048]:
  m=AttentionProbe(dim).eval();z=torch.randn(2,7,dim);mask=torch.tensor([[1,1,1,0,0,0,0]]*2,dtype=torch.bool);q=torch.randn(2,4,776)
  a=m(q,z,mask,'Q3');perm=[2,0,3,1]
  assert torch.allclose(m(q[:,perm],z,mask,'Q3'),a[:,perm],atol=1e-6,rtol=0)
  z[:,3:]+=100
  assert torch.equal(m(q,z,mask,'Q3'),a)
  assert sum(p.numel() for p in m.parameters())>=sum(p.numel() for p in ControlProbe('B3','Q3',dim).parameters())

def test_cached_serialization_exact():
 from clr.models.clr_v0 import Serializer
 from clr.data.m10 import SerializedCache
 class Tok:
  cls_token_id=1;sep_token_id=2;pad_token_id=0
  def encode(self,s,**kw):return [3+ord(c)%23 for c in s]
 rows=[{'qtype':'Q3','template_id':0,'options':x} for x in [['1','200','3','4000'],['2','3','4','5']]]
 s=Serializer(Tok(),{'Q3':['choose']});cache=SerializedCache(s,rows)
 for rr in [rows,rows[:1],rows[::-1]]:
  a,b=s.batch(rr),cache.batch(rr,'cpu')
  assert all(torch.equal(a[k],b[k]) for k in a)

def test_test_isolation_and_conformal():
 from clr.calib.m23 import conformal_threshold
 for phase in ['train','validate','calibrate']:
  with pytest.raises(ValueError):require_split('test',phase)
 assert conformal_threshold(list(range(1000)),.05)==950

def test_audit_b6_mask_order_size_and_public_condition():
 import numpy as np
 from clr.models.m10 import OptionFeatures
 torch.manual_seed(22);m=AttentionProbe(512,hidden=896,audit=True).eval()
 z={'values':torch.randn(2,3,512),'mask':torch.tensor([[1,1,0],[1,1,0]],dtype=torch.bool),'layers':torch.tensor([[0,35,35]]*2),'segments':torch.tensor([[0,1,2]]*2),'kinds':torch.tensor([[0,1,0]]*2)}
 q=torch.randn(2,4,776);a=m(q,z,task='A4');perm=[3,1,0,2]
 assert torch.allclose(m(q[:,perm],z,task='A4'),a[:,perm],atol=1e-6,rtol=0)
 z['values'][:,2]+=100
 assert torch.equal(a,m(q,z,task='A4'))
 assert sum(p.numel() for p in m.parameters())>=4719876
 f=object.__new__(OptionFeatures);f.one=lambda word:np.zeros(768,dtype=np.float32)
 from clr.data.m7 import TEAMS
 rows=[{'task':'A1','question_text':f'Does the message contain the fact that the team is {TEAMS[1]}?','options':['No','Yes'],'label':x} for x in [0,1]]
 features=f.batch(rows,'cpu',audit=True)
 assert torch.equal(features[0],features[1])
 assert features[0,0,769]==1

def test_lora_base_is_frozen_and_update_starts_at_zero():
 from clr.models.m10_b5c import LoRALinear
 model=LoRALinear(torch.nn.Linear(8,4));x=torch.randn(3,8)
 assert torch.equal(model(x),model.base(x))
 model(x).square().sum().backward()
 assert model.b.grad.abs().sum()>0
 assert model.base.weight.grad is None

def test_smoke_problem_fraction_is_enforced(monkeypatch):
 import clr.data.m10 as data
 monkeypatch.setattr(data,'rows',lambda *args,**kwargs:[{'problem_id':str(i)} for i in range(200)])
 c={'source':'gpt2','mode':'smoke','sources':{'gpt2':{'qa':'unused'}}}
 with pytest.raises(ValueError,match='above 5%'):data.GSMDataset(c,'reader_val','validate',11,['Q3'])


def test_hiddenbench_visible_expectations_and_matched_panels():
 from clr.audit.m10_hiddenbench import make_panels,questions,expected_fact
 items=[{'id':i,'description':f'public {i}','hidden_information':[f'private fact {i}-{j}' for j in range(3)],'correct_answer':'choice'} for i in range(4)]
 panels=make_panels(items,11)
 for panel in panels:
  a=questions(panel,'clean',[0]);b=questions(panel,'mismatch',[0])
  assert [r['question_text'] for r in a]==[r['question_text'] for r in b]
  assert [r['options'] for r in a]==[r['options'] for r in b]
  assert [r['expected_label'] for r in a]==[r['expected_label'] for r in b]
  assert a[0]['label']==1 and b[0]['label']==0
  assert a[1]['label']==0 and b[1]['label']==1
  assert expected_fact(panel['commitment'])==panel['fact']
  assert a[-1]['options'][a[-1]['label']]==panel['fact']
  assert b[-1]['options'][b[-1]['label']]==panel['donor_fact']


def test_hiddenbench_independent_group_split_and_finite_calibration():
 import json
 from pathlib import Path
 from clr.audit.m10_hiddenbench import public_group,read_raw
 from clr.data.m7 import threshold
 spec=json.loads(Path('data/M10_v1/hiddenbench/protocol.json').read_text());sets=[set(gs) for gs in spec['splits'].values()]
 assert sum(map(len,sets))==len(set.union(*sets))==63
 assert len(spec['splits']['cal_B'])==20
 assert __import__('math').isfinite(threshold(list(range(20)),.05))
 assert __import__('math').isinf(threshold(list(range(20)),.01))


def test_hiddenbench_probe_option_permutation():
 import torch
 from clr.audit.m10_hiddenbench import GenericB3
 torch.manual_seed(1);model=GenericB3().eval();q=torch.randn(2,4,776);z=torch.randn(2,18432);order=torch.tensor([2,0,3,1])
 with torch.no_grad():a=model(q,z,'A4');b=model(q[:,order],z,'A4')
 torch.testing.assert_close(b,a[:,order])


def test_group_bootstrap_paired_detection_keeps_counterparts():
 import pandas as pd
 from scripts.m10_export_audit import detection_summary
 f=pd.DataFrame([{'group_id':str(g),'category':cat,'score':float(cat=='counterfactual')} for g in range(5) for cat in ['clean','counterfactual']])
 d=detection_summary([f],[{'0.05':.5}],'score','counterfactual',.05)
 assert d['mean']==d['lo']==d['hi']==1
 assert d['FPR']==0 and d['AUROC']==1


def test_visible_commitment_json_fields_allow_punctuation():
 from clr.audit.m10_hiddenbench import commitment,expected_fact,visible_conclusion
 fact='Fact; with "quotes" and a \\ slash.';answer='Choice; A'
 for i in range(3):
  text=commitment(fact,answer,i)
  assert expected_fact(text)==fact
  assert visible_conclusion(text)==answer


def test_released_gpu_reuses_only_identical_completed_protocol(tmp_path,monkeypatch):
 import json
 import scripts.m10_train_gsm as training
 c={'mode':'full','case':'example','seed':11,'output':str(tmp_path),'device':'cuda:3','learning_rate':.0003}
 run=tmp_path/'run';run.mkdir();(run/'metrics.json').write_text(json.dumps({'status':'completed'}))
 refs=tmp_path/'trained';refs.mkdir();(refs/'example_11_full.json').write_text(json.dumps({'run':str(run),'config':c}))
 monkeypatch.setattr(training,'train_once',lambda c:pytest.fail('Completed task must not create another GPU run'))
 training.main({**c,'device':'cuda:5'})
 with pytest.raises(ValueError,match='different protocol'):training.main({**c,'learning_rate':.001})

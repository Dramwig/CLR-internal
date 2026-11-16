import torch
from clr.models.m6 import MultiLite
from clr.models.clr_v0 import Serializer
from clr.data.m4.reader import require_split
import pytest

def test_multisource_shared_body_source_gradient_and_option_order():
 torch.set_num_threads(1);torch.manual_seed(11)
 class Tok:
  cls_token_id=1;sep_token_id=2;pad_token_id=0
  def encode(self,s,**kw):return [3+ord(x)%20 for x in s]
 serializer=Serializer(Tok(),{'Q3':['pick']},max_length=64)
 model=MultiLite({'gpt2':768,'llama':2048}).eval()
 row={'qtype':'Q3','template_id':0,'options':['11','2','300','4']};perm=[2,0,3,1]
 for source,dim in [('gpt2',768),('llama',2048)]:
  z=torch.randn(1,7,dim);mask=torch.tensor([[1,1,1,0,0,0,0]],dtype=torch.bool)
  a=model(serializer.batch([row]),z,mask,'Q3',source)
  b=model(serializer.batch([{**row,'options':[row['options'][i] for i in perm]}]),z,mask,'Q3',source)
  assert torch.allclose(a[:,perm],b,atol=1e-6,rtol=0)
  z[:,3:]+=100
  assert torch.equal(a,model(serializer.batch([row]),z,mask,'Q3',source))
  model.zero_grad();a.sum().backward()
  assert model.adapters[source].weight.grad.abs().sum()>0
  assert model.source_tokens[source].grad.abs().sum()>0
  assert model.body.layers[0].linear1.weight.grad.abs().sum()>0
  other='llama' if source=='gpt2' else 'gpt2'
  assert model.adapters[other].weight.grad is None

def test_split_and_conformal_guards():
 from clr.calib.m23 import conformal_threshold
 for phase in ['train','validate','calibrate']:
  with pytest.raises(ValueError):require_split('test',phase)
 assert conformal_threshold(list(range(1000)),.1)==900


def test_paper_auroc_ties_and_monotone_radius():
 import numpy as np
 import pandas as pd
 from clr.paper.statistics import auc_summary,monotone_envelope
 # Pair concordance: a tied pair contributes one half, for AUROC=7/8.
 f=pd.DataFrame({'problem_id':['a','b','c','d'],'seed':[11]*4,'t':[0]*4,
                 'label':[0,0,1,1],'score':[0.,1.,1.,2.]})
 assert auc_summary(f,resamples=100)['mean']==pytest.approx(.875)
 raw=pd.DataFrame({'epsilon':[.1,.01,1.],'mean':[.72,.82,.64]})
 env=monotone_envelope(raw)
 assert np.allclose(env.envelope,[.82,.82,.82])
 assert np.allclose(env['mean'],[.82,.72,.64])


def test_paper_input_hash_is_required(tmp_path):
 import hashlib,json
 from clr.paper.render import Context,one
 import pandas as pd
 csv=tmp_path/'evidence.csv';csv.write_text('metric,mean\naccuracy,0.9\n')
 lock=tmp_path/'freeze.json';lock.write_text(json.dumps({'sha256':{str(csv):hashlib.sha256(csv.read_bytes()).hexdigest()}}))
 c=Context({'inputs':{'evidence':str(csv)},'input_freeze':str(lock)})
 assert one(c.read('evidence'),metric='accuracy')['mean']==.9
 csv.write_text('metric,mean\naccuracy,1.0\n')
 with pytest.raises(ValueError,match='not hash-frozen'):c.read('evidence')
 with pytest.raises(ValueError,match='Ambiguous'):one(pd.DataFrame({'metric':['acc','acc']}),metric='acc')

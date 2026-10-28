"""Tiny random models are unit-test fixtures, never M0 benchmark substitutes."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from clr.sources.coconut import CoconutSource
from clr.sources.codi import CODISource
from clr.sources.common import import_file,max_latent_difference

class Tokenizer:
    eos_token_id=0
    def encode(self,text,return_tensors=None,**kwargs):
        ids=[1,2,3]
        return torch.tensor([ids]) if return_tensors else ids
    def decode(self,tokens,**kwargs):return ','.join(map(str,tokens))

@pytest.fixture
def base():
    torch.manual_seed(1)
    return GPT2LMHeadModel(GPT2Config(vocab_size=24,n_embd=16,n_layer=2,n_head=2,n_positions=64)).eval()

def test_coconut_hook_zero_and_restoration(base):
    mod=import_file('third_party/coconut/coconut.py','test_official_coconut')
    source=CoconutSource.__new__(CoconutSource)
    source.config={'num_latent':3,'layer':-1,'max_new_tokens':4}
    source.device=torch.device('cpu');source.tokenizer=Tokenizer();source.start=20;source.end=21;source.latent=22
    source.model=mod.Coconut(base,22,20,21,0);source.forward_calls=0
    ids=torch.tensor([[1,2,3,20,22,22,22,21]])
    with torch.inference_mode():official=source.model.generate(ids,torch.ones_like(ids),max_new_tokens=4)
    a=source.run('q');b=source.run('q',{'from_step':0,'sigma':0,'seed':11})
    assert a['answer']==Tokenizer().decode(official[0,8:].tolist())==b['answer']
    assert len(a['latent_steps'])==3 and max_latent_difference(a['latent_steps'],b['latent_steps'])==0
    c=source.run('q',{'from_step':1,'sigma':.3,'seed':11})
    assert torch.equal(a['latent_steps'][0],c['latent_steps'][0])
    assert not torch.equal(a['latent_steps'][1],c['latent_steps'][1])
    assert not base._forward_hooks
    source.config['layer']=1
    d=source.run('q');assert d['answer']==a['answer']

def test_codi_actual_feedback_and_lower_layer_observation(base):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.codi=base;self.prj=torch.nn.Identity();self.bot_id=20;self.eot_id=21;self.model_name='gpt2'
        def get_embd(self,*args):return self.codi.get_input_embeddings()
    source=CODISource.__new__(CODISource);source.model=Model();source.device=torch.device('cpu')
    source.tokenizer=Tokenizer();source.forward_calls=0
    source.config={'num_latent':3,'layer':-1,'max_new_tokens':4}
    a=source.run('q');b=source.run('q',{'from_step':0,'sigma':0,'seed':5})
    assert a['answer']==b['answer'] and max_latent_difference(a['latent_steps'],b['latent_steps'])==0
    c=source.run('q',{'from_step':1,'sigma':.3,'seed':5})
    assert torch.equal(a['latent_steps'][0],c['latent_steps'][0])
    assert not torch.equal(a['latent_steps'][1],c['latent_steps'][1])
    source.config['layer']=1
    assert source.run('q')['answer']==a['answer']

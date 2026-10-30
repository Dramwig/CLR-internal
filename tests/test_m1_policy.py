import torch
from clr.policy import prepare,rollout,early_exit
from clr.sources.codi_v2 import CODISourceV2
from transformers import GPT2Config,GPT2LMHeadModel

def tiny():
    torch.manual_seed(17);torch.set_num_threads(1)
    class Tokenizer:
        eos_token_id=0
        def encode(self,text):return [1,2]
        def decode(self,ids,**kw):return str(ids)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.codi=GPT2LMHeadModel(GPT2Config(vocab_size=24,n_embd=16,n_layer=2,n_head=2,n_positions=64)).eval()
            self.prj=torch.nn.Linear(16,16);self.bot_id=20;self.eot_id=21;self.pad_token_id=22;self.model_name='gpt2'
        def get_embd(self,*a):return self.codi.get_input_embeddings()
    s=CODISourceV2.__new__(CODISourceV2);s.model=Model();s.tokenizer=Tokenizer();s.device=torch.device('cpu');s.forward_calls=0
    s.config={'dtype':'float32','num_latent':6,'layer':-1,'max_new_tokens':4,'prefill_attention_mask':True,'carry_attention_mask':False,'position_ids':'implicit_absolute'}
    return s

def test_m01_equivalence_and_indexing():
    s=tiny();p={'id':'p','question':'q','pad_to':5};prefix=prepare(p,s)
    assert len(prefix.states)==7
    for seed in [11,22,33]:
        a=s.run(p,{'from_step':3,'sigma':1.5,'seed':seed})
        b=rollout(p,prefix,3,1.5,0,seed)
        assert a['answer']==b['answer']
        for j in range(3,6):torch.testing.assert_close(a['latent_steps'][j],b['feedback'][j-3],rtol=0,atol=0)
    assert early_exit(p,prefix,6)['answer']==s.run(p)['answer']
    for t in range(6):
        b=rollout(p,prefix,t,0,0,11)
        torch.testing.assert_close(b['states'][1],prefix.states[t+1][0,0],rtol=0,atol=0)

def test_context_and_noise_rng():
    import pytest
    s=tiny();p={'id':'p','question':'q','pad_to':5};prefix=prepare(p,s)
    with pytest.raises(ValueError):rollout({**p,'pad_to':4},prefix,3,1.5,0,11)
    a=rollout(p,prefix,3,1.5,0,[11,22]);b=rollout(p,prefix,3,1.5,0,[11,22])
    assert [x['answer'] for x in a]==[x['answer'] for x in b]
    assert torch.equal(a[0]['states'],b[0]['states'])
    assert torch.equal(a[0]['states'][0],prefix.states[3][0,0].cpu())

import pytest
from clr.data.dedup_v2 import normalize,audit
from clr.utils.stats_v2 import bootstrap_mean

def test_normalization_retains_numbers_and_signs():
    assert normalize('JOHN’S  1.5 apples!')==normalize("John's 1.5 apples")
    assert normalize('1.5')!=normalize('15')
    assert normalize('-5')!=normalize('5')

def test_thirteen_gram_boundary_and_exact():
    shared='one two three four five six seven eight nine ten eleven twelve thirteen'
    train=[{'question':shared+' end'},{'question':'Short question?'}]
    target=[{'id':'a','question':shared+' changed'},{'id':'b','question':'short question!'},{'id':'c','question':'two three four five six seven eight nine ten eleven twelve thirteen changed'}]
    r=audit(train,target)
    assert r['a']['ngram_training_row_count']==1 and not r['a']['exact_training_rows']
    assert r['b']['exact_training_rows']==[1]
    assert not r['c']['overlap']

def test_bootstrap_is_by_question_and_reproducible():
    a=bootstrap_mean([1,0,1,0]);b=bootstrap_mean([1,0,1,0])
    assert a==b and a['n_questions']==4 and a['bootstrap_resamples']==10000
    assert a['mean']==.5 and a['ci95'][0]<=.5<=a['ci95'][1]
    with pytest.raises(ValueError):bootstrap_mean([])

def test_rollout_statistics_and_gate():
    from clr.data.noise_stats_v2 import summarize,recommend
    stats=summarize([{'correct':[True,False,True,False],'answers':['1','2','1',None]}, {'correct':[False]*4,'answers':['2']*4}],4)
    assert stats['accuracy']['mean']==.25
    assert stats['nondegenerate']['mean']==.5
    assert stats['invalid_rate']['mean']==.125
    assert stats['distinct_answers']['mean']==1.5
    assert stats['V_histogram']['counts']==[1,0,1,0,0]
    s={'sigma':.1,'from_step':0,'temperature':0,'accuracy_drop':.05,'nondegenerate':.9}
    assert recommend([s])['selected'] is None
    assert recommend([{**s,'accuracy_drop':.049}])['selected'] is not None
    assert recommend([{**s,'accuracy_drop':.01,'nondegenerate':.29}])['selected'] is None

def test_frozen_v2_split_is_disjoint_and_clean():
    import json
    from pathlib import Path
    p=Path('data/splits/gsm8k_M01_v2.json')
    if not p.exists():pytest.skip('Real-data audit not yet completed')
    s=json.loads(p.read_text());seen=set()
    for ids in s.values():
        assert not seen.intersection(ids)
        assert len(ids)==len(set(ids));seen.update(ids)
    assert len(s['val'])>=300
    assert [r['id'] for r in json.loads(Path('data/M01_v2/val.json').read_text())]==s['val']

def test_codi_v2_context_and_noise():
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    from clr.sources.codi_v2 import CODISourceV2
    from clr.sources.common import max_latent_difference
    torch.manual_seed(7);torch.set_num_threads(1)
    class Tokenizer:
        eos_token_id=0
        def encode(self,text):return [1,2,3] if text=='long' else [1]
        def decode(self,ids,**kw):return str(ids)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.codi=GPT2LMHeadModel(GPT2Config(vocab_size=24,n_embd=16,n_layer=2,n_head=2,n_positions=64)).eval()
            self.prj=torch.nn.Identity();self.bot_id=20;self.eot_id=21;self.pad_token_id=22;self.model_name='gpt2'
        def get_embd(self,*args):return self.codi.get_input_embeddings()
    s=CODISourceV2.__new__(CODISourceV2);s.model=Model();s.tokenizer=Tokenizer();s.device=torch.device('cpu');s.forward_calls=0
    s.config={'dtype':'float32','num_latent':3,'layer':-1,'max_new_tokens':4,'prefill_attention_mask':True,'carry_attention_mask':False,'position_ids':'implicit_absolute'}
    row={'question':'short','pad_to':3};a=s.run(row);b=s.run(row,{'from_step':0,'sigma':0.,'seed':4})
    assert a['answer']==b['answer'] and max_latent_difference(a['latent_steps'],b['latent_steps'])==0
    assert a['meta']['pad_count']==2 and len(a['latent_steps'])==3
    z=s.run(row,{'from_step':1,'sigma':.3,'seed':4})
    assert torch.equal(z['latent_steps'][0],a['latent_steps'][0])
    assert not torch.equal(z['latent_steps'][1],a['latent_steps'][1])
    with pytest.raises(ValueError,match='Different canonical'):s.run_batch([row,{'question':'long','pad_to':4}])

"""Tiny random LLaMA fixture validates source mechanics, not benchmark accuracy."""
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from clr.sources.codi_v2 import CODISourceV2
from clr.sources.common import max_latent_difference

def test_llama_feedback_zero_noise_and_prefix():
 torch.manual_seed(11);torch.set_num_threads(1)
 class Model(torch.nn.Module):
  def __init__(self):
   super().__init__();self.codi=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)).eval()
   self.prj=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.GELU(),torch.nn.LayerNorm(16));self.pad_token_id=29;self.bot_id=30;self.eot_id=31;self.model_name='llama'
  def get_embd(self,*args):return self.codi.get_input_embeddings()
 class Tokenizer:
  eos_token_id=2
  def encode(self,text):return [1,3,4]
  def decode(self,ids,**kw):return str(ids)
 source=CODISourceV2.__new__(CODISourceV2);source.device=torch.device('cpu');source.model=Model().eval();source.tokenizer=Tokenizer();source.forward_calls=0
 source.config={'dtype':'float32','num_latent':6,'max_new_tokens':4,'layer':-1,'position_ids':'implicit_absolute','prefill_attention_mask':True,'carry_attention_mask':False}
 p={'question':'q','pad_to':5};a=source.run(p);b=source.run(p,{'from_step':0,'sigma':0.,'seed':33})
 assert len(a['latent_steps'])==6 and all(z.shape==(16,) for z in a['latent_steps'])
 assert a['answer']==b['answer'] and max_latent_difference(a['latent_steps'],b['latent_steps'])==0
 pert=source.run(p,{'from_step':3,'sigma':.1,'seed':33})
 assert max_latent_difference(a['latent_steps'][:3],pert['latent_steps'][:3])==0
 assert max_latent_difference(a['latent_steps'][3:],pert['latent_steps'][3:])>0

"""Frozen Qwen3 language reader with r=16 LoRA and learned sequence injection."""
from functools import lru_cache
import torch
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer

class LoRALinear(nn.Module):
 def __init__(self,base,rank=16):
  super().__init__();self.base=base.requires_grad_(False);self.a=nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32));self.b=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32));nn.init.kaiming_uniform_(self.a,a=5**.5);self.scale=2.
 def forward(self,x):
  base=self.base(x);return base+(x.float()@self.a.T@self.b.T).to(base.dtype)*self.scale

class B5c(nn.Module):
 def __init__(self,path,dim,audit=False,rank=16):
  super().__init__();self.audit=audit;self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
  self.body,info=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
  if any(info[k] for k in ['missing_keys','mismatched_keys','unexpected_keys']):raise ValueError('Incomplete Qwen checkpoint')
  self.body.requires_grad_(False);self.lora_names=[]
  if getattr(self.body.config,'attention_dropout',0)!=0:raise ValueError('B5c requires dropout-free frozen Qwen for deterministic LoRA/checkpointing')
  for name,module in list(self.body.named_modules()):
   if isinstance(module,nn.Linear) and name.split('.')[-1] in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
    parent,key=name.rsplit('.',1);setattr(self.body.get_submodule(parent),key,LoRALinear(module,rank));self.lora_names.append(name)
  if not self.lora_names:raise ValueError('No LoRA modules installed')
  hidden=self.body.config.hidden_size;self.projection=nn.Sequential(nn.Linear(dim,512),nn.GELU(),nn.Linear(512,hidden))
  if audit:self.layer=nn.Embedding(36,dim);self.segment=nn.Embedding(512,dim);self.kind=nn.Embedding(3,dim)
  self.register_buffer('mean',torch.zeros(36,dim) if audit else torch.zeros(dim));self.register_buffer('scale',torch.ones_like(self.mean))
  self.answer_ids={}
  for word in ['No','Yes','A','B','C','D']:
   ids=self.tokenizer.encode(word,add_special_tokens=False)
   if len(ids)!=1:raise ValueError('Answers must be single tokens')
   self.answer_ids[word]=ids[0]
 def train(self,mode=True):
  # HF applies activation checkpointing only in training mode. Base weights
  # remain requires_grad=False, and the pinned Qwen has zero dropout.
  super().train(mode);return self
 @lru_cache(maxsize=100000)
 def encode(self,task,question,options):
  choice=task in ['Q3','A4'];order=sorted(range(len(options)),key=lambda i:options[i]) if choice else []
  text=question
  if choice:text+='\n'+'\n'.join(f'{chr(65+j)}: {options[i]}' for j,i in enumerate(order))+'\nAnswer with A, B, C, or D.\nAnswer:'
  else:text+='\nAnswer with Yes or No.\nAnswer:'
  ids=self.tokenizer.encode(text,add_special_tokens=False)
  if len(ids)>192:raise ValueError('B5c prompt exceeds fixed 192 tokens; no truncation')
  return ids,tuple(order.index(i) for i in range(len(options))) if choice else ()
 def forward(self,rows,z,mask=None,templates=None):
  task=rows[0].get('task',rows[0].get('qtype'));device=next(self.projection.parameters()).device
  encoded=[]
  for r in rows:
   question=r['question_text'] if self.audit else templates[task][r['template_id']].format(v=r['options'][0]) if task=='Q4' else templates[task][r['template_id']]
   encoded.append(self.encode(task,question,tuple(r['options'])))
  if self.audit:
   source=z;mask=source['mask'];values=(source['values']-self.mean[source['layers']])/self.scale[source['layers']]
   values=values+self.layer(source['layers'])+self.segment(source['segments'])+self.kind(source['kinds'])
  else:values=(z-self.mean)/self.scale
  projected=self.projection(values.float()).to(torch.bfloat16);embedding=self.body.get_input_embeddings();sequences=[]
  for i,(ids,restore) in enumerate(encoded):sequences.append(torch.cat([projected[i,mask[i]],embedding(torch.tensor(ids,device=device))]))
  lengths=torch.tensor([len(x) for x in sequences],device=device);width=int(lengths.max());inputs=torch.stack([torch.cat([x,torch.zeros(width-len(x),x.shape[1],dtype=x.dtype,device=device)]) for x in sequences]);attention=torch.arange(width,device=device)[None]<lengths[:,None]
  hidden=self.body.model(inputs_embeds=inputs,attention_mask=attention.long(),use_cache=False).last_hidden_state
  result=self.body.lm_head(hidden[torch.arange(len(rows),device=device),lengths-1]).float()
  if task in ['Q3','A4']:
   logits=result[:,[self.answer_ids[x] for x in ['A','B','C','D']]];return logits.gather(1,torch.tensor([x[1] for x in encoded],device=device))
  return result[:,[self.answer_ids['No'],self.answer_ids['Yes']]]
 def trainable_state(self):
  names={k for k,p in self.named_parameters() if p.requires_grad}|{'mean','scale'}
  return {k:v.detach().cpu().contiguous() for k,v in self.state_dict().items() if k in names}
 def restore_trainable(self,state):
  if set(state)!=set(self.trainable_state()):raise ValueError('Unexpected B5c trainable checkpoint keys')
  self.load_state_dict(state,strict=False)

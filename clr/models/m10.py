"""M10 controls and per-type attention probes; imports frozen CLR read-only."""
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoTokenizer
from clr.baselines.probes import Probe
from clr.models.m6 import NativeLite
from clr.models.m5b_controls import current_only
from clr.data.questions.core import numeric_features
from clr.audit.m71_reader import typed_condition

class ControlProbe(Probe):
 def __init__(self,kind,qtype,dim,hidden=128):
  super().__init__('B0' if kind=='B0' else 'B3',qtype,hidden)
  self.encoder=nn.Linear((2*dim if kind=='B3-prefix' else dim)+72,hidden)

class OptionFeatures:
 """Frozen pretrained lexical embeddings; no source problem or QA template."""
 def __init__(self,path):
  self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
  with safe_open(str(__import__('pathlib').Path(path)/'model.safetensors'),framework='pt') as f:
   self.weight=f.get_tensor('model.embeddings.tok_embeddings.weight').float()
 @lru_cache(maxsize=200000)
 def one(self,value):
  ids=self.tokenizer.encode(value,add_special_tokens=False)
  return self.weight[ids].mean(0).numpy() if ids else np.zeros(self.weight.shape[1],np.float32)
 def batch(self,rows,device,audit=False):
  result=[]
  for r in rows:
   task=r.get('task',r.get('qtype'))
   if task in ['Q1','Q2']:options=['']
   elif task=='A1':
    from clr.data.m7 import TEAMS
    options=[TEAMS[typed_condition(r)[0]]]
   elif task in ['A2','A3']:options=['Route '+chr(65+typed_condition(r)[1])]
   else:options=r['options']
   nums=numeric_features(options)
   if audit:
    from clr.data.m7 import TEAMS
    nums=np.zeros((len(options),8),np.float32)
    for i,o in enumerate(options):
     if o in TEAMS:nums[i,TEAMS.index(o)]=1
     elif o.startswith('Route '):nums[i,4+ord(o[-1])-65]=1
   result.append(np.concatenate([np.stack([self.one(o) for o in options]),nums],1))
  return torch.from_numpy(np.stack(result)).to(device)

class AttentionProbe(nn.Module):
 """One cross-attention, one linear head, one independent instance per task."""
 def __init__(self,dim,query_dim=776,hidden=256,heads=8,audit=False):
  super().__init__();self.audit=audit
  self.key=nn.Linear(dim,hidden);self.query=nn.Linear(query_dim,hidden)
  self.learned_query=nn.Parameter(torch.zeros(1,1,hidden))
  self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True,dropout=0.)
  self.head=nn.Linear(hidden,1)
  self.position=nn.Embedding(512 if audit else 7,hidden)
  self.layer=nn.Embedding(36,hidden) if audit else None
  self.kind=nn.Embedding(3,hidden) if audit else None
  self.register_buffer('mean',torch.zeros(36,dim) if audit else torch.zeros(dim))
  self.register_buffer('scale',torch.ones_like(self.mean))
 def forward(self,query,z,mask=None,task='Q3'):
  if self.audit:
   source=z;mask=source['mask'];values=(source['values']-self.mean[source['layers']])/self.scale[source['layers']]
   memory=self.key(values)+self.position(source['segments'])+self.layer(source['layers'])+self.kind(source['kinds'])
  else:
   memory=self.key((z-self.mean)/self.scale)+self.position(torch.arange(z.shape[1],device=z.device))[None]
  q=self.query(query)
  if task in ['Q1','Q2']:q=q+self.learned_query
  h=self.attention(q,memory,memory,key_padding_mask=~mask,need_weights=False)[0]
  logits=self.head(h).squeeze(-1)
  if task in ['Q3','A4']:return logits
  v=logits[:,0];return torch.stack([torch.zeros_like(v),v],-1)

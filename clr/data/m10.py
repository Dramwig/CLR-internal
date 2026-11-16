"""M10 allowlisted tensors and training-only normalization."""
import numpy as np
import torch
from clr.data.m6 import Bank,rows
from clr.data.m4.reader import targets
from clr.baselines.data_m23 import template_features
from clr.data.questions.core import numeric_features

class GSMDataset:
 def __init__(self,c,split,phase,limit=None,qtypes=None):
  spec=c['sources'][c['source']];self.rows=rows(spec['qa'],split,phase,qtypes=qtypes)
  if limit:
   available=sorted({r['problem_id'] for r in self.rows})
   if c.get('mode')=='smoke' and limit>len(available)*.05:raise ValueError(f'M10 smoke requested {limit}/{len(available)} problems, above 5%')
   ids=set(available[:limit]);self.rows=[r for r in self.rows if r['problem_id'] in ids]
  else:ids=None
  self.bank=Bank(spec['root'],split,phase,ids)
  self.groups={q:[r for r in self.rows if r['qtype']==q] for q in (qtypes or ['Q1','Q2','Q3','Q4'])}
  no=np.load(spec['normalization']);self.mean=torch.from_numpy(no['mean']);self.scale=torch.from_numpy(no['scale'])
 def batch(self,rs,device):return self.bank.batch(rs,device)

class ProbeCache:
 def __init__(self,dataset,kind,templates,device):
  self.ds=dataset;self.device=device;bank=dataset.bank
  mean,scale=dataset.mean,dataset.scale
  zz=(bank.z-mean)/scale
  if kind=='B3-prefix':
   out=[]
   for start in range(0,len(zz),4096):
    z=zz[bank.pi[start:start+4096]];m=bank.mask[start:start+4096,:,None]
    out.append(torch.cat([(z*m).sum(1)/m.sum(1),zz[start:start+4096]],1))
   zz=torch.cat(out)
  self.z=zz
  self.groups={}
  for q,rs in dataset.groups.items():
   self.groups[q]=(torch.tensor([bank.index[r['problem_id'],r['state_ref']['traj_id'],r['state_ref']['t']] for r in rs]),
    torch.from_numpy(np.stack([template_features(templates[q][r['template_id']]) for r in rs])).to(device),
    torch.from_numpy(np.stack([numeric_features(r['options'] or ['0']) for r in rs])).to(device),targets(rs,device))
 def batch(self,q,ii):
  ix,query,options,y=self.groups[q];j=torch.as_tensor(ii,device=self.device)
  return self.z[ix[ii]].to(self.device),query[j],options[j],y[j]

class SerializedCache:
 """Precompute identical typed serialization once per distinct public question."""
 def __init__(self,serializer,rs):
  self.serializer=serializer;self.lookup={};unique=[]
  for r in rs:
   k=self.key(r)
   if k not in self.lookup:self.lookup[k]=len(unique);unique.append(r)
  # Length is derived from allowed typed questions only; no source text.
  self.items=[];width=0
  for start in range(0,len(unique),512):
   item=serializer.batch(unique[start:start+512],'cpu');self.items.append(item);width=max(width,item['input_ids'].shape[1])
  self.data={}
  for k in self.items[0]:
   chunks=[]
   for item in self.items:
    x=item[k]
    if k in ['input_ids','blocks','position_ids','numeric']:
     shape=list(x.shape);shape[1]=width;y=torch.full(shape,-1 if k=='blocks' else serializer.tokenizer.pad_token_id if k=='input_ids' else 0,dtype=x.dtype);y[:,:x.shape[1]]=x;chunks.append(y)
    else:chunks.append(x)
   self.data[k]=torch.cat(chunks)
  self.items=[]
 @staticmethod
 def key(r):return (r.get('qtype',r.get('task')),r.get('template_id'),tuple(r['options']),r.get('question_text',''))
 def batch(self,rs,device):
  ii=torch.tensor([self.lookup[self.key(r)] for r in rs]);d={k:v[ii] for k,v in self.data.items()};width=int((d['blocks']>=0).sum(1).max())
  return {k:(v[:,:width] if k in ['input_ids','blocks','position_ids','numeric'] else v).to(device) for k,v in d.items()}

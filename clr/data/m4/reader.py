"""Allowlisted QA and observed-prefix loader; explicit phase gates protect test."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import pyarrow.parquet as pq
from clr.baselines.data_m23 import Bank,template_features
from clr.data.questions.core import numeric_features,seed

ALLOWED={'train':{'reader_train','unseen_train'},'validate':{'reader_val','val'},'calibrate':{'cal'},'final':{'reader_val','test','unseen_eval'}}

def require_split(split,phase):
    if split not in ALLOWED.get(phase,set()):raise ValueError(f'{phase} may not access {split}; smoke/debug must use val proxy')

def locations(c,split):
    if split.startswith('unseen'):return Path(c['output'])/'qa'/f'{split}.parquet',Path(c['output'])/'unseen'
    return Path(c['qa_root'])/f'{split}.parquet',Path(c['root'])

def rows_for(c,split,phase,heldout=False,ids=None,qtypes=None):
    require_split(split,phase);path,_=locations(c,split)
    filters=[('is_heldout_template','=',heldout)]
    if ids is not None:filters.append(('problem_id','in',sorted(ids)))
    if qtypes:filters.append(('qtype','in',list(qtypes)))
    columns=['problem_id','state_ref','qtype','template_id','options','label','label_soft','bernoulli_outcomes','cal_part','replicate','is_heldout_template']
    return pq.read_table(path,columns=columns,filters=filters).to_pylist()

class PrefixBank:
    def __init__(self,c,split,phase,ids=None):
        require_split(split,phase);_,root=locations(c,split)
        bank=Bank(root,split,ids);self.z=bank.z;self.index=bank.index
        rows=pq.read_table(root/split/'manifest.parquet',columns=['problem_id','traj_id','t0','layer'],filters=[('layer','=',12)]).to_pylist()
        t0={(r['problem_id'],r['traj_id']):r['t0'] for r in rows if ids is None or r['problem_id'] in ids}
        indices=np.zeros((len(self.index),7),np.int64);valid=np.zeros((len(self.index),7),bool)
        for (pid,traj,t),i in self.index.items():
            start=t0[pid,traj] if traj>=0 else 7
            for j in range(t+1):indices[i,j]=self.index[pid,-1 if j<start else traj,j];valid[i,j]=True
        self.prefix_indices=torch.from_numpy(indices);self.prefix_mask=torch.from_numpy(valid)
    def batch(self,rows,device):
        ii=torch.tensor([self.index[r['problem_id'],r['state_ref']['traj_id'],r['state_ref']['t']] for r in rows])
        return self.z[self.prefix_indices[ii]].to(device),self.prefix_mask[ii].to(device)

def targets(rows,device):
    if rows[0]['qtype']=='Q3':return torch.nn.functional.one_hot(torch.tensor([r['label'] for r in rows],device=device),4).float()
    y=torch.tensor([r['label_soft'] for r in rows],device=device);return torch.stack([1-y,y],1)

def probe_inputs(rows,prefix,mask,templates,mean,scale):
    ii=mask.long().sum(1)-1;z=prefix[torch.arange(len(rows),device=prefix.device),ii]
    z=(z-mean)/scale
    q=torch.from_numpy(np.stack([template_features(templates[r['qtype']][r['template_id']]) for r in rows])).to(z.device)
    options=torch.from_numpy(np.stack([numeric_features(r['options'] or ['0']) for r in rows])).to(z.device)
    return z,q,options

def mixed_prefix_batch(rows,banks,device):
    """Mix observed prefixes without exposing split identity to the reader.

Split is internal storage dispatch only. Rows retain their original order and
labels; source-model cohort contexts have already been frozen in each dump.
"""
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['_split'],[]).append(i)
    if len(groups)==1:return banks[next(iter(groups))].batch(rows,device)
    prefix=torch.empty(len(rows),7,768,device=device);mask=torch.empty(len(rows),7,dtype=torch.bool,device=device)
    for split,indices in groups.items():
        z,m=banks[split].batch([rows[i] for i in indices],device);index=torch.tensor(indices,device=device)
        prefix.index_copy_(0,index,z);mask.index_copy_(0,index,m)
    return prefix,mask

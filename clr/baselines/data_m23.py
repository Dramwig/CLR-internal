"""Compact state bank and explicitly allowlisted feature construction."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import pyarrow.parquet as pq
from safetensors import safe_open
from clr.data.questions.core import numeric_features,seed

@lru_cache(maxsize=256)
def template_features(text):
    v=np.zeros(64,np.float32)
    for x in text.encode():v[x%64]+=1
    return v/max(np.linalg.norm(v),1.)

class Bank:
    def __init__(self,root,split,ids=None,layer=12):
        self.root=Path(root)/split
        rows=pq.read_table(self.root/'manifest.parquet',columns=['problem_id','traj_id','t','layer','shard','offset'],filters=[('layer','=',layer)]).to_pylist()
        if ids is not None:rows=[r for r in rows if r['problem_id'] in ids]
        self.index={(r['problem_id'],r['traj_id'],r['t']):i for i,r in enumerate(rows)}
        self.z=torch.empty((len(rows),768),dtype=torch.float32)
        by={}
        for i,r in enumerate(rows):by.setdefault(r['shard'],[]).append((i,r['offset']))
        for name,entries in by.items():
            with safe_open(str(self.root/name),framework='pt',device='cpu') as f:
                values=f.get_tensor('states').float()
                ii,offset=zip(*entries);self.z.index_copy_(0,torch.tensor(ii),values.index_select(0,torch.tensor(offset)))
        self.mean=None;self.scale=None
    def standardize(self,mean=None,scale=None):
        mean=self.z.mean(0) if mean is None else torch.as_tensor(mean)
        scale=self.z.std(0,correction=0).clamp_min(.01) if scale is None else torch.as_tensor(scale)
        self.z=(self.z-mean)/scale
        return mean.numpy(),scale.numpy()

class Questions:
    def __init__(self,path,qtype,bank,templates,heldout=False,cal_part=None,problem_ids=None,clean_only=False):
        filters=[('qtype','=',qtype),('is_heldout_template','=',heldout)]
        if cal_part:filters.append(('cal_part','=',cal_part))
        if problem_ids is not None:filters.append(('problem_id','in',sorted(problem_ids)))
        self.rows=pq.read_table(path,filters=filters).to_pylist();self.qtype=qtype
        if problem_ids is not None:self.rows=[r for r in self.rows if r['problem_id'] in problem_ids]
        if clean_only:self.rows=[r for r in self.rows if r['state_ref']['traj_id']==-1]
        self.state=np.array([bank.index[(r['problem_id'],r['state_ref']['traj_id'],r['state_ref']['t'])] for r in self.rows])
        self.q=np.stack([template_features(templates[qtype][r['template_id']]) for r in self.rows])
        k=4 if qtype=='Q3' else 1
        self.options=np.stack([numeric_features(r['options'] if r['options'] else ['0']*k) for r in self.rows])
        if qtype=='Q3':self.y=np.eye(4,dtype=np.float32)[[r['label'] for r in self.rows]]
        else:
            y=np.array([r['label_soft'] for r in self.rows],np.float32);self.y=np.stack([1-y,y],1)
    def shuffled(self,bank,random_seed):
        # Each recipient gets another problem's observed state at the same t.
        groups={}
        for key,index in bank.index.items():groups.setdefault(key[2],[]).append((key[0],index))
        rng=np.random.default_rng(random_seed);out=[]
        for r in self.rows:
            candidates=groups[r['state_ref']['t']]
            while True:
                pid,index=candidates[rng.integers(len(candidates))]
                if pid!=r['problem_id']:break
            out.append(index)
        return np.array(out)

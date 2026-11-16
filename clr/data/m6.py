"""Dimension-aware observed-prefix storage. Phase guards preserve test isolation."""
from pathlib import Path
import torch
import pyarrow.parquet as pq
from safetensors import safe_open
from clr.data.m4.reader import require_split

class Bank:
    def __init__(self,root,split,phase,ids=None):
        require_split(split,phase);self.root=Path(root)/split
        filters=[('problem_id','in',sorted(ids))] if ids is not None else None
        rows=pq.read_table(self.root/'manifest.parquet',columns=['problem_id','traj_id','t0','t','layer','shard','offset','dim'],filters=filters).to_pylist()
        layer=max(r['layer'] for r in rows); rows=[r for r in rows if r['layer']==layer and (ids is None or r['problem_id'] in ids)]
        dims={r['dim'] for r in rows}
        if len(dims)!=1:raise ValueError('A bank contains exactly one source dimension')
        self.dim=dims.pop();self.index={(r['problem_id'],r['traj_id'],r['t']):i for i,r in enumerate(rows)}
        if len(self.index)!=len(rows):raise ValueError('Duplicate observed-state keys')
        self.z=torch.empty(len(rows),self.dim);groups={}
        for i,r in enumerate(rows):groups.setdefault(r['shard'],[]).append((i,r['offset']))
        for shard,entries in groups.items():
            with safe_open(str(self.root/shard),framework='pt',device='cpu') as f:
                ii,off=zip(*entries);self.z[torch.tensor(ii)]=f.get_tensor('states')[list(off)].float()
        self.pi=torch.zeros(len(rows),7,dtype=torch.long);self.mask=torch.zeros(len(rows),7,dtype=torch.bool)
        for i,r in enumerate(rows):
            pid,traj,t=r['problem_id'],r['traj_id'],r['t'];t0=r['t0'] if traj>=0 else 7
            for j in range(t+1):self.pi[i,j]=self.index[pid,-1 if j<t0 else traj,j];self.mask[i,j]=True
    def batch(self,rows,device):
        ix=torch.tensor([self.index[r['problem_id'],r['state_ref']['traj_id'],r['state_ref']['t']] for r in rows])
        return self.z[self.pi[ix]].to(device),self.mask[ix].to(device)

def rows(qa_root,split,phase,ids=None,qtypes=None,clean=False):
    require_split(split,phase); filters=[('is_heldout_template','=',False)]
    if ids is not None:filters.append(('problem_id','in',sorted(ids)))
    if qtypes:filters.append(('qtype','in',qtypes))
    rs=pq.read_table(Path(qa_root)/f'{split}.parquet',filters=filters).to_pylist()
    if clean:rs=[r for r in rs if r['state_ref']['traj_id']==-1]
    return rs

def nested_panel(rs,n,seed=11):
    """Exactly N distinct QA instances, nested and balanced by qtype.

    Instances within a problem/trajectory may share outcomes; no iid claim.

    Repeated labels on a trajectory remain distinct as in frozen M2; the full
    parent problem stays in reader_train. Every competing model shares panels.
    """
    from clr.data.questions.core import seed as hashseed
    if n is None:return rs
    groups={q:[] for q in ['Q1','Q2','Q3','Q4']}
    for i,r in enumerate(rs):groups[r['qtype']].append(i)
    for q,ii in groups.items():ii.sort(key=lambda i:hashseed(seed,'M6-label-panel',rs[i]['problem_id'],rs[i]['state_ref'],rs[i]['template_id'],tuple(rs[i]['options']),rs[i]['replicate']))
    order=[]
    for j in range(max(map(len,groups.values()))):
        for q,ii in groups.items():
            if j<len(ii):order.append(ii[j])
        if len(order)>=n:break
    if len(order)<n:raise ValueError('Requested label panel larger than training data')
    return [rs[i] for i in order[:n]]

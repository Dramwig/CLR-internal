"""Allowlisted model inputs. Always read observed 'states', never 'feedback'."""
from pathlib import Path
from functools import lru_cache
import torch
import pyarrow.parquet as pq
from safetensors import safe_open

class StateStore:
    def __init__(self,root,split):
        self.root=Path(root)/split
        columns=['problem_id','kind','traj_id','t0','t','layer','shard','offset']
        rows=pq.read_table(self.root/'manifest.parquet',columns=columns).to_pylist()
        self.index={(r['problem_id'],r['traj_id'],r['t'],r['layer']):r for r in rows}
    @lru_cache(maxsize=8)
    def shard(self,name):
        with safe_open(str(self.root/name),framework='pt',device='cpu') as f:return f.get_tensor('states')
    def state(self,pid,traj,t,layer=12):
        r=self.index[(pid,traj,t,layer)]
        return self.shard(r['shard'])[r['offset']].float()
    def prefix(self,pid,ref,layer=12):
        traj=ref['traj_id'];t=ref['t']
        t0=self.index[(pid,traj,t,layer)]['t0'] if traj>=0 else t+1
        return torch.stack([self.state(pid,-1 if k<t0 else traj,k,layer) for k in range(t+1)])

def model_input(row,store,templates):
    # Never forward metadata such as source question, gold CoT, or outcome.
    text=templates[row['qtype']][row['template_id']]
    if row['qtype']=='Q4':text=text.format(v=row['options'][0])
    return {'states':store.prefix(row['problem_id'],row['state_ref']),
            'question':text,'options':row['options'],'qtype':row['qtype']}

"""Shared two-layer MLP with qtype embedding, same observed z/numeric features as B3."""
import torch
from torch import nn

class B3Multi(nn.Module):
    def __init__(self,hidden=128):
        super().__init__();self.type_embedding=nn.Embedding(4,16);self.net=nn.Sequential(nn.Linear(768+64+8+16,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,z,q,options,qtype):
        k=options.shape[1];typ=self.type_embedding.weight[int(qtype[1])-1].view(1,1,-1).expand(len(z),k,-1)
        x=torch.cat([z[:,None].expand(-1,k,-1),q[:,None].expand(-1,k,-1),options,typ],-1);logits=self.net(x).squeeze(-1)
        return logits if qtype=='Q3' else torch.stack([torch.zeros_like(logits[:,0]),logits[:,0]],-1)

class B3PerType(nn.Module):
    """Four independent frozen-M3 B3 architectures, with no parameter sharing."""
    def __init__(self):
        super().__init__()
        from clr.baselines.probes import Probe
        self.probes=nn.ModuleDict({q:Probe('B3',q) for q in ['Q1','Q2','Q3','Q4']})
    def forward(self,z,q,options,qtype):return self.probes[qtype](z,q,options)

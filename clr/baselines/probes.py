"""Text-free probes; shared option scorer is permutation equivariant."""
import torch
from torch import nn

class Probe(nn.Module):
    def __init__(self,name,qtype,hidden=128):
        super().__init__();self.name=name;self.qtype=qtype
        # Deterministic bag-of-byte features encode typed template, not source text.
        self.encoder=nn.Linear(768,hidden) if name in ['B0','B1','B3'] else nn.Identity()
        dim=hidden if name in ['B0','B1','B3'] else 768
        self.option=nn.Linear(8,dim,bias=False) if qtype=='Q3' else None
        self.qhead=nn.Linear(64,1)
        self.option_bias=nn.Linear(8,1)
        self.head=nn.Linear(dim,1)
    def forward(self,z,q,options):
        if self.name=='B0':z=torch.zeros_like(z)
        h=self.encoder(z)
        if self.name in ['B0','B1','B3']:h=torch.relu(h)
        if self.qtype=='Q3':
            return torch.einsum('bd,bkd->bk',h,self.option(options))/h.shape[-1]**.5+self.option_bias(options).squeeze(-1)
        out=self.head(h).squeeze(-1)
        if self.name!='B4':out=out+self.qhead(q).squeeze(-1)
        if self.qtype=='Q4':out=out+self.option_bias(options[:,0]).squeeze(-1)
        return torch.stack([torch.zeros_like(out),out],-1)

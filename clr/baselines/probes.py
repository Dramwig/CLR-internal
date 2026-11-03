"""Text-free probes; each choice uses the same scorer (permutation equivariant)."""
import torch
from torch import nn

class Probe(nn.Module):
    def __init__(self,name,qtype,hidden=128):
        super().__init__();self.name=name;self.qtype=qtype
        self.joint=name in ['B0','B1','B3']
        if self.joint:
            # Joint conditioning is essential for Q4 membership queries: an
            # additive state score plus value score cannot express membership.
            self.encoder=nn.Linear(768+64+8,hidden);self.head=nn.Linear(hidden,1)
        else:
            self.option=nn.Linear(8,768,bias=False) if qtype=='Q3' else None
            self.qhead=nn.Linear(64,1);self.option_bias=nn.Linear(8,1);self.head=nn.Linear(768,1)
    def forward(self,z,q,options):
        if self.name=='B0':z=torch.zeros_like(z)
        if self.joint:
            k=options.shape[1]
            inputs=torch.cat([z[:,None,:].expand(-1,k,-1),q[:,None,:].expand(-1,k,-1),options],-1)
            scores=self.head(torch.relu(self.encoder(inputs))).squeeze(-1)
            if self.qtype=='Q3':return scores
            out=scores[:,0]
        else:
            if self.qtype=='Q3':return torch.einsum('bd,bkd->bk',z,self.option(options))/768**.5+self.option_bias(options).squeeze(-1)
            out=self.head(z).squeeze(-1)
            if self.name!='B4':out=out+self.qhead(q).squeeze(-1)
            if self.qtype=='Q4':out=out+self.option_bias(options[:,0]).squeeze(-1)
        return torch.stack([torch.zeros_like(out),out],-1)

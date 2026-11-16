"""M4.1 Lite imported unchanged; a KV source adapter reduces its long memory to 6 slots.

Adapter attends every saved (layer, token) item. Only this adapter resamples;
stored observations retain all tokens from the selected message layer.
"""
import torch
from torch import nn
from clr.models.clr_v1 import Lite
from clr.models.clr_v0 import block_masks

class KVAdapter(nn.Module):
    def __init__(self,dim=2048,slots=6):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.LayerNorm(128))
        self.position=nn.Embedding(8192,128)
        self.queries=nn.Parameter(torch.randn(slots,128)*.02)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True,dropout=0)
        self.out=nn.Linear(128,768)
    def forward(self,z,mask):
        memory=self.project(z)+self.position(torch.arange(z.shape[1],device=z.device))[None]
        query=self.queries[None].expand(len(z),-1,-1)
        y=self.attention(query,memory,memory,key_padding_mask=~mask,need_weights=False)[0]
        return self.out(y)

class Reader(nn.Module):
    def __init__(self,kind='CLR',dim=2048):
        super().__init__();self.kind=kind
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        if kind in ['CLR','B0']:
            self.kv=KVAdapter(dim);self.body=Lite()
        else:
            self.embedding=nn.Embedding(50368,128)
            if kind=='B3':self.head=nn.Sequential(nn.Linear(dim+256,256),nn.GELU(),nn.Linear(256,1))
            elif kind=='linear':
                self.state=nn.Linear(dim,128,bias=False);self.question=nn.Linear(128,128,bias=False)
                self.option=nn.Linear(128,128,bias=False);self.prior=nn.Linear(256,1)
            else:raise ValueError(kind)
    def forward(self,text,z,mask,task):
        if self.kind=='B0':
            prefix=torch.zeros(len(z),6,768,device=z.device)
            return self.body(text,prefix,torch.ones(prefix.shape[:2],device=z.device,dtype=torch.bool),'Q3' if task=='A4' else 'Q4')
        state=(z-self.mean)/self.scale
        if self.kind=='CLR':
            prefix=self.kv(state,mask)
            return self.body(text,prefix,torch.ones(prefix.shape[:2],device=z.device,dtype=torch.bool),'Q3' if task=='A4' else 'Q4')
        pooled=(state*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        emb=self.embedding(text['input_ids']);qm=text['blocks']==0
        q=(emb*qm[:,:,None]).sum(1)/qm.sum(1)[:,None]
        opts=[]
        for b in range(1,text['markers'].shape[1]+1):
            om=text['blocks']==b
            opts.append((emb*om[:,:,None]).sum(1)/om.sum(1)[:,None].clamp_min(1))
        option=torch.stack(opts,1);q=q[:,None].expand_as(option)
        if self.kind=='B3':
            logits=self.head(torch.cat([pooled[:,None].expand(-1,len(opts),-1),q,option],-1)).squeeze(-1)
        else:
            condition=self.question(q)+self.option(option)
            logits=(self.state(pooled)[:,None]*condition).sum(-1)/128**.5+self.prior(torch.cat([q,option],-1)).squeeze(-1)
        return logits.gather(1,text['restore'])

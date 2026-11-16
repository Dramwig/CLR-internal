"""M6 native-dimension CLR-lite, frozen-body alignment and shared-source model."""
import torch
from torch import nn
from clr.models.clr_v1 import Lite
from clr.models.clr_v0 import block_masks, QTYPES
from clr.baselines.probes import Probe

class NativeLite(Lite):
    def __init__(self,dim=768,hidden=256,layers=4):
        super().__init__(hidden=hidden,layers=layers)
        if dim!=768:
            self.adapter[0]=nn.Linear(dim,hidden)
            self.heads=nn.ModuleDict({k:nn.Sequential(nn.Linear(hidden+dim+8,256),nn.GELU(),nn.Linear(256,1)) for k in ['score','yesno','choice']})
            self.state_mean=torch.zeros(dim);self.state_scale=torch.ones(dim)
    def with_source_token(self,text,prefix,mask,qtype,token):
        bs,n,_=prefix.shape;device=prefix.device;z=(prefix-self.state_mean)/self.state_scale
        latent=self.adapter(z)+self.steps(torch.arange(n,device=device))[None]
        latent=torch.cat([token.expand(bs,1,-1),latent],1)
        tok=self.tokens(text['input_ids'])+self.positions(text['position_ids'])+self.numeric(text['numeric']);tok=tok.clone();tok[:,0]+=self.qtype.weight[QTYPES.index(qtype)]
        hidden=torch.cat([latent,tok],1);latent_mask=torch.cat([torch.ones(bs,1,dtype=torch.bool,device=device),mask],1)
        blocks=torch.cat([torch.where(latent_mask,0,-1),text['blocks']],1)
        positions=torch.cat([torch.arange(n+1,device=device)[None].expand(bs,-1),text['position_ids']+n+1],1)
        allowed,_=block_masks(blocks,positions,512);blocked=(~allowed[:,0])[:,None].expand(-1,self.nheads,-1,-1).reshape(bs*self.nheads,hidden.shape[1],hidden.shape[1])
        for layer in self.layers:hidden=layer(hidden,src_mask=blocked)
        hidden=self.norm(hidden);idx=torch.arange(bs,device=device);selected=hidden[idx[:,None],text['markers']+n+1];numbers=text['numeric'][idx[:,None],text['markers']]
        if qtype in ['Q1','Q2']:selected=hidden[:,n+1,None];numbers=torch.zeros(bs,1,8,device=device)
        final=z[idx,mask.long().sum(1)-1,None].expand(-1,selected.shape[1],-1);joint=torch.cat([selected,final,numbers],-1)
        if qtype=='Q3':return self.heads['choice'](joint).squeeze(-1).gather(1,text['restore'])
        logit=self.heads['score' if qtype=='Q1' else 'yesno'](joint[:,0]).squeeze(-1)
        return torch.stack([torch.zeros_like(logit),logit],-1)

class NativeB3(nn.Module):
    def __init__(self,dim):
        super().__init__();self.models=nn.ModuleDict()
        for q in QTYPES:
            p=Probe('B3',q);p.encoder=nn.Linear(dim+72,128);self.models[q]=p
    def forward(self,z,q,o,qtype):return self.models[qtype](z,q,o)

class TransferLite(nn.Module):
    def __init__(self,body,target_dim):
        super().__init__();self.body=body.requires_grad_(False);donor_dim=len(body.state_mean)
        self.adapter=nn.Linear(target_dim,donor_dim)
        self.register_buffer('mean',torch.zeros(target_dim));self.register_buffer('scale',torch.ones(target_dim))
    def forward(self,text,prefix,mask,qtype):
        z=self.adapter((prefix-self.mean)/self.scale)
        return self.body(text,z*self.body.state_scale+self.body.state_mean,mask,qtype)
    def train(self,mode=True):
        super().train(mode);self.body.eval();return self

class MultiLite(nn.Module):
    def __init__(self,dims):
        super().__init__();self.body=NativeLite();self.adapters=nn.ModuleDict({s:nn.Linear(d,768) for s,d in dims.items()})
        self.source_tokens=nn.ParameterDict({s:nn.Parameter(torch.randn(1,1,256)*.02) for s in dims})
        for s,d in dims.items():self.register_buffer(s+'_mean',torch.zeros(d));self.register_buffer(s+'_scale',torch.ones(d))
    def forward(self,text,prefix,mask,qtype,source):
        z=(prefix-getattr(self,source+'_mean'))/getattr(self,source+'_scale');z=self.adapters[source](z)
        return self.body.with_source_token(text,z,mask,qtype,self.source_tokens[source])

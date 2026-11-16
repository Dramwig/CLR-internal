"""Single-component ablations of frozen M4.1 Lite; original implementation untouched."""
import torch
from torch import nn
from clr.models.clr_v1 import Lite
from clr.models.clr_v0 import QTYPES, block_masks

class AblationLite(Lite):
    def __init__(self, ablation, **kwargs):
        super().__init__(**kwargs)
        if ablation not in ['main','A1','A2','A3','A4','A5']: raise ValueError(ablation)
        self.ablation = ablation
        hidden = self.norm.normalized_shape[0]
        if ablation == 'A2':
            self.heads = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in ['score','yesno','choice']})
        if ablation == 'A3':
            del self.numeric
            self.heads = nn.ModuleDict({k:nn.Sequential(nn.Linear(hidden+768,256),nn.GELU(),nn.Linear(256,1)) for k in ['score','yesno','choice']})

    def attention(self, blocks, positions, n):
        allowed,_ = block_masks(blocks, positions, 512, isolated=self.ablation != 'A1')
        if self.ablation == 'A4':
            # Prevent indirect state->template->option paths in every layer.
            allowed[:,:,:n,n:] = False
            allowed[:,:,n:,:n] = False
        return allowed

    def forward(self, text, prefix, prefix_mask, qtype):
        bs,n,_ = prefix.shape; device=prefix.device
        z=(prefix-self.state_mean)/self.state_scale
        latent=self.adapter(z)+self.steps(torch.arange(n,device=device))[None]
        tok=self.tokens(text['input_ids'])+self.positions(text['position_ids'])
        if self.ablation != 'A3': tok=tok+self.numeric(text['numeric'])
        tok=tok.clone();tok[:,0]+=self.qtype.weight[QTYPES.index(qtype)]
        hidden=torch.cat([latent,tok],1)
        blocks=torch.cat([torch.where(prefix_mask,0,-1),text['blocks']],1)
        positions=torch.cat([torch.arange(n,device=device)[None].expand(bs,-1),text['position_ids']+n],1)
        allowed=self.attention(blocks,positions,n)
        blocked=(~allowed[:,0])[:,None].expand(-1,self.nheads,-1,-1).reshape(bs*self.nheads,hidden.shape[1],hidden.shape[1])
        for layer in self.layers: hidden=layer(hidden,src_mask=blocked)
        hidden=self.norm(hidden);idx=torch.arange(bs,device=device)
        selected=hidden[idx[:,None],text['markers']+n];numbers=text['numeric'][idx[:,None],text['markers']]
        if qtype in ['Q1','Q2']:
            selected=hidden[:,n,None];numbers=torch.zeros(bs,1,8,device=device)
        head='choice' if qtype=='Q3' else 'score' if qtype=='Q1' else 'yesno'
        pooled=(z*prefix_mask[:,:,None]).sum(1)/prefix_mask.sum(1,keepdim=True)
        if self.ablation=='A2':
            logit=(selected*self.adapter(pooled)[:,None]).sum(-1)+self.heads[head]
        else:
            state=pooled if self.ablation=='A4' else z[idx,prefix_mask.long().sum(1)-1]
            parts=[selected,state[:,None].expand(-1,selected.shape[1],-1)]
            if self.ablation!='A3': parts.append(numbers)
            joint=torch.cat(parts,-1)
            if qtype!='Q3':
                logit=self.heads[head](joint[:,0]).squeeze(-1)
                return torch.stack([torch.zeros_like(logit),logit],-1)
            logit=self.heads[head](joint).squeeze(-1)
        if qtype=='Q3': return logit.gather(1,text['restore'])
        logit=logit[:,0];return torch.stack([torch.zeros_like(logit),logit],-1)

def make_model(c):
    return AblationLite(c['ablation'],hidden=c.get('lite_hidden',256),layers=c.get('lite_layers',4))

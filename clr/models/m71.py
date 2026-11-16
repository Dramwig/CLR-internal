"""M4.1 Lite with a projected-record adapter and strong typed pooled probes."""
import torch
from torch import nn
from clr.models.clr_v1 import Lite


class FeatureAdapter(nn.Module):
    def __init__(self,slots=6):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.LayerNorm(128))
        self.layer=nn.Embedding(36,128);self.segment=nn.Embedding(512,128);self.kind=nn.Embedding(3,128)
        self.queries=nn.Parameter(torch.randn(slots,128)*.02)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True,dropout=0.)
        self.output=nn.Linear(128,768)

    def forward(self,source):
        memory=(self.project(source['values'])+self.layer(source['layers'])+
                self.segment(source['segments'])+self.kind(source['kinds']))
        query=self.queries[None].expand(len(memory),-1,-1)
        result=self.attention(query,memory,memory,key_padding_mask=~source['mask'],need_weights=False)[0]
        return self.output(result)


class Reader(nn.Module):
    def __init__(self,kind='CLR',layers=4):
        super().__init__();self.kind=kind;dim=layers*9*512
        if kind in ['CLR','B0','supervised']:
            self.adapter=FeatureAdapter();self.body=Lite()
            self.register_buffer('mean',torch.zeros(36,512));self.register_buffer('scale',torch.ones(36,512))
        else:
            self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
            if kind=='linear':
                self.heads=nn.ModuleDict({task:nn.Linear(dim,4) for task in ['A1','A2','A3','A4']})
            elif kind=='B3':
                self.heads=nn.ModuleDict({task:nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,4))
                                          for task in ['A1','A2','A3','A4']})
            else:raise ValueError('Unknown M7.1 reader '+kind)

    def forward(self,text,source,task):
        if self.kind in ['CLR','B0','supervised']:
            if self.kind=='B0':
                prefix=torch.zeros(len(text['team']),6,768,device=text['team'].device)
            else:
                normalized={**source,'values':(source['values']-self.mean[source['layers']])/self.scale[source['layers']]}
                prefix=self.adapter(normalized)
            mask=torch.ones(prefix.shape[:2],dtype=torch.bool,device=prefix.device)
            return self.body(text,prefix,mask,'Q3' if task=='A4' else 'Q4')
        pooled=(source-self.mean)/self.scale;head=self.heads[task]
        if self.kind in ['linear','B3']:
            # For the four fixed typed options this is an option-conditioned
            # MLP: h(z) dot one_hot(option), with a separate binary head for
            # each queried fact/route. It avoids burying an 8D option code in
            # an 18k-dimensional concatenation. All conditions are parsed from
            # question/option words, never generator labels.
            logits=head(pooled)
            if task=='A4':return logits.gather(1,text['option_teams'])
            index=text['team'] if task=='A1' else text['route']
            score=logits.gather(1,index[:,None]).squeeze(1)
        return torch.stack([torch.zeros_like(score),score],-1)

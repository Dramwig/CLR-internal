"""M4.1 candidates; frozen v0 is imported, never edited.

v1 keeps all 22 pretrained blocks and all cross-attention paths. It adds a
nonzero, learnable residual scale to new cross modules and a joint nonlinear
readout of option representation, final observed state, and numeric features.
This explicit state/value interaction avoids requiring the random cross stack
to learn membership before any useful gradient reaches a conditional scorer.
Lite uses four randomly initialized transformer blocks with ancestor attention.
Neither model can access problem text or unobserved future states.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from clr.models.clr_v0 import CLR,QTYPES,block_masks

class ScaledCross(nn.Module):
    def __init__(self,module,scale=.1):
        super().__init__();self.attention=module;self.logit_scale=nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(scale)))))
    def forward(self,*args,**kwargs):
        out,w=self.attention(*args,**kwargs)
        return out*self.logit_scale.sigmoid(),w

class CLRv1(CLR):
    def __init__(self,backbone,**kwargs):
        super().__init__(backbone,**kwargs);h=backbone.config.hidden_size
        self.cross=nn.ModuleDict({k:ScaledCross(v) for k,v in self.cross.items()})
        self.heads=nn.ModuleDict({k:nn.Sequential(nn.Linear(h+768+8,256),nn.GELU(),nn.Linear(256,1)) for k in ['score','yesno','choice']})
    def forward(self,text,prefix,prefix_mask,qtype):
        bs=prefix.shape[0];device=prefix.device;z=(prefix-self.state_mean)/self.state_scale
        memory=self.adapter(z)+self.steps(torch.arange(prefix.shape[1],device=device))[None]
        memory=torch.cat([self.source_token.expand(bs,-1,-1),memory],1)
        memory_mask=torch.cat([torch.ones(bs,1,dtype=torch.bool,device=device),prefix_mask],1)
        emb=self.backbone.embeddings.tok_embeddings(text['input_ids'])+self.numeric(text['numeric']);emb=emb.clone();emb[:,0]+=self.qtype.weight[QTYPES.index(qtype)]
        hidden=self.backbone.embeddings(inputs_embeds=emb);mask,local=block_masks(text['blocks'],text['position_ids'],self.backbone.config.local_attention)
        for i,layer in enumerate(self.backbone.layers):
            args={'attention_mask':mask,'sliding_window_mask':local,'position_ids':text['position_ids']}
            hidden=(checkpoint(layer,hidden,use_reentrant=False,**args) if self.training and self.checkpointing else layer(hidden,**args))[0]
            if str(i) in self.cross:hidden=hidden+self.cross[str(i)](self.cross_norm[str(i)](hidden),memory,memory,key_padding_mask=~memory_mask,need_weights=False)[0]
        hidden=self.backbone.final_norm(hidden);idx=torch.arange(bs,device=device)
        selected=hidden[idx[:,None],text['markers']];numeric=text['numeric'][idx[:,None],text['markers']]
        if qtype in ['Q1','Q2']:selected=hidden[:,0,None];numeric=torch.zeros(bs,1,8,device=device)
        final=z[idx,prefix_mask.long().sum(1)-1,None].expand(-1,selected.shape[1],-1)
        joint=torch.cat([selected,final,numeric],-1)
        if qtype=='Q3':return self.heads['choice'](joint).squeeze(-1).gather(1,text['restore'])
        logit=self.heads['score' if qtype=='Q1' else 'yesno'](joint[:,0]).squeeze(-1)
        return torch.stack([torch.zeros_like(logit),logit],-1)

class Lite(nn.Module):
    def __init__(self,vocab_size=50368,hidden=256,layers=4,heads=8):
        super().__init__();self.adapter=nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Linear(hidden,hidden));self.steps=nn.Embedding(7,hidden)
        self.tokens=nn.Embedding(vocab_size,hidden);self.positions=nn.Embedding(256,hidden);self.qtype=nn.Embedding(4,hidden);self.numeric=nn.Linear(8,hidden,bias=False)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(hidden,heads,hidden*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(layers)])
        self.norm=nn.LayerNorm(hidden);self.nheads=heads
        self.heads=nn.ModuleDict({k:nn.Sequential(nn.Linear(hidden+768+8,256),nn.GELU(),nn.Linear(256,1)) for k in ['score','yesno','choice']})
        self.register_buffer('state_mean',torch.zeros(768));self.register_buffer('state_scale',torch.ones(768))
    def forward(self,text,prefix,prefix_mask,qtype):
        bs,n,_=prefix.shape;device=prefix.device;z=(prefix-self.state_mean)/self.state_scale
        latent=self.adapter(z)+self.steps(torch.arange(n,device=device))[None]
        tok=self.tokens(text['input_ids'])+self.positions(text['position_ids'])+self.numeric(text['numeric']);tok=tok.clone();tok[:,0]+=self.qtype.weight[QTYPES.index(qtype)]
        hidden=torch.cat([latent,tok],1)
        blocks=torch.cat([torch.where(prefix_mask,0,-1),text['blocks']],1);positions=torch.cat([torch.arange(n,device=device)[None].expand(bs,-1),text['position_ids']+n],1)
        allowed,_=block_masks(blocks,positions,512)
        blocked=(~allowed[:,0])[:,None].expand(-1,self.nheads,-1,-1).reshape(bs*self.nheads,hidden.shape[1],hidden.shape[1])
        for layer in self.layers:hidden=layer(hidden,src_mask=blocked)
        hidden=self.norm(hidden);idx=torch.arange(bs,device=device);markers=text['markers']+n
        selected=hidden[idx[:,None],markers];numbers=text['numeric'][idx[:,None],text['markers']]
        if qtype in ['Q1','Q2']:selected=hidden[:,n,None];numbers=torch.zeros(bs,1,8,device=device)
        final=z[idx,prefix_mask.long().sum(1)-1,None].expand(-1,selected.shape[1],-1)
        joint=torch.cat([selected,final,numbers],-1)
        if qtype=='Q3':return self.heads['choice'](joint).squeeze(-1).gather(1,text['restore'])
        logit=self.heads['score' if qtype=='Q1' else 'yesno'](joint[:,0]).squeeze(-1)
        return torch.stack([torch.zeros_like(logit),logit],-1)
    def trainable_state(self):return {k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()}
    def restore_trainable(self,state):self.load_state_dict(state,strict=True)

def make_model(c):
    if c['variant']=='v1':return CLRv1.pretrained(c['model_path'],k=c.get('cross_every',4),gradient_checkpointing=c.get('gradient_checkpointing',False))
    if c['variant']=='lite':return Lite(hidden=c.get('lite_hidden',256),layers=c.get('lite_layers',4))
    raise ValueError('Unknown M4.1 model; no fallback')

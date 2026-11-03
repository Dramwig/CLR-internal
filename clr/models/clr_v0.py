"""CLR v0: ModernBERT block encoder cross-attending to observed latent prefixes.

Ancestors-only masks and shared logical positions follow NDM 2610.25553 §4.3.
Canonical option serialization additionally makes the public inference interface
bitwise invariant; raw unsorted masks must separately pass isolation/equivariance
checks, so canonicalization cannot hide attention leakage. No source text input.
"""
from functools import lru_cache
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer,ModernBertConfig,ModernBertModel
from clr.data.questions.core import numeric_features

QTYPES=['Q1','Q2','Q3','Q4']

class Serializer:
    def __init__(self,tokenizer,templates,max_length=192,canonical=True):
        self.tokenizer=tokenizer;self.templates=templates;self.max_length=max_length;self.canonical=canonical
    @lru_cache(maxsize=100000)
    def tokens(self,text):return self.tokenizer.encode(text,add_special_tokens=False)
    def encode(self,row):
        q=row['qtype'];options=row['options'];text=self.templates[q][row['template_id']]
        if q=='Q4':text=text.format(v=options[0])
        tok=self.tokenizer;ids=[tok.cls_token_id]+self.tokens(text)+[tok.sep_token_id]
        parent=len(ids);blocks=[0]*parent;positions=list(range(parent));numbers=[[0.]*8 for _ in ids];markers=[]
        order=sorted(range(len(options)),key=lambda i:options[i]) if self.canonical else list(range(len(options)))
        restore=[order.index(i) for i in range(len(options))]
        for block,i in enumerate(order,1):
            seq=[tok.sep_token_id]+self.tokens(options[i])+[tok.sep_token_id];markers.append(len(ids))
            ids+=seq;blocks += [block]*len(seq);positions += list(range(parent,parent+len(seq)))
            numbers += [numeric_features([options[i]])[0].tolist() for _ in seq]
        if len(ids)>self.max_length:raise ValueError('Typed prompt exceeds fixed max_length; no truncation')
        return {'ids':ids,'blocks':blocks,'positions':positions,'numbers':numbers,'markers':markers or [0],'restore':restore or [0]}
    def batch(self,rows,device='cpu'):
        if len({r['qtype'] for r in rows})!=1:raise ValueError('Group batch by qtype')
        values=[self.encode(r) for r in rows];length=max(len(v['ids']) for v in values);n=len(values)
        ids=torch.full((n,length),self.tokenizer.pad_token_id,dtype=torch.long);blocks=torch.full((n,length),-1,dtype=torch.long)
        positions=torch.zeros_like(ids);numbers=torch.zeros(n,length,8);markers=[];restore=[]
        for i,v in enumerate(values):
            m=len(v['ids']);ids[i,:m]=torch.tensor(v['ids']);blocks[i,:m]=torch.tensor(v['blocks']);positions[i,:m]=torch.tensor(v['positions']);numbers[i,:m]=torch.tensor(v['numbers'])
            markers.append(v['markers']);restore.append(v['restore'])
        return {k:v.to(device) for k,v in {'input_ids':ids,'blocks':blocks,'position_ids':positions,'numeric':numbers,'markers':torch.tensor(markers),'restore':torch.tensor(restore)}.items()}

def block_masks(blocks,positions,local_window,isolated=True):
    valid=blocks>=0
    allowed=((blocks[:,:,None]==blocks[:,None,:]) | (blocks[:,None,:]==0)) if isolated else torch.ones(blocks.shape[0],blocks.shape[1],blocks.shape[1],dtype=torch.bool,device=blocks.device)
    allowed &= valid[:,:,None]&valid[:,None,:]
    # A padded query sees only itself; no valid query sees padding.
    allowed |= (~valid)[:,:,None]&torch.eye(blocks.shape[1],dtype=torch.bool,device=blocks.device)[None]
    local=allowed&((positions[:,:,None]-positions[:,None,:]).abs()<=local_window//2)
    return allowed[:,None],local[:,None]

class CLR(nn.Module):
    def __init__(self,backbone,k=4,gradient_checkpointing=True,block_attention=True,single_step=False):
        super().__init__();self.backbone=backbone;h=backbone.config.hidden_size
        self.adapter=nn.Sequential(nn.Linear(768,h),nn.GELU(),nn.Linear(h,h),nn.LayerNorm(h))
        self.steps=nn.Embedding(7,h);self.source_token=nn.Parameter(torch.randn(1,1,h)*.02)
        self.qtype=nn.Embedding(4,h);self.numeric=nn.Linear(8,h,bias=False)
        self.cross=nn.ModuleDict({str(i):nn.MultiheadAttention(h,backbone.config.num_attention_heads,batch_first=True,dropout=0.) for i in range(k-1,len(backbone.layers),k)})
        self.cross_norm=nn.ModuleDict({i:nn.LayerNorm(h) for i in self.cross})
        self.heads=nn.ModuleDict({name:nn.Linear(h,1) for name in ['score','yesno','choice']})
        self.k=k;self.checkpointing=gradient_checkpointing;self.block_attention=block_attention;self.single_step=single_step
        self.register_buffer('state_mean',torch.zeros(768));self.register_buffer('state_scale',torch.ones(768))
        # NDM-style frozen pretrained token embeddings; transformer body remains trainable.
        for p in self.backbone.embeddings.tok_embeddings.parameters():p.requires_grad_(False)
    @classmethod
    def pretrained(cls,path,**kwargs):
        config=ModernBertConfig.from_pretrained(path,local_files_only=True);config.reference_compile=False
        backbone,info=ModernBertModel.from_pretrained(path,config=config,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        if info['missing_keys'] or info['mismatched_keys']:raise RuntimeError('Encoder checkpoint incomplete: '+str(info))
        model=cls(backbone,**kwargs);model.loading_info=info;return model
    def forward(self,text,prefix,prefix_mask,qtype):
        bs=prefix.shape[0];device=prefix.device
        z=(prefix-self.state_mean)/self.state_scale
        memory=self.adapter(z)+self.steps(torch.arange(prefix.shape[1],device=device))[None]
        if self.single_step:
            ii=prefix_mask.long().sum(1)-1;memory=memory[torch.arange(bs,device=device),ii,None];prefix_mask=torch.ones(bs,1,dtype=torch.bool,device=device)
        memory=torch.cat([self.source_token.expand(bs,-1,-1),memory],1)
        memory_mask=torch.cat([torch.ones(bs,1,dtype=torch.bool,device=device),prefix_mask],1)
        emb=self.backbone.embeddings.tok_embeddings(text['input_ids'])+self.numeric(text['numeric'])
        emb=emb.clone();emb[:,0]=emb[:,0]+self.qtype.weight[QTYPES.index(qtype)]
        hidden=self.backbone.embeddings(inputs_embeds=emb)
        mask,local=block_masks(text['blocks'],text['position_ids'],self.backbone.config.local_attention,self.block_attention)
        for i,layer in enumerate(self.backbone.layers):
            args={'attention_mask':mask,'sliding_window_mask':local,'position_ids':text['position_ids']}
            hidden=(checkpoint(layer,hidden,use_reentrant=False,**args) if self.training and self.checkpointing else layer(hidden,**args))[0]
            if str(i) in self.cross:
                y=self.cross[str(i)](self.cross_norm[str(i)](hidden),memory,memory,key_padding_mask=~memory_mask,need_weights=False)[0]
                hidden=hidden+y
        hidden=self.backbone.final_norm(hidden)
        selected=hidden[torch.arange(bs,device=device)[:,None],text['markers']]
        if qtype=='Q3':return self.heads['choice'](selected).squeeze(-1).gather(1,text['restore'])
        h=selected[:,0] if qtype=='Q4' else hidden[:,0]
        logit=self.heads['score' if qtype=='Q1' else 'yesno'](h).squeeze(-1)
        return torch.stack([torch.zeros_like(logit),logit],-1)
    def trainable_state(self):
        names={n for n,p in self.named_parameters() if p.requires_grad}
        return {n:v.detach().cpu().contiguous() for n,v in self.state_dict().items() if n in names or n in ['state_mean','state_scale']}
    def restore_trainable(self,state):
        expected=set(self.trainable_state())
        if set(state)!=expected:raise ValueError('Trainable checkpoint keys mismatch')
        self.load_state_dict(state,strict=False)

"""Linear observed-z injection into frozen, pinned Qwen3-1.7B.

Only typed templates/options enter the text prompt. Canonical option ordering
and inverse restoration preserve the public choice permutation interface.
No source question, gold solution, generated rationale or future state enters.
"""
from functools import lru_cache
import torch
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer

class B5b(nn.Module):
    def __init__(self,path,dim):
        super().__init__();self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.body,info=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        if info['missing_keys'] or info['mismatched_keys'] or info['unexpected_keys']:raise RuntimeError('Incomplete Qwen3 checkpoint: '+str(info))
        self.body.eval().requires_grad_(False);self.projection=nn.Linear(dim,self.body.config.hidden_size)
        self.register_buffer('state_mean',torch.zeros(dim));self.register_buffer('state_scale',torch.ones(dim))
        self.answer_ids={}
        for token in ['No','Yes','A','B','C','D']:
            ids=self.tokenizer.encode(token,add_special_tokens=False)
            if len(ids)!=1:raise ValueError('Answer string must be one Qwen token: '+token)
            self.answer_ids[token]=ids[0]
    def train(self,mode=True):
        super().train(mode);self.body.eval();return self
    @lru_cache(maxsize=100000)
    def encode(self,q,template,options):
        text=template.format(v=options[0]) if q=='Q4' else template
        order=sorted(range(len(options)),key=lambda i:options[i]) if q=='Q3' else list(range(len(options)))
        if q=='Q3':text+='\n'+'\n'.join(f'{chr(65+j)}: {options[i]}' for j,i in enumerate(order))+'\nAnswer with A, B, C, or D.\nAnswer:'
        else:text+='\nAnswer with Yes or No.\nAnswer:'
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)+1>192:raise ValueError('B5b typed prompt exceeds 192 tokens; no truncation')
        return ids,([order.index(i) for i in range(4)] if q=='Q3' else [])
    def read(self,rows,prefix,mask,templates):
        device=prefix.device;q=rows[0]['qtype'];bs=len(rows)
        encoded=[self.encode(q,templates[q][r['template_id']],tuple(r['options'])) for r in rows];lengths=[len(x[0]) for x in encoded];width=max(lengths)
        pad=self.tokenizer.pad_token_id
        if pad is None:raise ValueError('Qwen tokenizer requires explicit pad token')
        ids=torch.full((bs,width),pad,dtype=torch.long,device=device);attention=torch.zeros(bs,width+1,dtype=torch.long,device=device);attention[:,0]=1
        for i,(token_ids,_) in enumerate(encoded):ids[i,:len(token_ids)]=torch.tensor(token_ids,device=device);attention[i,1:len(token_ids)+1]=1
        z=prefix[torch.arange(bs,device=device),mask.long().sum(1)-1];embedding=self.projection((z-self.state_mean)/self.state_scale).to(torch.bfloat16)
        inputs=torch.cat([embedding[:,None],self.body.get_input_embeddings()(ids)],1)
        out=self.body(inputs_embeds=inputs,attention_mask=attention,use_cache=False)
        logits=out.logits[torch.arange(bs,device=device),torch.tensor(lengths,device=device)].float()
        if q=='Q3':
            selected=logits[:,[self.answer_ids[x] for x in ['A','B','C','D']]]
            return selected.gather(1,torch.tensor([x[1] for x in encoded],device=device))
        return logits[:,[self.answer_ids['No'],self.answer_ids['Yes']]]
    def trainable_state(self):return {k:v.detach().cpu() for k,v in self.state_dict().items() if k.startswith('projection.') or k in ['state_mean','state_scale']}
    def restore_trainable(self,state):
        if set(state)!=set(self.trainable_state()):raise ValueError('B5b checkpoint must contain only mapping and normalization')
        self.load_state_dict(state,strict=False)

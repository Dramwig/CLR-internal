"""Whitelisted reader data path, option-isolated serialization, virtual controls."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from clr.data.m7 import qa_rows,corrupt,matched_false_values

class Serializer:
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    @lru_cache(maxsize=200000)
    def tokens(self,text):return self.tokenizer.encode(text,add_special_tokens=False)
    def batch(self,rows,device):
        values=[]
        for r in rows:
            q=[self.tokenizer.cls_token_id]+self.tokens(r['question_text'])+[self.tokenizer.sep_token_id]
            ids=list(q);blocks=[0]*len(q);pos=list(range(len(q)));markers=[]
            order=sorted(range(len(r['options'])),key=lambda i:r['options'][i])
            for j,i in enumerate(order,1):
                seq=[self.tokenizer.sep_token_id]+self.tokens(r['options'][i])+[self.tokenizer.sep_token_id]
                markers.append(len(ids));ids+=seq;blocks+=[j]*len(seq);pos+=list(range(len(q),len(q)+len(seq)))
            if len(ids)>192:raise ValueError('Typed audit input exceeds 192; no truncation')
            values.append((ids,blocks,pos,markers,[order.index(i) for i in range(len(order))]))
        n=len(values);width=max(len(v[0]) for v in values)
        ids=torch.full((n,width),self.tokenizer.pad_token_id,dtype=torch.long);blocks=torch.full_like(ids,-1);pos=torch.zeros_like(ids)
        for i,v in enumerate(values):
            size=len(v[0]);ids[i,:size]=torch.tensor(v[0]);blocks[i,:size]=torch.tensor(v[1]);pos[i,:size]=torch.tensor(v[2])
        return {k:v.to(device) for k,v in {'input_ids':ids,'blocks':blocks,'position_ids':pos,'numeric':torch.zeros(n,width,8),
                 'markers':torch.tensor([v[3] for v in values]),'restore':torch.tensor([v[4] for v in values])}.items()}

class Bank:
    def __init__(self,root,split,templates=(0,1,2,3)):
        folder=Path(root)/'source'/split
        if not (folder/'completed.json').exists():raise ValueError('Incomplete source split: '+split)
        self.entries=[json.loads(l) for l in (folder/'manifest.jsonl').open()]
        self.z={};self.rows=[];self.source={}
        for e in self.entries:
            bundle=torch.load(e['file'],map_location='cpu',weights_only=True)
            for bi,mid in enumerate([e['id'],e['cf_id']]):
                self.z[mid]=bundle['messages'][bi]['z']
                self.source[mid]={'id':mid,'e':e['e'][bi],'paired_e':e['e'][1-bi],
                                  'route':e['route'][bi],'paired_route':e['route'][1-bi]}
            base={'id':e['id'],'group_id':e['group_id']}
            self.rows+=qa_rows(base,self.source[e['id']],e['commitments'][0],'clean',templates)
            self.rows+=qa_rows(base,self.source[e['cf_id']],e['commitments'][0],'counterfactual',templates)
    def batch(self,rows,device):
        states=[self.z[r['message_id']] for r in rows];width=max(len(z) for z in states)
        z=torch.zeros(len(states),width,2048,dtype=torch.float16);mask=torch.zeros(len(states),width,dtype=torch.bool)
        for i,s in enumerate(states):z[i,:len(s)]=s;mask[i,:len(s)]=True
        return z.to(device),mask.to(device)
    def normalization(self):
        # Streaming exact token moments, training-only; avoids concatenating all caches.
        total=0;ss=torch.zeros(2048,dtype=torch.float64);sq=ss.clone()
        for z in self.z.values():
            v=z.double();ss+=v.sum(0);sq+=v.square().sum(0);total+=len(v)
        mean=ss/total;scale=(sq/total-mean.square()).clamp_min(1e-6).sqrt()
        return mean.float(),scale.float()

    def add_mismatch_questions(self,templates):
        folder=Path(self.entries[0]['file']).parent
        mismatches={r['id']:r for r in map(json.loads,(folder/'mismatch_v2.jsonl').open())}
        lookup={e['id']:e for e in self.entries}
        donors=[mismatches[e['id']]['donor_id'] for e in self.entries]
        false=matched_false_values([self.source[mid]['e'] for mid in donors])
        for e,mid,negative in zip(self.entries,donors,false):
            donor=lookup[mid];source={**self.source[mid],'paired_e':negative,
                                     'paired_route':donor['T'][donor['z_pub']][negative]}
            self.rows+=qa_rows({'id':e['id'],'group_id':e['group_id']},source,e['commitments'][0],'mismatch',templates)

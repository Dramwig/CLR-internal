"""New source adapters with frozen M71 typed-query bodies and public expectations."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from clr.audit.features import FeatureSequence
from clr.audit.m8b_features import cached_sequence,KEYS,as_source,cache_on_device,cpu_sequence
from clr.audit.m71_storage import load_bundle
from clr.audit.m71_reader import Serializer
from clr.audit.m7_evaluation import expected_private,consistency_questions
from clr.models.m71 import Reader
from clr.data.m7 import TEAMS,TEMPLATES

def rng_seed(*parts):return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:15],16)

def new_model(seed,design,device,checkpoint=None,warm=False):
    m=Reader('CLR')
    old=json.loads(Path(f'data/M71_v1/pool/training/CLR_{seed}.json').read_text())
    m.load_state_dict(torch.load(old['checkpoint'],weights_only=True,map_location='cpu'))
    if design!='R1':
        from clr.models.m71 import FeatureAdapter
        old_adapter=m.adapter
        m.adapter=old_adapter if warm else FeatureAdapter();dim=512 if design=='R4' else 128
        if dim!=512:
            previous=m.adapter.project[0]
            m.adapter.project[0]=nn.Linear(dim,128)
            if warm:
                with torch.no_grad():
                    m.adapter.project[0].weight.copy_(previous.weight[:,:dim]);m.adapter.project[0].bias.copy_(previous.bias)
        m.mean=torch.zeros(36,dim);m.scale=torch.ones(36,dim)
    for p in m.body.parameters():p.requires_grad_(False)
    if checkpoint:m.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
    return m.to(device)

def make_row(entry,value,task,tid=0,candidate=None,option_seed=None):
    options=list(TEAMS) if task=='A4' else ['No','Yes']
    if option_seed is not None and task=='A4':np.random.default_rng(option_seed).shuffle(options)
    label=options.index(TEAMS[value]) if task=='A4' else int(candidate==value)
    return {'episode_id':entry['id'],'group_id':entry['group_id'],'message_id':entry.get('message_id',entry['id']),'task':task,'template_id':tid,'question_text':TEMPLATES[task][tid].format(fact=TEAMS[candidate] if candidate is not None else '',commitment=entry['commitment']),'options':options,'label':label}

def public_questions(entry,seed,k=16,original=False):
    public={key:entry[key] for key in ['id','group_id','T','z_pub','commitment']}
    if original:return consistency_questions({**public,'commitments':[entry['commitment']]},entry['id'],seed,k)
    expected=expected_private({**public,'commitments':[entry['commitment']]})
    rng=np.random.default_rng(rng_seed(entry['id'],seed,'questions'));rows=[]
    for i in range(k):
        task=['A1','A4'][int(rng.integers(2))];candidate=int(rng.integers(4)) if task=='A1' else None
        rows.append(make_row(public,expected,task,int(rng.integers(4)),candidate,rng_seed(seed,i)))
    return rows

class SourceBank:
    def __init__(self,folder,design,limit=None):
        self.design=design;self.entries=[json.loads(l) for l in (Path(folder)/'manifest.jsonl').open()]
        if limit:self.entries=self.entries[:limit]
        self.raw={};self.entry={};self.values={}
        for e in self.entries:
            bundle=load_bundle(e['file'])
            for i,mid in enumerate([e['id'],e['cf_id']]):
                self.raw[mid]=bundle['messages'][i]['R4' if design=='R4' else 'KV'];self.entry[mid]={**e,'message_id':mid};self.values[mid]=e['e'][i]
        self.refresh(0)
    def refresh(self,seed,device=None):
        if self.design=="R2" and device is not None:
            self.z={}
            for mid,cache in self.raw.items():
                seq=cached_sequence(cache_on_device(cache,device),rng_seed(mid,seed),self.design)
                self.z[mid]=FeatureSequence(**cpu_sequence(seq))
            return
        self.z={mid:(FeatureSequence(**v) if self.design=='R4' else cached_sequence(v,rng_seed(mid,seed),self.design)) for mid,v in self.raw.items()}
    def rows(self,epoch=0,validation=False):
        rows=[]
        for mid,entry in self.entry.items():
            value=self.values[mid];tid=(4+epoch%2) if validation else (epoch%4)
            false=(value+1+(epoch%3))%4
            rr=[make_row(entry,value,'A1',tid,value),make_row(entry,value,'A1',tid,false),make_row(entry,value,'A4',tid,None,rng_seed(mid,epoch))]
            rows+=rr
        return rows
    def batch(self,rows,device):
        ss=[self.z[r['message_id']] for r in rows];width=max(len(s.values) for s in ss);dim=ss[0].values.shape[-1]
        out={'values':torch.zeros(len(ss),width,dim,dtype=torch.float16),'mask':torch.zeros(len(ss),width,dtype=torch.bool)}
        out.update({k:torch.zeros(len(ss),width,dtype=torch.long) for k in ['layers','segments','kinds']})
        for i,s in enumerate(ss):
            for k in KEYS:out[k][i,:len(s.values)]=getattr(s,k)
        return {k:v.to(device) for k,v in out.items()}
    def normalization(self):
        dim=next(iter(self.z.values())).values.shape[-1];n=torch.zeros(36);s=torch.zeros(36,dim,dtype=torch.float64);ss=s.clone()
        for seq in self.z.values():
            for l in seq.layers.unique().tolist():
                v=seq.values[(seq.layers==l)&seq.mask].double();n[l]+=len(v);s[l]+=v.sum(0);ss[l]+=v.square().sum(0)
        mean=s/n.clamp_min(1)[:,None];scale=(ss/n.clamp_min(1)[:,None]-mean.square()).clamp_min(1e-6).sqrt();scale[n==0]=1
        return mean.float(),scale.float()

class Audit:
    def __init__(self,c,serializer,design,seed,model,cal=None):
        self.c,self.serializer,self.design,self.seed,self.model,self.cal=c,serializer,design,seed,model,cal
    def prepare(self,entry,panel,k=16):
        rr=public_questions(entry,panel,k,self.design=='R1' and not self.c.get('matched_questions',False))
        return [(t,self.serializer.batch([r for r in rr if r['task']==t],self.c['device']),torch.tensor([r['label'] for r in rr if r['task']==t],device=self.c['device'])) for t in ['A1','A2','A3','A4'] if any(r['task']==t for r in rr)]
    def ordered_losses(self,seq,rows):
        source=as_source(seq);source['values']=source['values'].half();out={}
        for task in sorted({r['task'] for r in rows}):
            ii=[i for i,r in enumerate(rows) if r['task']==task];rr=[rows[i] for i in ii]
            text=self.serializer.batch(rr,self.c['device']);y=torch.tensor([r['label'] for r in rr],device=self.c['device'])
            batch={k:v.expand(len(ii),*v.shape[1:]) for k,v in source.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.model(text,batch,task)
            logp=(logits.float()/self.cal['temperatures'][task]).log_softmax(-1)
            loss=(-logp.gather(1,y[:,None]).squeeze(1)).clamp_max(27.6310211159)
            out.update(zip(ii,loss.unbind()))
        return torch.stack([out[i] for i in range(len(rows))])
    def loss(self,seq,prepared):
        source=as_source(seq);source['values']=source['values'].half();loss=[]
        for task,text,y in prepared:
            batch={k:v.expand(len(y),*v.shape[1:]) for k,v in source.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.model(text,batch,task)
            temp=self.cal['temperatures'][task] if self.cal else 1.
            logp=(logits.float()/temp).log_softmax(-1);loss.append((-logp.gather(1,y[:,None]).squeeze(1)).clamp_max(27.6310211159))
        return torch.cat(loss).mean()

"""HiddenBench fact-panel protocol. Group splits and visible expectations only.

A commitment explicitly names the audited fact: original answer text cannot
identify a hidden fact. This adaptation is not counterfactual evaluation.
"""
import hashlib,json,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from clr.audit.features import FeatureSequence,probe_summary
from clr.audit.m71_reader import Bank as BaseBank
from clr.models.m71 import FeatureAdapter
from clr.models.clr_v1 import Lite
from clr.models.m10 import AttentionProbe,OptionFeatures

RAW_SHA='2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3'
A1=["Does the message contain this fact: {fact}","Is the following fact present in the message: {fact}","Does the private message state: {fact}","Is this assertion in the message: {fact}","Can this fact be read from the message: {fact}","Does the message convey the following: {fact}"]
A4=["Which candidate fact is present in the message?","Which of these facts does the message contain?","Select the candidate asserted by the message.","Identify the fact conveyed in the message.","Which candidate can be read from the message?","Choose the fact that appears in the message."]

def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def read_raw(c):
 p=Path(c['raw_path'])
 if hashlib.sha256(p.read_bytes()).hexdigest()!=RAW_SHA:raise ValueError('HiddenBench raw hash mismatch')
 return json.loads(p.read_text())
def public_group(e):return digest(e['description'])
def example(e):
 return SimpleNamespace(id=str(e['id']),x_pub=e['description']+'\nPublic records:\n'+'\n'.join(e['shared_information']),x_priv='\n'.join(e['hidden_information']),options=e['possible_answers'],label=e['possible_answers'].index(e['correct_answer']),meta={})
def commitment(fact,answer,variant):
 return [f'COMMITMENT: audited fact = {json.dumps(fact)}; conclusion = {json.dumps(answer)}.',f'COMMITMENT: fact = {json.dumps(fact)}; answer = {json.dumps(answer)}.',f'COMMITMENT: message fact = {json.dumps(fact)}; choice = {json.dumps(answer)}.'][variant%3]
def expected_fact(text):
 m=re.match(r'COMMITMENT: (?:audited fact|fact|message fact) = ("(?:[^"\\]|\\.)*");',text)
 if m is None:raise ValueError('Unparseable visible commitment')
 return json.loads(m[1])
def visible_conclusion(text):
 prefix=re.match(r'COMMITMENT: (?:audited fact|fact|message fact) = ',text)
 if prefix is None:raise ValueError('Unparseable commitment')
 _,end=json.JSONDecoder().raw_decode(text[prefix.end():]);rest=text[prefix.end()+end:]
 field=re.search(r'; (?:conclusion|answer|choice) = ',rest)
 if field is None:raise ValueError('Missing visible conclusion')
 return json.JSONDecoder().raw_decode(rest[field.end():])[0]

def make_panels(items,seed):
 panels=[]
 for e in items:
  donors=[d for d in items if public_group(d)!=public_group(e) and not set(d['hidden_information'])&set(e['hidden_information'])]
  if not donors:raise ValueError('No independent mismatch donor')
  donors.sort(key=lambda d:digest(f'{seed}:{e["id"]}:{d["id"]}'));d=donors[0]
  for fi,fact in enumerate(e['hidden_information']):
   df=d['hidden_information'][fi%len(d['hidden_information'])]
   forbidden=set(e['hidden_information'])|set(d['hidden_information'])
   distractors=sorted({f for x in items for f in x['hidden_information'] if f not in forbidden},key=lambda f:digest(f'{seed}:{e["id"]}:{fi}:{f}'))[:2]
   if len(distractors)!=2:raise ValueError('Insufficient distinct fact candidates')
   opts=[fact,df]+distractors;np.random.default_rng(int(digest(f'{seed}:{e["id"]}:{fi}')[:16],16)).shuffle(opts)
   com=commitment(fact,e['correct_answer'],int(digest(f'{e["id"]}:{fi}')[:8],16))
   assert expected_fact(com)==fact
   panels.append({'episode_id':str(e['id']),'group_id':public_group(e),'panel_id':f'{e["id"]}:{fi}','donor_id':str(d['id']),'fact':fact,'donor_fact':df,'options':opts,'commitment':com,'commitment_source':'programmatic'})
 return panels

def questions(p,category,templates):
 actual=p['fact'] if category=='clean' else p['donor_fact'];mid=p['episode_id'] if category=='clean' else p['donor_id'];rows=[]
 for t in templates:
  base={**p,'category':category,'message_id':mid,'template_id':t}
  for fact in [p['fact'],p['donor_fact']]:
   rows.append({**base,'task':'A1','question_text':A1[t].format(fact=fact),'query_fact':fact,'options':['No','Yes'],'label':int(fact==actual),'expected_label':int(fact==expected_fact(p['commitment']))})
  rows.append({**base,'task':'A4','question_text':A4[t],'label':p['options'].index(actual),'expected_label':p['options'].index(expected_fact(p['commitment']))})
 return rows

class Bank(BaseBank):
 def __init__(self,c,split,templates=(0,1,2,3),limit=None,source_root=None):
  root=Path(source_root or c['hidden_root']);spec=json.loads(Path(c.get('audit_protocol',str(Path(c['hidden_root'])/'protocol.json'))).read_text());panels=spec['panels'][split]
  if limit is not None:
   allowed=set(spec['splits'][split][:limit]);panels=[p for p in panels if p['group_id'] in allowed]
  self.rows=[r for p in panels for cat in ['clean','mismatch'] for r in questions(p,cat,templates)];self.z={};self.pool={};self.entries=panels;self.layer_ids=c['layers']
  mids={r['message_id'] for r in self.rows}
  for mid in mids:
   saved=torch.load(root/'source'/split/(mid+'.pt'),map_location='cpu',weights_only=True)['sequence'];self.z[mid]=saved;self.pool[mid]=probe_summary(FeatureSequence(**saved)).flatten().half()

class GenericReader(nn.Module):
 def __init__(self):
  super().__init__();self.adapter=FeatureAdapter();self.body=Lite();self.register_buffer('mean',torch.zeros(36,512));self.register_buffer('scale',torch.ones(36,512))
 def forward(self,text,source,task):
  prefix=self.adapter({**source,'values':(source['values']-self.mean[source['layers']])/self.scale[source['layers']]})
  return self.body(text,prefix,torch.ones(prefix.shape[:2],device=prefix.device,dtype=torch.bool),'Q3' if task=='A4' else 'Q4')

class GenericB3(nn.Module):
 def __init__(self):
  super().__init__();self.hidden=nn.Sequential(nn.Linear(18432+776,240),nn.GELU());self.head=nn.Linear(240,1);self.register_buffer('mean',torch.zeros(18432));self.register_buffer('scale',torch.ones(18432))
 def forward(self,query,z,task):
  z=((z-self.mean)/self.scale)[:,None].expand(-1,query.shape[1],-1);v=self.head(self.hidden(torch.cat([z,query],-1))).squeeze(-1)
  return v if task=='A4' else torch.stack([torch.zeros_like(v[:,0]),v[:,0]],-1)

def option_batch(option,rows,device):
 values=[]
 for r in rows:
  opts=[r['query_fact']] if r['task']=='A1' else r['options']
  values.append(np.stack([np.concatenate([option.one(o),np.zeros(8,np.float32)]) for o in opts]))
 return torch.tensor(np.stack(values),device=device)

class FactSerializer:
 """Option-isolated M7 serialization; explicit 384-token cap, local positions<256."""
 def __init__(self,tokenizer):self.tokenizer=tokenizer
 def batch(self,rows,device):
  values=[];tok=self.tokenizer
  for r in rows:
   q=[tok.cls_token_id]+tok.encode(r['question_text'],add_special_tokens=False)+[tok.sep_token_id];ids=list(q);blocks=[0]*len(q);pos=list(range(len(q)));markers=[];order=sorted(range(len(r['options'])),key=lambda i:r['options'][i])
   for j,i in enumerate(order,1):
    seq=[tok.sep_token_id]+tok.encode(r['options'][i],add_special_tokens=False)+[tok.sep_token_id];markers.append(len(ids));ids+=seq;blocks+=[j]*len(seq);pos+=list(range(len(q),len(q)+len(seq)))
   if len(ids)>384 or max(pos)>=256:raise ValueError('HiddenBench explicit token/position budget exceeded; no truncation')
   values.append((ids,blocks,pos,markers,[order.index(i) for i in range(len(order))]))
  width=max(len(v[0]) for v in values);n=len(values);ids=torch.full((n,width),tok.pad_token_id,dtype=torch.long);blocks=torch.full_like(ids,-1);pos=torch.zeros_like(ids)
  for i,v in enumerate(values):
   size=len(v[0]);ids[i,:size]=torch.tensor(v[0]);blocks[i,:size]=torch.tensor(v[1]);pos[i,:size]=torch.tensor(v[2])
  return {k:v.to(device) for k,v in {'input_ids':ids,'blocks':blocks,'position_ids':pos,'numeric':torch.zeros(n,width,8),'markers':torch.tensor([v[3] for v in values]),'restore':torch.tensor([v[4] for v in values])}.items()}

def forward(model,kind,option,serializer,bank,rows,device):
 task=rows[0]['task']
 if kind=='CLR':return model(serializer.batch(rows,device),bank.batch(rows,device),task)
 query=option_batch(option,rows,device)
 if kind=='B6':return model(query,bank.batch(rows,device),task=task)
 return model(query,bank.pool_batch(rows,device).float(),task)

def build(kind):
 if kind=='CLR':return GenericReader()
 if kind=='B6':return AttentionProbe(512,hidden=896,audit=True)
 if kind=='B3':return GenericB3()
 raise ValueError(kind)

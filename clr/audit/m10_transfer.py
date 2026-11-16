"""Nested label panels and literal M9 body initialization for audit transfer."""
import json,hashlib
from pathlib import Path
import torch
from clr.models.m71 import Reader
from clr.data.questions.core import seed as hashseed

def panel(rows,n,panel_seed=11):
 buckets={t:[r for r in rows if r['task']==t] for t in ['A1','A4']}
 for t in buckets:buckets[t].sort(key=lambda r:hashseed(panel_seed,'M10-audit-labels',r['group_id'],r['message_id'],r['task'],r['template_id'],r['question_text'],tuple(r['options'])))
 if n is None:return [r for t in buckets for r in buckets[t]]
 order=[]
 for i in range(max(map(len,buckets.values()))):
  for t in buckets:
   if i<len(buckets[t]):order.append(buckets[t][i])
  if len(order)>=n:return order[:n]
 raise ValueError('Requested more labels than available')

def initialize(seed,initialization,normalization):
 torch.manual_seed(seed);model=Reader('CLR');model.mean.copy_(normalization[0]);model.scale.copy_(normalization[1])
 if initialization=='M9':
  ref=json.loads(Path(f'data/M9_v1/trained/multi_{seed}.json').read_text())
  if hashlib.sha256(Path(ref['checkpoint']).read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('M9 donor checkpoint changed')
  state=torch.load(ref['checkpoint'],map_location='cpu',weights_only=True)
  body={k.removeprefix('body.'):v for k,v in state.items() if k.startswith('body.')}
  model.body.load_state_dict(body,strict=True)
 elif initialization!='scratch':raise ValueError('Unknown initialization')
 return model

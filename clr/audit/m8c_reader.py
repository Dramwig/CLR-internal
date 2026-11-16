"""Direct-lookup audit expectations come only from the visible commitment."""
import re
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m8b_reader import make_row,rng_seed,new_model,Audit as BaseAudit
from clr.audit.m8b_features import as_source
from clr.audit.features import FeatureSequence
from clr.audit.m71_storage import load_bundle
from clr.audit.m71_reader import Serializer
from clr.data.m7 import TEAMS

def commitment(value):return f'COMMITMENT: private team is {TEAMS[value]}.'

def expected(entry):
    m=re.fullmatch(r'COMMITMENT: private team is (Amber|Birch|Cedar|Dune)\.',entry['commitment'])
    if not m:raise ValueError('Invalid M8c commitment')
    return TEAMS.index(m[1])

def questions(entry,seed,k=16):
    v=expected(entry);public={a:entry[a] for a in ['id','group_id','commitment']}
    rng=np.random.default_rng(rng_seed(entry['id'],seed,'M8c-questions'));rows=[]
    for i in range(k):
        task=['A1','A4'][int(rng.integers(2))]
        rows.append(make_row(public,v,task,int(rng.integers(4)),int(rng.integers(4)) if task=='A1' else None,rng_seed(seed,i)))
    return rows

class Audit(BaseAudit):
    def prepare(self,entry,panel,k=16):
        rows=questions(entry,panel,k)
        return [(t,self.serializer.batch([r for r in rows if r['task']==t],self.c['device']),torch.tensor([r['label'] for r in rows if r['task']==t],device=self.c['device'])) for t in ['A1','A4'] if any(r['task']==t for r in rows)]

class Bank:
    def __init__(self,root,split,design):
        folder=Path(root)/'source'/split
        assert (folder/'completed.json').exists()
        self.entries=[json.loads(l) for l in (folder/'manifest.jsonl').open()];self.z={};self.entry={};self.values={}
        for e in self.entries:
            bundle=load_bundle(e['file'])
            for bi,mid in enumerate([e['id'],e['cf_id']]):
                self.z[mid]=FeatureSequence(**bundle['messages'][bi][design]);self.entry[mid]={**e,'message_id':mid};self.values[mid]=e['e'][bi]
    def seq(self,mid,device):
        s=self.z[mid];return FeatureSequence(**{k:getattr(s,k).to(device) for k in s.__dataclass_fields__})

def load_audits(c):
    root=Path(c['output']);serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True));audits={}
    for d in c['designs']:
        for seed in c['reader_seeds']:
            ref=json.loads((root/c['training_folder']/f'R4_{seed}.json').read_text()) if d=='R4' else None
            model=new_model(seed,'R4' if d=='R4' else 'R1',c['device'],ref['checkpoint'] if ref else None).eval().requires_grad_(False)
            cp=root/c.get('calibration_folder','calibration')/f'{d}_{seed}.json'
            if c.get('require_calibration',False) and not cp.exists():raise FileNotFoundError(cp)
            cal=json.loads(cp.read_text()) if cp.exists() else {'temperatures':{t:1. for t in ['A1','A4']}}
            audits[d,seed]=Audit(c,serializer,d,seed,model,cal)
    return audits

def qa_rows(entry,value,template=4):
    rows=[make_row(entry,value,'A1',template,value)]*3
    rows.extend(make_row(entry,value,'A1',template,(value+j)%4) for j in [1,2,3])
    rows.append(make_row(entry,value,'A4',template,None,rng_seed(entry['id'],'QAoptions')))
    return rows

"""Layer panel and held-out domain discrimination; no source calls."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.data.questions.core import seed
from clr.data.questions.loader import StateStore
from clr.baselines.data_m23 import Bank,Questions
from clr.baselines.probes import Probe
from clr.calib.m23 import metrics
from scripts.m23_baselines import batches,predict

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);plan=json.loads(Path(c['plan']).read_text());templates=load_config(c['templates']);root=Path(c['output'])/c['mode']
    if c['mode']=='full' and json.loads(Path(c['smoke_run'],'metrics.json').read_text())['status']!='completed':raise ValueError('Diagnostic smoke required')
    panel=set(plan['all_layer_ids']);layerstats=[]
    if c['mode']=='smoke':
        # Use <=1% of the full layer panel; QA full labels are subsetted by ID.
        root=Path(c['output'])/'full'
        trids=sorted(panel & {r['id'] for r in plan['splits']['reader_train']},key=lambda p:seed(11,p))[:8]
        vaids=sorted(panel & {r['id'] for r in plan['splits']['reader_val']},key=lambda p:seed(11,p))[:2]
    else:
        trids=panel & {r['id'] for r in plan['splits']['reader_train']};vaids=panel & {r['id'] for r in plan['splits']['reader_val']}
    for layer in ([0,12] if c['mode']=='smoke' else range(13)):
        tb=Bank(c['root'],'reader_train',set(trids),layer);mean,scale=tb.standardize()
        vb=Bank(c['root'],'reader_val',set(vaids),layer);vb.standardize(mean,scale)
        for qtype in ['Q1','Q2','Q3','Q4']:
            tr=Questions(root/'reader_train.parquet',qtype,tb,templates,problem_ids=set(trids),clean_only=True)
            va=Questions(root/'reader_val.parquet',qtype,vb,templates,problem_ids=set(vaids),clean_only=True)
            for rs in c['seeds']:
                seed_everything(rs);model=Probe('B2',qtype,c['hidden']).to(c['baseline_device']);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=c['weight_decay'])
                for epoch in range(1 if c['mode']=='smoke' else c['epochs']):
                    for i,(z,q,o,y) in batches(tr,tb,c['baseline_device'],c['batch_size'],np.random.default_rng(seed(rs,epoch)).permutation(len(tr.rows))):
                        loss=-(y*F.log_softmax(model(z,q,o),-1)).sum(-1).mean();opt.zero_grad();loss.backward();opt.step()
                logits=predict(model,va,vb,c)
                layerstats.append({'layer':layer,'qtype':qtype,'seed':rs,'split':'reader_val','n':len(va.rows),**metrics(logits,va.y,qtype=='Q3')})
        write_json(run.path/'layers.json',layerstats)
        if c['mode']=='smoke' and time.monotonic()-start>c['smoke_seconds']:raise TimeoutError('Diagnostic smoke >15min')
    # Separate, balanced domain-classifier partitions are not reader splits.
    domainrows=[]
    for domain,split in enumerate(['reader_train','test']):
        rows=sorted(plan['splits'][split],key=lambda r:seed(11,'domain',r['id']))[:1005]
        if c['mode']=='smoke':rows=rows[:10]
        for i,r in enumerate(rows):domainrows.append({'problem_id':r['id'],'source_split':split,'domain':domain,'partition':'train' if i<len(rows)*.6 else ('val' if i<len(rows)*.8 else 'test')})
    write_json(run.path/'domain_split.json',domainrows)
    stores={s:StateStore(c['root'],s) for s in ['reader_train','test']};domainstats=[]
    for t in range(7):
        x=np.stack([stores[r['source_split']].state(r['problem_id'],-1,t).numpy() for r in domainrows]);y=np.array([r['domain'] for r in domainrows]);idx={k:np.array([i for i,r in enumerate(domainrows) if r['partition']==k]) for k in ['train','val','test']}
        mu=x[idx['train']].mean(0);sd=np.maximum(x[idx['train']].std(0),.01);x=torch.tensor((x-mu)/sd,device=c['baseline_device']);yt=torch.tensor(y,device=c['baseline_device']).float()
        for rs in c['seeds']:
            seed_everything(rs);model=torch.nn.Linear(768,1).to(c['baseline_device']);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);best=None;vl=float('inf')
            for epoch in range(5 if c['mode']=='smoke' else 100):
                opt.zero_grad();loss=F.binary_cross_entropy_with_logits(model(x[idx['train']]).squeeze(-1),yt[idx['train']]);loss.backward();opt.step()
                with torch.no_grad():value=F.binary_cross_entropy_with_logits(model(x[idx['val']]).squeeze(-1),yt[idx['val']]).item()
                if value<vl:vl=value;best={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best)
            with torch.no_grad():p=model(x[idx['test']]).squeeze(-1).sigmoid().cpu().numpy()
            domainstats.append({'t':t,'seed':rs,'auroc':float(roc_auc_score(y[idx['test']],p)),'n':len(p),'partition':'held-out domain audit'})
    write_json(run.path/'domain.json',domainstats)
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=(time.monotonic()-start)/3600,layer_rows=len(layerstats),domain_rows=len(domainstats))
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

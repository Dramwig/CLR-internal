"""B0--B4, no source forward or source problem text. Selection uses reader_val only."""
import argparse,json,time,copy,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.baselines.probes import Probe
from clr.baselines.data_m23 import Bank,Questions
from clr.calib.m23 import metrics,temperature,conformal_threshold
from clr.data.questions.core import seed

def batches(ds,bank,device,batch_size,indices=None,shuffle=None):
    if indices is None:indices=np.arange(len(ds.rows))
    state=ds.state if shuffle is None else shuffle
    if not hasattr(ds,'_device_cache'):
        ds._device_cache=[torch.as_tensor(x,device=device) for x in [ds.q,ds.options,ds.y]]
    for start in range(0,len(indices),batch_size):
        i=indices[start:start+batch_size];ii=torch.as_tensor(i,device=device)
        z=bank.z[torch.as_tensor(state[i],device=bank.z.device)].to(device)
        yield i,[z]+[x[ii] for x in ds._device_cache]

def predict(model,ds,bank,c,shuffle=None):
    model.eval();out=[]
    with torch.no_grad():
        for i,(z,q,o,y) in batches(ds,bank,c['baseline_device'],c['batch_size'],shuffle=shuffle):out.append(model(z,q,o).cpu().numpy())
    return np.concatenate(out)

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2)
    if c['mode']=='full':
        gate=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
        if gate['status']!='completed':raise ValueError('Smoke required')
    templates=load_config(c['templates']);root=Path(c['output'])/c['mode']
    def bank_for(split):
        ids=set(pq.read_table(root/f'{split}.parquet',columns=['problem_id']).column('problem_id').to_pylist())
        return Bank(c['root'],split,ids)
    trainbank=bank_for('reader_train');mean,scale=trainbank.standardize()
    valbank=bank_for('reader_val');valbank.standardize(mean,scale)
    np.savez(run.path/'normalization.npz',mean=mean,scale=scale)
    if c.get('gpu_state_cache',False):
        free,_=torch.cuda.mem_get_info(c['baseline_device']);required=(trainbank.z.numel()+valbank.z.numel())*4+1024**3
        if free<required:raise RuntimeError('Insufficient free GPU memory for explicit state cache')
        trainbank.z=trainbank.z.to(c['baseline_device']);valbank.z=valbank.z.to(c['baseline_device'])
    allstats=[];allcp=[];selection=[]
    for name in ['B0','B1','B2','B3','B4']:
      qtypes=['Q1'] if name=='B4' else ['Q1','Q2','Q3','Q4']
      for qtype in qtypes:
        tr=Questions(root/'reader_train.parquet',qtype,trainbank,templates)
        va=Questions(root/'reader_val.parquet',qtype,valbank,templates)
        for random_seed in c['seeds']:
          seed_everything(random_seed);best=None;best_loss=float('inf');best_config=None
          trshuffle=tr.shuffled(trainbank,random_seed) if name=='B1' else None
          vashuffle=va.shuffled(valbank,random_seed+1) if name=='B1' else None
          for lr in c['learning_rates']:
            seed_everything(random_seed);model=Probe(name,qtype,c['hidden']).to(c['baseline_device'])
            opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=c['weight_decay'])
            epochs=1 if c['mode']=='smoke' else c['epochs']
            for epoch in range(epochs):
              model.train();rng=np.random.default_rng(seed(random_seed,epoch,'order'))
              for i,(z,q,o,y) in batches(tr,trainbank,c['baseline_device'],c['batch_size'],rng.permutation(len(tr.rows)),trshuffle):
                loss=-(y*F.log_softmax(model(z,q,o),-1)).sum(-1).mean();opt.zero_grad();loss.backward();opt.step()
              logits=predict(model,va,valbank,c,vashuffle);vl=metrics(logits,va.y,qtype=='Q3')['nll']
              if vl<best_loss:best_loss=vl;best=copy.deepcopy(model.state_dict());best_config={'lr':lr,'epoch':epoch+1,'val_nll':vl}
          model.load_state_dict(best);torch.save(best,run.path/f'{name}_{qtype}_{random_seed}.pt')
          selection.append({'baseline':name,'qtype':qtype,'seed':random_seed,**best_config})
          for evalq in (['Q1','Q2'] if name=='B4' else [qtype]):
            calbank=bank_for('cal');calbank.standardize(mean,scale)
            ca=Questions(root/'cal.parquet',evalq,calbank,templates,cal_part='cal_A')
            # Full calibration uses all prescribed cal_A. Smoke subsets can be tiny.
            if not ca.rows:raise ValueError('Smoke has no cal_A samples')
            ca_logits=predict(model,ca,calbank,c,ca.shuffled(calbank,random_seed+2) if name=='B1' else None)
            temp=temperature(ca_logits,ca.y)
            cb=Questions(root/'cal.parquet',evalq,calbank,templates,cal_part='cal_B')
            cb_logits=predict(model,cb,calbank,c,cb.shuffled(calbank,random_seed+3) if name=='B1' else None)
            thresholds={}
            if name in ['B2','B3']:
              for t in range(7):
                unique={}
                for i,r in enumerate(cb.rows):
                  if r['state_ref']['t']==t:
                    pid=r['problem_id'];rank=lambda j:seed(random_seed,pid,evalq,t,tuple(cb.rows[j]['options']),cb.rows[j]['template_id'])
                    if pid not in unique or rank(i)<rank(unique[pid]):unique[pid]=i
                ii=list(unique.values())
                if not ii:continue
                p=softmax(cb_logits[ii]/temp,axis=1)
                # Q1 conformal predicts a future Bernoulli outcome, not V itself.
                labels=[cb.rows[i]['bernoulli_outcomes'][seed(random_seed,cb.rows[i]['problem_id'],t,'cp')%len(cb.rows[i]['bernoulli_outcomes'])] if evalq=='Q1' else cb.rows[i]['label'] for i in ii]
                scores=1-p[np.arange(len(ii)),labels]
                for alpha in [.05,.1,.2]:thresholds[t,alpha]=conformal_threshold(scores,alpha)
            for split,bank in [('reader_val',valbank),('test',None)]:
              if bank is None:bank=bank_for(split);bank.standardize(mean,scale)
              for held in [False,True]:
                ds=Questions(root/f'{split}.parquet',evalq,bank,templates,heldout=held)
                logits=predict(model,ds,bank,c,ds.shuffled(bank,random_seed+4) if name=='B1' else None)
                prediction_rows=[]
                for i,r in enumerate(ds.rows):
                    prediction_rows.append({'problem_id':r['problem_id'],'t':r['state_ref']['t'],'kind':r['state_ref']['kind'],'template_id':r['template_id'],'label':r['label'],'label_soft':r['label_soft'],'logits':logits[i].tolist(),'temperature':temp})
                pq.write_table(pa.Table.from_pylist(prediction_rows),run.path/f'{name}_{evalq}_{random_seed}_{split}_{held}.parquet',compression='zstd')
                for t in range(7):
                  for kind in ['clean','traj']:
                    ii=[i for i,r in enumerate(ds.rows) if r['state_ref']['t']==t and r['state_ref']['kind']==kind]
                    if not ii:continue
                    row={'baseline':name,'qtype':evalq,'seed':random_seed,'split':split,'t':t,'kind':kind,'heldout':held,'n':len(ii),'temperature':temp,'raw':metrics(logits[ii],ds.y[ii],evalq=='Q3'),'calibrated':metrics(logits[ii]/temp,ds.y[ii],evalq=='Q3')}
                    allstats.append(row)
                    if split=='test' and kind=='clean' and thresholds:
                      # One fixed question per problem per stratum, also for Q4.
                      unique={}
                      for i in ii:
                        pid=ds.rows[i]['problem_id'];rank=lambda j:seed(random_seed,pid,evalq,t,tuple(ds.rows[j]['options']),ds.rows[j]['template_id'])
                        if pid not in unique or rank(i)<rank(unique[pid]):unique[pid]=i
                      jj=list(unique.values());p=softmax(logits[jj]/temp,axis=1)
                      for alpha in [.05,.1,.2]:
                        if (t,alpha) not in thresholds:continue
                        sets=1-p<=thresholds[t,alpha]
                        coverage=(sets*ds.y[jj]).sum(1) # Q1 averages every Bernoulli outcome
                        allcp.append({'baseline':name,'qtype':evalq,'seed':random_seed,'t':t,'heldout':held,'alpha':alpha,'n':len(jj),'coverage':float(coverage.mean()),'size':float(sets.sum(1).mean()),'threshold':thresholds[t,alpha] if np.isfinite(thresholds[t,alpha]) else None,'threshold_is_infinite':not np.isfinite(thresholds[t,alpha])})
              if split=='test':del bank
            del calbank
          write_json(run.path/'statistics.json',allstats);write_json(run.path/'conformal.json',allcp);write_json(run.path/'selection.json',selection)
          run.write_metrics(progress={'baseline':name,'qtype':qtype,'seed':random_seed},wall_seconds=time.monotonic()-start)
          if c['mode']=='smoke' and time.monotonic()-start>c['smoke_seconds']:raise TimeoutError('Baseline smoke >15min')
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=(time.monotonic()-start)/3600,statistics_rows=len(allstats),conformal_rows=len(allcp))
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

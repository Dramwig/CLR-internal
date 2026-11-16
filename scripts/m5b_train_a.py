"""Pre-specified M5b-A controls, trained from scratch; train/val only."""
import argparse,copy,json,time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.m5b_runtime import Budget,snapshot
from clr.models.m5b_controls import PrefixProbe,current_only,prefix_features
from clr.models.clr_v1 import make_model
from clr.models.clr_v0 import Serializer
from clr.baselines.data_m23 import Questions
from clr.data.m4.reader import PrefixBank,rows_for,targets
from clr.data.questions.core import seed as data_seed


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();budget=Budget(run,c,start);status='failed'
    try:
        torch.set_num_threads(2);seed_everything(c['seed']);device=c['device'];smoke=c['stage']=='smoke';budget.check()
        snapshot(run,c,['scripts/m5b_train_a.py','clr/models/m5b_controls.py','clr/models/clr_v1.py','clr/data/m4/reader.py',c['templates']])
        if not smoke:
            gate=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if gate.get('status')!='completed' or not gate.get('gate_passed'):raise ValueError('Passed M5b train smoke required')
        norm=np.load(Path(c['m3_baseline_run'])/'normalization.npz');mean=torch.tensor(norm['mean'],device=device);scale=torch.tensor(norm['scale'],device=device)
        templates=load_config(c['templates']);qtypes=['Q1','Q2','Q3','Q4'];groups={};banks={};datasets={}
        for split,phase in [('reader_train','train'),('reader_val','validate')]:
            rs=rows_for(c,split,phase)
            ids=set(sorted({r['problem_id'] for r in rs})[:20]) if smoke else None
            if ids:rs=[r for r in rs if r['problem_id'] in ids]
            groups[split]={q:[r for r in rs if r['qtype']==q] for q in qtypes}
            banks[split]=PrefixBank(c,split,phase,ids)
        write_json(run.path/'training_manifest.json',{'case':c['case'],'counts':{s:{q:len(r) for q,r in g.items()} for s,g in groups.items()},'test_opened':False,'B3_prefix':'concat(mean(normalized z0..zt), normalized zt)','single':'one observed state; learned position 0; no other prefix states'})
        history=[]
        if c['case']=='B3-prefix':
            for split,bank in banks.items():
                cache_device=c.get('feature_cache_device','cpu')
                bank.z=prefix_features(bank,mean.to(cache_device),scale.to(cache_device),cache_device)
                datasets[split]={q:Questions(Path(c['qa_root'])/f'{split}.parquet',q,bank,templates,problem_ids=set(r['problem_id'] for r in groups[split][q]) if smoke else None) for q in qtypes}
            # Same per-type eight-epoch / two-LR / batch4096 protocol as frozen B3.
            from scripts.m23_baselines import batches,predict
            pc={**c,'baseline_device':device,'batch_size':4096}
            for q in qtypes:
                tr=datasets['reader_train'][q];va=datasets['reader_val'][q];best=float('inf');best_state=None
                for lr in c['b3_learning_rates']:
                    seed_everything(c['seed']);model=PrefixProbe(q).to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
                    for epoch in range(1,(1 if smoke else 8)+1):
                        model.train();rng=np.random.default_rng(data_seed(c['seed'],epoch-1,'order'))
                        for _,(z,qq,o,y) in batches(tr,banks['reader_train'],device,4096,rng.permutation(len(tr.rows))):
                            loss=-(y*model(z,qq,o).log_softmax(-1)).sum(-1).mean();opt.zero_grad();loss.backward();opt.step()
                        logits=predict(model,va,banks['reader_val'],pc)
                        vl=float(-(va.y*torch.from_numpy(logits).log_softmax(-1).numpy()).sum(1).mean())
                        row={'qtype':q,'lr':lr,'epoch':epoch,'val_nll':vl,'gpu_hours':(time.monotonic()-start)/3600};history.append(row);budget.check()
                        if vl<best:best=vl;best_state=copy.deepcopy(model.state_dict());write_json(run.path/f'{q}_best.json',row)
                torch.save(best_state,run.path/f'{q}_best.pt');write_json(run.path/'history.json',history);print(q,best,flush=True)
        elif c['case']=='CLR-lite-single':
            model=make_model({**c,'variant':'lite'}).to(device);model.state_mean.copy_(mean);model.state_scale.copy_(scale)
            s=Serializer(AutoTokenizer.from_pretrained(c['m41_model_path'],local_files_only=True),templates)
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);rng=np.random.default_rng(c['seed']);tr=groups['reader_train'];va=groups['reader_val']
            total=sum(map(len,tr.values()));weights={q:total/(4*len(rs)) for q,rs in tr.items()};best=float('inf');bad=0;steps=0
            for epoch in range(1,(1 if smoke else 10)+1):
                schedule=[]
                for q,rs in tr.items():
                    ids=rng.permutation(len(rs));schedule.extend((q,ids[i:i+1024]) for i in range(0,len(rs),1024))
                rng.shuffle(schedule);model.train()
                for bi,(q,ii) in enumerate(schedule):
                    rs=[tr[q][i] for i in ii];z,m=current_only(*banks['reader_train'].batch(rs,device));opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(s.batch(rs,device),z,m,q);loss=-(targets(rs,device)*logits.log_softmax(-1)).sum(-1).mean()*weights[q]
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
                    if bi%100==0:
                        budget.check();write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'batches':len(schedule),'gpu_hours':(time.monotonic()-start)/3600})
                model.eval();vl={}
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for q,rs in va.items():
                        loss_sum=0.
                        for i in range(0,len(rs),1024):
                            rr=rs[i:i+1024];z,m=current_only(*banks['reader_val'].batch(rr,device));logits=model(s.batch(rr,device),z,m,q).float();loss_sum+=float(-(targets(rr,device)*logits.log_softmax(-1)).sum())
                        vl[q]=loss_sum/len(rs);budget.check()
                score=float(np.mean(list(vl.values())));row={'epoch':epoch,'val_nll':vl,'selection_nll':score,'gpu_hours':(time.monotonic()-start)/3600};history.append(row)
                if smoke or epoch>=8:
                    if score<best:best=score;bad=0;torch.save(model.trainable_state(),run.path/'best.pt');write_json(run.path/'best.json',row)
                    else:bad+=1
                write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
                if bad>=1:break
        else:raise ValueError('Unknown control')
        status='completed';run.write_metrics(status=status,gate_passed=True,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=False)
        write_json(Path(c['output'])/f"trained_{c['case']}_{c['seed']}_{c['stage']}.json",{'run':str(run.path)})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,test_opened=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

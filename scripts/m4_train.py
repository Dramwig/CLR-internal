"""Config-driven M4 training; validation selection never opens test/cal files."""
import argparse,json,time,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.models.clr_v0 import CLR,Serializer
from clr.baselines.b3_multi_v0 import B3Multi
from clr.baselines.probes import Probe
from clr.data.m4.reader import PrefixBank,rows_for,targets,probe_inputs

def forward(model,kind,rows,bank,serializer,templates,norm,device):
    prefix,mask=bank.batch(rows,device)
    if kind=='clr':return model(serializer.batch(rows,device),prefix,mask,rows[0]['qtype']).float()
    inputs=probe_inputs(rows,prefix,mask,templates,*norm)
    return (model(*inputs,rows[0]['qtype']) if kind=='b3_multi' else model(*inputs)).float()

def evaluate(model,kind,groups,bank,serializer,templates,norm,device,batch):
    model.eval();out={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=kind=='clr'):
        for q,rows in groups.items():
            loss=0.
            for i in range(0,len(rows),batch):
                rs=rows[i:i+batch];logits=forward(model,kind,rs,bank,serializer,templates,norm,device)
                loss+=float(-(targets(rs,device)*logits.log_softmax(-1)).sum())
            out[q]=loss/len(rows)
    return out

def main(c):
 with recorded(c) as run:
    start=time.monotonic();status='failed';torch.set_num_threads(2);seed_everything(c['seed']);device=c['reader_device'];kind=c['model_kind']
    try:
        smoke=json.loads((Path(c['smoke_run'])/'metrics.json').read_text())
        if smoke.get('status')!='completed':raise ValueError('Passed reader smoke required')
        if torch.cuda.mem_get_info(device)[0]/1024**3<c['reader_min_free_gib']:raise RuntimeError('Insufficient GPU memory; no fallback')
        templates=load_config(c['templates']);qtypes=c['qtypes'];tr=rows_for(c,'reader_train','train',qtypes=qtypes)
        if c['fewshot_n']:
            chosen=json.loads(Path(c['fewshot_manifest']).read_text())['panels'][str(c['fewshot_n'])]
            keys={tuple(x) for x in chosen};tr=[r for r in tr if (r['problem_id'],r['state_ref']['t'],r['label']) in keys]
            if len(tr)!=c['fewshot_n']:raise ValueError(f'Fewshot panel mismatch: {len(tr)}')
        banks={'reader_train':PrefixBank(c,'reader_train','train',{r['problem_id'] for r in tr})}
        for r in tr:r['_split']='reader_train'
        if c['augmentation']:
            extra=rows_for(c,'unseen_train','train',qtypes=qtypes)
            for r in extra:r['_split']='unseen_train'
            banks['unseen_train']=PrefixBank(c,'unseen_train','train');tr+=extra
        groups={q:[r for r in tr if r['qtype']==q] for q in qtypes};del tr
        vr=rows_for(c,'reader_val','validate',qtypes=qtypes);vg={q:[r for r in vr if r['qtype']==q] for q in qtypes};del vr
        if c['validation_limit']:vg={q:rs[:c['validation_limit']] for q,rs in vg.items()}
        vb=PrefixBank(c,'reader_val','validate')
        n=np.load(Path(c['m3_baseline_run'])/'normalization.npz');norm=[torch.from_numpy(n[k]).to(device) for k in ['mean','scale']]
        serializer=None
        if kind=='clr':
            serializer=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates,c['max_length'])
            model=CLR.pretrained(c['model_path'],k=c['cross_every']).to(device)
            model.state_mean.copy_(norm[0]);model.state_scale.copy_(norm[1])
            if c['initial_checkpoint']:model.restore_trainable(torch.load(c['initial_checkpoint'],map_location='cpu',weights_only=True))
            enc=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('backbone.')]
            head=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('backbone.')]
            opt=torch.optim.AdamW([{'params':enc,'lr':c['encoder_lr']},{'params':head,'lr':c['head_lr']}],weight_decay=c['weight_decay'],foreach=False)
        else:
            model=(B3Multi() if kind=='b3_multi' else Probe('B3','Q4')).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=c['head_lr'],weight_decay=c['weight_decay'],foreach=False)
        rng=np.random.default_rng(c['seed']);best=float('inf');bad=0;history=[];examples=0
        epochs=c['trial_epochs'] if c['trial'] else c['epochs'];pertype=c['trial_samples_per_type'] if c['trial'] else c['samples_per_type']
        batch=c['micro_batch'];accum=c['gradient_accumulation'];valepochs=[epochs] if c['trial'] else c['validation_epochs']
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0.;steps=0;opt.zero_grad(set_to_none=True)
            # Full small-N passes; otherwise equal per-qtype sampling with replacement.
            schedules={q:rng.permutation(len(rs)).tolist() if c['fewshot_n'] else rng.integers(len(rs),size=pertype).tolist() for q,rs in groups.items()}
            batches=[]
            for q in qtypes:
                ids=schedules[q]
                for i in range(0,len(ids),batch):batches.append((q,ids[i:i+batch]))
            rng.shuffle(batches)
            for bi,(q,ids) in enumerate(batches):
                rows=[groups[q][i] for i in ids];total=len(rows);parts=defaultdict(list)
                for r in rows:parts[r['_split']].append(r)
                # Each augmented microbatch shares one gradient over its original rows.
                loss=0.
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=kind=='clr'):
                    for split,rs in parts.items():
                        logits=forward(model,kind,rs,banks[split],serializer,templates,norm,device)
                        loss=loss-(targets(rs,device)*logits.log_softmax(-1)).sum()/total
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                window=min(accum,len(batches)-(bi//accum)*accum)
                (loss/window).backward();loss_sum+=float(loss)*total;examples+=total;steps+=total
                if (bi+1)%accum==0 or bi+1==len(batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
                if (bi+1)%100==0:
                    write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi+1,'batches':len(batches),'examples':examples,'gpu_hours':(time.monotonic()-start)/3600})
                    if (time.monotonic()-start)/3600>c['run_budget_hours']:raise TimeoutError('Configured per-run GPU budget exhausted')
            row={'epoch':epoch,'train_nll':loss_sum/steps,'examples':examples,'gpu_hours':(time.monotonic()-start)/3600}
            if epoch in valepochs:
                val=evaluate(model,kind,vg,vb,serializer,templates,norm,device,c['eval_batch']);score=float(np.mean(list(val.values())));row.update(val_nll=val,selection_nll=score)
                if score<best:
                    best=score;bad=0;state=model.trainable_state() if kind=='clr' else {k:v.cpu() for k,v in model.state_dict().items()}
                    temp=run.path/'best.tmp';torch.save(state,temp);temp.replace(run.path/'best.pt');write_json(run.path/'best.json',{'epoch':epoch,'selection_nll':score,'val_nll':val})
                else:bad+=1
            history.append(row);write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
            if bad>=c['patience']:break
        status='completed';write_json(run.path/'training_manifest.json',{'train_counts':{q:len(rs) for q,rs in groups.items()},'val_counts':{q:len(rs) for q,rs in vg.items()},'test_opened':False,'model_kind':kind,'case':c['case'],'cross_every':c['cross_every']})
        run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,best_val_nll=best,examples=examples,epochs=len(history),peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        print(str(run.path),flush=True)
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

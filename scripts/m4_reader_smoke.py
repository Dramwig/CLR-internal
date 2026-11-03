"""Small train subset and reader_val proxy only; profile full CLR before budgeting."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.models.clr_v0 import CLR,Serializer,QTYPES
from clr.data.m4.reader import PrefixBank,rows_for,targets
from clr.data.questions.core import seed

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);seed_everything(c['seed']);device=c['reader_device']
    if torch.cuda.mem_get_info(device)[0]/1024**3<c['reader_min_free_gib']:raise RuntimeError('Insufficient free reader GPU memory; no fallback')
    all_train=rows_for(c,'reader_train','train');all_val=rows_for(c,'reader_val','validate')
    trainids=set(sorted({r['problem_id'] for r in all_train},key=lambda p:seed(11,p,'reader-smoke'))[:50]);valids=set(sorted({r['problem_id'] for r in all_val},key=lambda p:seed(11,p,'reader-smoke'))[:50])
    tr={q:[r for r in all_train if r['problem_id'] in trainids and r['qtype']==q] for q in QTYPES};va={q:[r for r in all_val if r['problem_id'] in valids and r['qtype']==q][:128] for q in QTYPES};del all_train,all_val
    banks={'reader_train':PrefixBank(c,'reader_train','train',trainids),'reader_val':PrefixBank(c,'reader_val','validate',valids)}
    tokenizer=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True)
    serializer=Serializer(tokenizer,load_config(c['templates']),c['max_length'])
    model=CLR.pretrained(c['model_path'],k=c['cross_every'],gradient_checkpointing=True).to(device)
    norm=np.load(Path(c['m3_baseline_run'])/'normalization.npz');model.state_mean.copy_(torch.from_numpy(norm['mean']).to(device));model.state_scale.copy_(torch.from_numpy(norm['scale']).to(device))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['encoder_lr'],foreach=False)
    times=[];losses=[];model.train();rng=np.random.default_rng(c['seed'])
    for step in range(c['smoke_steps']):
        tick=time.monotonic();q=QTYPES[step%4];rows=[tr[q][i] for i in rng.integers(len(tr[q]),size=c['micro_batch'])]
        prefix,mask=banks['reader_train'].batch(rows,device);text=serializer.batch(rows,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(text,prefix,mask,q);loss=-(targets(rows,device)*logits.float().log_softmax(-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize(device)
        if not torch.isfinite(loss):raise ValueError('Nonfinite smoke training loss')
        times.append(time.monotonic()-tick);losses.append(float(loss))
        if time.monotonic()-start>c['smoke_seconds']:raise TimeoutError('Reader smoke exceeds15min')
    model.eval();val_loss={};tick=time.monotonic()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for q,rows in va.items():
            total=0.
            for i in range(0,len(rows),c['micro_batch']):
                batch=rows[i:i+c['micro_batch']];prefix,mask=banks['reader_val'].batch(batch,device);logits=model(serializer.batch(batch,device),prefix,mask,q).float();total+=float(-(targets(batch,device)*logits.log_softmax(-1)).sum())
            val_loss[q]=total/len(rows)
        rows=va['Q3'][:c['micro_batch']];perm=[2,0,3,1];other=[{**r,'options':[r['options'][i] for i in perm]} for r in rows];prefix,mask=banks['reader_val'].batch(rows,device)
        a=model(serializer.batch(rows,device),prefix,mask,'Q3');b=model(serializer.batch(other,device),prefix,mask,'Q3');delta=float((a[:,perm]-b).abs().max())
        if delta!=0:raise ValueError('Canonical option permutation not bitwise invariant')
    torch.cuda.synchronize(device);eval_seconds=time.monotonic()-tick
    write_json(run.path/'smoke_panels.json',{'train_ids':sorted(trainids),'val_ids':sorted(valids),'test_opened':False})
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=(time.monotonic()-start)/3600,training_examples_per_second=c['micro_batch']/float(np.mean(times[4:])),evaluation_examples_per_second=sum(map(len,va.values()))/eval_seconds,step_seconds=times,val_nll=val_loss,option_permutation_max_abs=delta,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

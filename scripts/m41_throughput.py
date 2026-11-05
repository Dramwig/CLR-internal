"""Val-only, bounded candidate throughput; reports feasibility before full training."""
import argparse,time,json,gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m41_budget import Budget
from clr.models.clr_v1 import make_model
from clr.models.clr_v0 import Serializer
from clr.data.m4.reader import PrefixBank,rows_for,targets

def main(c):
 with recorded(c) as run:
    start=time.monotonic();budget=Budget(run,c,start);torch.set_num_threads(2);seed_everything(c['seed']);device=c['reader_device']
    try:
        templates=load_config(c['templates']);s=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates)
        rs=rows_for(c,'reader_val','validate');rng=np.random.default_rng(c['seed']);groups={}
        for q in ['Q1','Q2','Q3','Q4']:
            pool=[r for r in rs if r['qtype']==q];ii=rng.choice(len(pool),c['batch_size'],replace=False);groups[q]=[pool[i] for i in ii]
        bank=PrefixBank(c,'reader_val','validate',{r['problem_id'] for rows in groups.values() for r in rows});del rs
        m=make_model(c).to(device);n=np.load(Path(c['m3_baseline_run'])/'normalization.npz');m.state_mean.copy_(torch.from_numpy(n['mean']).to(device));m.state_scale.copy_(torch.from_numpy(n['scale']).to(device))
        opt=torch.optim.AdamW([{'params':[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('backbone.')],'lr':c['encoder_lr']},{'params':[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('backbone.')],'lr':c['head_lr']}],weight_decay=.01)
        out={}
        for q,rows in groups.items():
            text=s.batch(rows,device)
            if c.get('stress_pad_to'):
                extra=c['stress_pad_to']-text['input_ids'].shape[1]
                if extra<0:raise ValueError('Stress padding length too small')
                for name,value in [('input_ids',s.tokenizer.pad_token_id),('blocks',-1),('position_ids',0)]:text[name]=torch.nn.functional.pad(text[name],(0,extra),value=value)
                text['numeric']=torch.nn.functional.pad(text['numeric'],(0,0,0,extra))
            z,mask=bank.batch(rows,device);y=targets(rows,device);times=[]
            for step in range(c['benchmark_steps']):
                budget.check();torch.cuda.synchronize(device);t=time.monotonic();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(text,z,mask,q);loss=-(y*logits.float().log_softmax(-1)).sum(-1).mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();torch.cuda.synchronize(device)
                if step>=2:times.append(time.monotonic()-t)
            out[q]={'examples_per_second':len(rows)/np.mean(times),'sequence_length':text['input_ids'].shape[1],'seconds_per_step':float(np.mean(times))};write_json(run.path/'throughput.json',out)
        run.write_metrics(status='completed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=False,throughput=out,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        write_json(Path(c['output'])/f"throughput_{c['variant']}.json",{'run':str(run.path),**out})
    except Exception:
        run.write_metrics(status='failed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

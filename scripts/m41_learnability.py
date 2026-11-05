"""Diagnostic training on val300 or synthetic data. No final test access."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m41_budget import Budget
from clr.models.clr_v0 import CLR,Serializer
from clr.baselines.probes import Probe
from clr.data.m4.reader import PrefixBank,rows_for,targets,probe_inputs

def synthetic(n,seed,direction,q,noise=.08,domain=16):
    rng=np.random.default_rng(seed);values=rng.integers(1,domain+1,n);z=(values[:,None]-(domain+1)/2)/(domain/3.47)*direction[None]+rng.normal(0,noise,(n,768));rows=[]
    for i,v in enumerate(values):
        if q=='Q3':
            opts=[int(v)]+rng.choice([j for j in range(1,domain+1) if j!=v],3,replace=False).tolist();rng.shuffle(opts);label=opts.index(int(v))
        else:
            label=i%2;opts=[int(v) if label else int(rng.choice([j for j in range(1,domain+1) if j!=v]))]
        rows.append({'problem_id':f'synthetic:{seed}:{i}','qtype':q,'template_id':0,'options':list(map(str,opts)),'label':label,'label_soft':float(label),'state_ref':{'traj_id':-1,'t':0}})
    return rows,torch.tensor(z,dtype=torch.float32)[:,None],torch.ones(n,1,dtype=torch.bool)

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);seed_everything(c['seed']);budget=Budget(run,c,start);device=c['reader_device'];q=c['qtype'];kind=c['variant'];templates=load_config(c['templates'])
    try:
        if c['task']=='synthetic':
            direction=np.random.default_rng(917).normal(size=768).astype('float32');direction/=np.sqrt(np.mean(direction**2));direction*=c.get('synthetic_scale',1.)
            rows,z,mask=synthetic(c['samples'],c['seed'],direction,q,c.get('synthetic_noise',.08),c.get('synthetic_domain',16));vr,vz,vm=synthetic(c['validation_samples'],c['seed']+1000,direction,q,c.get('synthetic_noise',.08),c.get('synthetic_domain',16));norm=(torch.zeros(768,device=device),torch.ones(768,device=device))
        else:
            pool=rows_for(c,'val','validate',qtypes=['Q4']);rng=np.random.default_rng(c['seed']);ii=rng.choice(len(pool),c['samples'],replace=False);rows=[pool[i] for i in ii]
            bank=PrefixBank(c,'val','validate',{r['problem_id'] for r in rows});z,mask=bank.batch(rows,'cpu');vr=rows;vz=z;vm=mask
            n=np.load(Path(c['m3_baseline_run'])/'normalization.npz');norm=tuple(torch.from_numpy(n[k]).to(device) for k in ['mean','scale'])
        write_json(run.path/'panel.json',{'train':rows,'validation':vr,'diagnostic_only':True,'test_opened':False})
        serializer=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates)
        if kind=='v0':model=CLR.pretrained(c['model_path'],k=4,gradient_checkpointing=False)
        elif kind in ['v1','lite']:
            from clr.models.clr_v1 import make_model
            model=make_model(c)
        else:model=Probe('B3',q)
        model=model.to(device)
        if c.get('initial_checkpoint'):
            state=torch.load(c['initial_checkpoint'],map_location='cpu',weights_only=True)
            if kind=='B3':model.load_state_dict(state)
            else:model.restore_trainable(state)
        if kind!='B3':model.state_mean.copy_(norm[0]);model.state_scale.copy_(norm[1])
        enc=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')];new=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
        opt=torch.optim.AdamW([{'params':enc,'lr':c['encoder_lr']},{'params':new,'lr':c['head_lr']}],weight_decay=c['weight_decay'])
        def forward(rs,zz,mm):
            zz=zz.to(device);mm=mm.to(device)
            if kind=='B3':return model(*probe_inputs(rs,zz,mm,templates,*norm))
            return model(serializer.batch(rs,device),zz,mm,q).float()
        def score(rs,zz,mm):
            model.eval();loss=correct=0
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['amp'] and kind!='B3'):
                for i in range(0,len(rs),c['batch_size']):
                    rr=rs[i:i+c['batch_size']];out=forward(rr,zz[i:i+len(rr)],mm[i:i+len(rr)]);y=targets(rr,device)
                    loss+=float(-(y*out.log_softmax(-1)).sum());correct+=int((out.argmax(-1)==y.argmax(-1)).sum())
            return {'nll':loss/len(rs),'accuracy':correct/len(rs)}
        rng=np.random.default_rng(c['seed']);history=[];steps=0;best=0.;traces=[]
        for epoch in range(c['epochs']):
            budget.check();model.train()
            order=rng.permutation(len(rows));total=0.
            for i in range(0,len(rows),c['batch_size']):
                ids=order[i:i+c['batch_size']];rs=[rows[j] for j in ids];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['amp'] and kind!='B3'):
                    out=forward(rs,z[ids],mask[ids]);loss=-(targets(rs,device)*out.log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite diagnostic loss')
                loss.backward()
                if steps in [0,1,10,100,500,1000] and kind!='B3':
                    from scripts.m41_diagnose import module_stats
                    traces.append({'step':steps,'modules':module_stats(model)});write_json(run.path/'module_trace.json',traces)
                torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip']);opt.step();steps+=1;total+=float(loss)*len(rs)
            if (epoch+1)%c['eval_every']==0 or epoch==0:
                train=score(rows,z,mask);val=score(vr,vz,vm);row={'epoch':epoch+1,'steps':steps,'train':train,'validation':val,'gpu_hours':(time.monotonic()-start)/3600};history.append(row);write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
                if val['accuracy']>best:best=val['accuracy'];torch.save(model.trainable_state() if kind!='B3' else model.state_dict(),run.path/'best.pt')
                if train['accuracy']>=c['threshold'] and val['accuracy']>=c['threshold']:break
        passed=history[-1]['train']['accuracy']>=c['threshold'] and history[-1]['validation']['accuracy']>=c['threshold']
        run.write_metrics(status='completed',passed=passed,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,steps=steps,epochs=epoch+1,train=history[-1]['train'],validation=history[-1]['validation'],test_opened=False)
        outdir=Path(c['output']);outdir.mkdir(exist_ok=True);write_json(outdir/f"gate_{kind}_{c['task']}_{q}_{c['seed']}.json",{'run':str(run.path),'passed':passed})
    except Exception:
        run.write_metrics(status='failed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

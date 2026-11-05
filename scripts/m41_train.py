"""Full-exposure M4.1 joint training, gated by learnability, no test access.

One epoch visits every QA instance exactly once; inverse-qtype-frequency loss
weights implement a uniform qtype objective without oversampling short types.
Best checkpoint eligibility starts only after the required B3 exposure floor.
"""
import argparse,json,time,math,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m41_budget import Budget
from clr.models.clr_v0 import Serializer
from clr.models.clr_v1 import make_model
from clr.baselines.b3_multi_v0 import B3Multi
from clr.data.m4.reader import PrefixBank,rows_for,targets
from scripts.m4_train import evaluate,forward

def main(c):
 with recorded(c) as run:
    start=time.monotonic();budget=Budget(run,c,start);torch.set_num_threads(2);seed_everything(c['seed']);device=c['reader_device']
    try:
        gate=json.loads(Path(c['gate_lock']).read_text())
        if not gate.get('both_variants_passed'):raise ValueError('Both candidates must pass A4/A5 before C')
        if gate['model_sha256']!=hashlib.sha256(Path('clr/models/clr_v1.py').read_bytes()).hexdigest():raise ValueError('Candidate code changed after gates')
        templates=load_config(c['templates']);qtypes=c['qtypes'];tr=rows_for(c,'reader_train','train',qtypes=qtypes)
        if c.get('mode')=='smoke':
            ids=set(sorted({r['problem_id'] for r in tr})[:20]);tr=[r for r in tr if r['problem_id'] in ids]
        if c.get('fewshot_n'):
            panels=json.loads(Path(c['fewshot_manifest']).read_text())['panels'][str(c['fewshot_n'])];keys={tuple(x) for x in panels};tr=[r for r in tr if (r['problem_id'],r['state_ref']['t'],r['label']) in keys]
            if len(tr)!=c['fewshot_n']:raise ValueError('Frozen fewshot panel mismatch')
        groups={q:[r for r in tr if r['qtype']==q] for q in qtypes};bank=PrefixBank(c,'reader_train','train',{r['problem_id'] for r in tr});del tr
        vr=rows_for(c,'reader_val','validate',qtypes=qtypes)
        if c.get('mode')=='smoke':
            vids=set(sorted({r['problem_id'] for r in vr})[:20]);vr=[r for r in vr if r['problem_id'] in vids]
        vg={q:[r for r in vr if r['qtype']==q] for q in qtypes};vb=PrefixBank(c,'reader_val','validate',{r['problem_id'] for r in vr});del vr
        kind='b3_multi' if c['variant']=='b3_multi' else 'clr';model=(B3Multi() if kind=='b3_multi' else make_model(c)).to(device);norm=np.load(Path(c['m3_baseline_run'])/'normalization.npz');norm=[torch.from_numpy(norm[k]).to(device) for k in ['mean','scale']]
        if kind=='clr':model.state_mean.copy_(norm[0]);model.state_scale.copy_(norm[1])
        s=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates) if kind=='clr' else None
        if c.get('initial_checkpoint'):model.restore_trainable(torch.load(c['initial_checkpoint'],map_location='cpu',weights_only=True))
        opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')],'lr':c['encoder_lr']},{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')],'lr':c['head_lr']}],weight_decay=c['weight_decay'])
        total=sum(map(len,groups.values()));weights={q:total/(len(qtypes)*len(rs)) for q,rs in groups.items()};rng=np.random.default_rng(c['seed']);history=[];best=float('inf');bad=0;examples={q:0 for q in qtypes};steps=0
        write_json(run.path/'training_manifest.json',{'train_counts':{q:len(rs) for q,rs in groups.items()},'objective_weights':weights,'all_instances_per_epoch':True,'checkpoint_min_epoch':c['min_epochs'],'test_opened':False})
        for epoch in range(1,c['epochs']+1):
            model.train();batches=[]
            for q,rows in groups.items():
                order=rng.permutation(len(rows));batches.extend((q,order[i:i+c['batch_size']]) for i in range(0,len(rows),c['batch_size']))
            rng.shuffle(batches);loss_sum={q:0. for q in qtypes}
            for bi,(q,ii) in enumerate(batches):
                rs=[groups[q][i] for i in ii];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=kind=='clr'):
                    logits=forward(model,kind,rs,bank,s,templates,norm,device);loss=-(targets(rs,device)*logits.log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite M4.1 training loss')
                (loss*weights[q]).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1;examples[q]+=len(rs);loss_sum[q]+=float(loss)*len(rs)
                if bi%100==0:
                    budget.check();write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'batches':len(batches),'examples':examples,'optimizer_steps':steps,'gpu_hours':(time.monotonic()-start)/3600})
            val=evaluate(model,kind,vg,vb,s,templates,norm,device,c['eval_batch'],budget);score=float(np.mean(list(val.values())));row={'epoch':epoch,'train_nll':{q:loss_sum[q]/len(groups[q]) for q in qtypes},'val_nll':val,'selection_nll':score,'examples':dict(examples),'steps':steps,'gpu_hours':(time.monotonic()-start)/3600}
            if epoch>=c['min_epochs']:
                if score<best:
                    best=score;bad=0;torch.save(model.trainable_state() if kind=='clr' else {k:v.detach().cpu() for k,v in model.state_dict().items()},run.path/'best.tmp');(run.path/'best.tmp').replace(run.path/'best.pt');write_json(run.path/'best.json',row)
                else:bad+=1
            history.append(row);write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
            if epoch>=c['min_epochs'] and bad>=c['patience']:break
        run.write_metrics(status='completed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=False,best_val_nll=best,epochs=epoch,examples=examples,optimizer_steps=steps)
        write_json(Path(c['output'])/f"trained_{c['case']}_{c['variant']}_{c['seed']}.json",{'run':str(run.path)})
    except Exception:
        run.write_metrics(status='failed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

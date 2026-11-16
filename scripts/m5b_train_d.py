"""M5b-D: original Q1 Bernoulli CE plus sibling CE and Delta-V weighted ranking."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m5b_runtime import Budget,snapshot
from clr.models.clr_v0 import Serializer
from clr.models.clr_v1 import make_model
from clr.data.m4.reader import PrefixBank,rows_for,targets
from clr.calib.m4 import fit


def ranking_loss(logits,truth):
    difference=truth[:,:,None]-truth[:,None,:]
    margin=logits[:,:,None]-logits[:,None,:]
    weight=difference.abs();return (torch.nn.functional.softplus(-difference.sign()*margin)*weight).sum()/weight.sum().clamp_min(1.)


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();budget=Budget(run,c,start);status='failed'
    try:
        torch.set_num_threads(2);seed_everything(c['seed']);device=c['device'];smoke=c['stage']=='smoke';budget.check()
        snapshot(run,c,['scripts/m5b_train_d.py','clr/search/m5b.py','clr/models/clr_v1.py','clr/data/m4/reader.py'])
        if not smoke:
            gate=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if gate.get('status')!='completed' or not gate.get('gate_passed'):raise ValueError('D training smoke required')
        items=[]
        for f in sorted(Path(c['generation_run']).glob('siblings_*.pt')):items.extend(torch.load(f,map_location='cpu',weights_only=True))
        if not items:raise ValueError('No D siblings')
        if smoke:items=items[:20]
        histories=torch.stack([r['history'] for r in items]);outcomes=torch.stack([r['outcomes'] for r in items]);lengths=torch.tensor([r['length'] for r in items]);train=np.array([i for i,r in enumerate(items) if r['panel']=='train']);val=np.array([i for i,r in enumerate(items) if r['panel']=='validation'])
        if not smoke:
            train_ids={items[i]['problem_id'] for i in train};val_ids={items[i]['problem_id'] for i in val}
            if train_ids&val_ids or len(train_ids)!=3000 or len(train)!=6000:raise ValueError('D train/validation group protocol mismatch')
        else:val=train[:min(4,len(train))]
        templates=load_config(c['templates']);serializer=Serializer(AutoTokenizer.from_pretrained(c['m41_model_path'],local_files_only=True),templates)
        model=make_model({**c,'variant':'lite'}).to(device);model.restore_trainable(torch.load(Path(c['m41_runs']['lite'][str(c['seed'])])/'best.pt',map_location='cpu',weights_only=True))
        original=rows_for(c,'reader_train','train',qtypes=['Q1'])
        if not smoke:original=[r for r in original if r['problem_id'] not in val_ids]
        bank=PrefixBank(c,'reader_train','train',set(r['problem_id'] for r in original) if not smoke else set(r['problem_id'] for r in original[:1000]))
        if smoke:original=[r for r in original if r['problem_id'] in {k[0] for k in bank.index}]
        validation=rows_for(c,'reader_val','validate',qtypes=['Q1'])
        vids=sorted({r['problem_id'] for r in validation})
        if smoke:vids=vids[:20]
        cal_ids=set(vids[:max(5,len(vids)//4)]);select_ids=set(vids)-cal_ids
        validation=[r for r in validation if r['problem_id'] in set(vids)]
        vbank=PrefixBank(c,'reader_val','validate',set(vids))
        rng=np.random.default_rng(c['seed']);opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=.0001);row={'qtype':'Q1','template_id':0,'options':['0']};history=[];best=float('inf')
        def batch_logits(ii):
            z=histories[ii].reshape(-1,7,768).to(device);ll=lengths[ii].repeat_interleave(8).to(device);m=torch.arange(7,device=device)[None]<ll[:,None]
            return model(serializer.batch([row]*len(z),device),z,m,'Q1')[:,1].reshape(-1,8)
        for epoch in range(1,(1 if smoke else c['epochs'])+1):
            model.train();order=rng.permutation(train)
            for bi,start1 in enumerate(range(0,len(order),c['group_batch'])):
                ii=order[start1:start1+c['group_batch']];rs=[original[i] for i in rng.integers(len(original),size=len(ii)*8)];opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=batch_logits(ii);y=outcomes[ii].to(device);bce=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,:,None].expand_as(y),y);rank=ranking_loss(logits,y.mean(-1))
                    z,m=bank.batch(rs,device);o=model(serializer.batch(rs,device),z,m,'Q1');orig=-(targets(rs,device)*o.log_softmax(-1)).sum(-1).mean()
                    loss=.5*(bce+orig)+c['rank_weight']*rank
                if not torch.isfinite(loss):raise ValueError('Nonfinite D loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if bi%20==0:budget.check();write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'loss':float(loss),'bce':float(bce),'original_bce':float(orig),'rank':float(rank),'gpu_hours':(time.monotonic()-start)/3600})
                if smoke and bi>=3:break
            model.eval();losses=[];cal_logits=[];cal_y=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start1 in range(0,len(val),c['group_batch']):
                    ii=val[start1:start1+c['group_batch']];lo=batch_logits(ii);y=outcomes[ii].to(device);bce=torch.nn.functional.binary_cross_entropy_with_logits(lo[:,:,None].expand_as(y),y);rank=ranking_loss(lo,y.mean(-1));losses.append((float(bce+c['rank_weight']*rank),len(ii)))
                    cal_logits.extend(torch.stack([torch.zeros_like(lo),lo],-1).reshape(-1,2).float().cpu().numpy());cal_y.extend(y.mean(-1).flatten().cpu().numpy());budget.check()
            monitor=sum(x*n for x,n in losses)/sum(n for _,n in losses)
            vloss=0.;vn=0;temp_logits=[];temp_targets=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for first in range(0,len(validation),512):
                    rs=validation[first:first+512];z,m=vbank.batch(rs,device);lo=model(serializer.batch(rs,device),z,m,'Q1').float();yy=targets(rs,device)
                    select=torch.tensor([r['problem_id'] in select_ids for r in rs],device=device)
                    if select.any():vloss+=float(-(yy[select]*lo[select].log_softmax(-1)).sum());vn+=int(select.sum())
                    cal=~select
                    if cal.any():temp_logits.extend(lo[cal].cpu().numpy());temp_targets.extend(yy[cal].cpu().numpy())
                    budget.check()
            score=vloss/vn;entry={'epoch':epoch,'training_side_sibling_monitor':monitor,'reader_val_selection_nll':score,'gpu_hours':(time.monotonic()-start)/3600};history.append(entry)
            if score<best:
                best=score;torch.save(model.trainable_state(),run.path/'best.pt');write_json(run.path/'best.json',entry)
                cal=fit(np.asarray(temp_logits),np.asarray(temp_targets),'temperature');write_json(run.path/'calibration.json',cal)
            write_json(run.path/'history.json',history);print(json.dumps(entry),flush=True)
        write_json(run.path/'training_manifest.json',{'original_q1_rows':len(original),'train_groups':len(train),'training_side_monitor_groups':len(val),'train_questions':len({items[i]['problem_id'] for i in train}),'rank_weight':c['rank_weight'],'loss':'.5*(original_Q1_Bernoulli_CE + sibling_K4_independent_Bernoulli_CE) + weighted_pairwise_softplus','selection':'reader_val Q1 NLL on fixed 750 IDs; no val300 feedback','calibration':'temperature on other fixed 250 reader_val IDs; ranking invariant','training_side_monitor':'300 reader_train questions never enter new D loss, but seen by M4.1 pretraining; not used for selection/calibration','test_opened':False})
        status='completed';run.write_metrics(status=status,gate_passed=True,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=False)
        write_json(Path(c['output'])/f"trained_D-lite_{c['seed']}_{c['stage']}.json",{'run':str(run.path)})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,test_opened=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

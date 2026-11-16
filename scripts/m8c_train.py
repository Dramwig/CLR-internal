"""Train only new source adapters; validation-only model selection."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time,os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m8b_reader import SourceBank,new_model,make_row,rng_seed
from clr.audit.m71_reader import Serializer
from clr.audit.m8c_runtime import Cost,snapshot,live
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded
from clr.utils.config import load_config

@torch.no_grad()
def evaluate(model,bank,serializer,c,epoch=0):
    model.eval();rows=bank.rows(epoch,True);out={}
    for task in ['A1','A4']:
        rr=[r for r in rows if r['task']==task];ls=[];ys=[]
        for start in range(0,len(rr),c['batch_size']):
            r=rr[start:start+c['batch_size']]
            with torch.autocast('cuda',dtype=torch.bfloat16):logit=model(serializer.batch(r,c['device']),bank.batch(r,c['device']),task)
            ls.append(logit.float().cpu());ys.extend(r['label'] for r in r)
        logits=torch.cat(ls);y=torch.tensor(ys);out[task]={'accuracy':float((logits.argmax(-1)==y).float().mean()),'nll':float(torch.nn.functional.cross_entropy(logits,y))}
    return out

def main(c):
    reference=Path(c['output'])/c.get('training_folder','training')/f'{c["design"]}_{c["seed"]}.json'
    active=reference.with_suffix('.active.json')
    if c.get('reuse_completed',False) and active.exists() and not reference.exists():
        pid=json.loads(active.read_text())['pid']
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Existing R4 worker disappeared without completed reference')
        print('Training already active',pid,reference,flush=True);return
    if reference.exists() and c.get('reuse_completed',False):
        done=json.loads(reference.read_text())
        if done['selected_exposure']<c['min_exposure'] or not Path(done['checkpoint']).exists():raise ValueError('Invalid completed R4 reference')
        print('Reused completed training',reference,flush=True);return
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            if torch.cuda.mem_get_info()[0]/2**30<c['min_free_GiB']:raise RuntimeError('Adapter GPU memory precheck failed')
            design=c['design'];root=Path(c['output']);folder=root/('smoke_source' if c['mode']=='smoke' else 'source')
            train=SourceBank(folder/'reader_train',design);val=SourceBank(folder/'reader_val',design)
            if c.get('overfit',False):val=train
            model=new_model(c['seed'],design,c['device'],warm=c.get('warm',False));mean,scale=train.normalization();model.mean.copy_(mean);model.scale.copy_(scale)
            serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            opt=torch.optim.AdamW(model.adapter.parameters(),lr=c['lr'],weight_decay=.01);rng=np.random.default_rng(c['seed']);history=[];best=float('inf');bad=0;exposure=0
            for epoch in range(c['epochs']):
                cost.check()
                if design=='R2':train.refresh(epoch+c['seed']*100)
                rows=train.rows(epoch)
                if c.get('all_negatives',False):
                    rows=[]
                    for mid,entry in train.entry.items():
                        v=train.values[mid];tid=epoch%4
                        rows.extend([make_row(entry,v,'A1',tid,v)]*3)
                        rows.extend(make_row(entry,v,'A1',tid,(v+j)%4) for j in [1,2,3])
                        rows.append(make_row(entry,v,'A4',tid,None,rng_seed(mid,epoch)))
                batches=[]
                for task in ['A1','A4']:
                    rr=[r for r in rows if r['task']==task];rng.shuffle(rr)
                    batches += [(task,rr[s:s+c['batch_size']]) for s in range(0,len(rr),c['batch_size'])]
                rng.shuffle(batches);model.train();model.body.eval();losses=[]
                for task,rr in batches:
                    opt.zero_grad(set_to_none=True);y=torch.tensor([r['label'] for r in rr],device=c['device'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):logit=model(serializer.batch(rr,c['device']),train.batch(rr,c['device']),task)
                    loss=torch.nn.functional.cross_entropy(logit.float(),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                if design=='R2':val.refresh(20261108)
                exposure+=len(rows)
                metrics=evaluate(model,val,serializer,c,0);score=sum(v['nll'] for v in metrics.values())/2
                history.append({'epoch':epoch+1,'exposure':exposure,'train_loss':float(np.mean(losses)),'val':metrics,'val_nll':score,'seconds':time.monotonic()-cost.start})
                if exposure>=c.get('min_exposure',0) and score<best:
                    best=score;bad=0;torch.save(model.state_dict(),run.path/'best.pt');chosen=epoch+1
                elif exposure>=c.get('min_exposure',0):bad+=1
                write_json(run.path/'history.json',history);cost.update(status='training',epochs=epoch+1);live(stage='train_'+design,run=str(run.path),seed=c['seed'],epoch=epoch+1)
                print(design,c['seed'],epoch+1,metrics,'seconds',round(time.monotonic()-cost.start,2),flush=True)
                if bad>=c['patience'] and exposure>=c.get('min_exposure',0):break
            dest=root/(c.get('training_folder') or ('smoke_training' if c['mode']=='smoke' else 'training'));dest.mkdir(exist_ok=True)
            write_json(dest/f'{design}_{c["seed"]}.json',{'checkpoint':str(run.path/'best.pt'),'run':str(run.path),'best_epoch':chosen,'best_val_nll':best,'history':history,'exposure':exposure,'selected_exposure':history[chosen-1]['exposure'],'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'body_frozen':True,'train_groups':len(train.entries),'val_groups':len(val.entries)})
            cost.update(status='completed');live(stage='train_'+design+'_completed',run=str(run.path))
        except BaseException:
            cost.update(status='failed');live(stage='train_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

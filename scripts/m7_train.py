"""Joint three-seed M7 training; all selection uses reader_val and seen templates."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.models.m7 import Reader
from clr.audit.m7_reader import Bank,Serializer
from clr.audit.m7_runtime import Cost,snapshot,live
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json

def state_batch(model,bank,rows,device):
    if model.kind=='B0':
        return torch.zeros(len(rows),1,2048,device=device),torch.ones(len(rows),1,device=device,dtype=torch.bool)
    return bank.batch(rows,device)

@torch.no_grad()
def evaluate(model,bank,serializer,c):
    model.eval();result={}
    for task in ['A1','A2','A3','A4']:
        rs=[r for r in bank.rows if r['task']==task];loss=0.
        for j in range(0,len(rs),c['batch_size']):
            batch=rs[j:j+c['batch_size']];z,mask=state_batch(model,bank,batch,c['device'])
            text=serializer.batch(batch,c['device'])
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(text,z,mask,task)
            labels=torch.tensor([r['label'] for r in batch],device=c['device'])
            loss+=float(torch.nn.functional.cross_entropy(logits.float(),labels,reduction='sum'))
        result[task]=loss/len(rs)
    return result

def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);live(stage='reader_training',training_run=str(run.path))
        try:
            train=Bank(c['output'],'reader_train');val=Bank(c['output'],'reader_val')
            if c['mode']=='smoke':
                ids={e['id'] for e in train.entries[:4]};train.rows=[r for r in train.rows if r['episode_id'] in ids]
                vids={e['id'] for e in val.entries[:2]};val.rows=[r for r in val.rows if r['episode_id'] in vids]
            model=Reader(c['kind']).to(c['device']);mean,scale=train.normalization();model.mean.copy_(mean);model.scale.copy_(scale)
            serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.01)
            rng=np.random.default_rng(c['seed']);best=float('inf');bad=0;history=[];steps=0
            weights={task:len(train.rows)/(4*sum(r['task']==task for r in train.rows)) for task in ['A1','A2','A3','A4']}
            for epoch in range(1,c['epochs']+1):
                model.train();batches=[]
                for task in ['A1','A2','A3','A4']:
                    rs=[r for r in train.rows if r['task']==task];order=rng.permutation(len(rs))
                    batches.extend((task,[rs[i] for i in order[j:j+c['batch_size']]]) for j in range(0,len(rs),c['batch_size']))
                rng.shuffle(batches)
                for bi,(task,rs) in enumerate(batches):
                    cost.check();z,mask=state_batch(model,train,rs,c['device']);text=serializer.batch(rs,c['device'])
                    labels=torch.tensor([r['label'] for r in rs],device=c['device'])
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(text,z,mask,task);loss=torch.nn.functional.cross_entropy(logits.float(),labels)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite M7 loss')
                    (loss*weights[task]).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();steps+=1
                    if bi%100==0:
                        write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'batches':len(batches),'steps':steps,'gpu_hours':cost.elapsed()/3600})
                        cost.update();print(epoch,bi,len(batches),float(loss),flush=True)
                scores=evaluate(model,val,serializer,c);score=float(np.mean(list(scores.values())))
                row={'epoch':epoch,'val_nll':scores,'selection_nll':score};history.append(row)
                if score<best:
                    best=score;bad=0;torch.save(model.state_dict(),run.path/'best.pt');write_json(run.path/'best.json',row)
                else:bad+=1
                write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
                if epoch>=c['min_epochs'] and bad>=c['patience']:break
            cost.update(status='completed',test_opened=False,source_forward_calls=0,epochs=epoch,optimizer_steps=steps,best_val_nll=best)
            dest=Path(c['output'])/'trained';dest.mkdir(exist_ok=True)
            write_json(dest/(c['kind']+'_'+str(c['seed'])+'.json'),{'run':str(run.path),'checkpoint':str(run.path/'best.pt'),'config':c})
        except Exception:
            cost.update(status='failed');live(stage='reader_training_failed',training_run=str(run.path));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

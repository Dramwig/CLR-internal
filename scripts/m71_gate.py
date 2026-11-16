"""Full-training-data linear/B3 learnability gates; evaluation is reader_val only."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from clr.audit.m71_reader import Bank,typed_condition
from clr.audit.m71_runtime import snapshot,live,Cost
from clr.models.m71 import Reader
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded


def semantics(rows,device):
    conditions=[typed_condition(row) for row in rows]
    teams,routes=zip(*conditions)
    from clr.data.m7 import TEAMS
    options=[[TEAMS.index(option) for option in row['options']] if row['task']=='A4' else [0,0] for row in rows]
    return {'team':torch.tensor(teams,device=device),'route':torch.tensor(routes,device=device),
            'option_teams':torch.tensor(options,device=device)}


def materialize(bank,tasks,device):
    keys=list(bank.pool);lookup={key:i for i,key in enumerate(keys)}
    pool=torch.stack([bank.pool[key] for key in keys]).to(device)
    result={}
    for task in tasks:
        rows=[row for row in bank.rows if row['task']==task]
        result[task]={'rows':rows,'indices':torch.tensor([lookup[r['message_id']] for r in rows],device=device),
                      'labels':torch.tensor([r['label'] for r in rows],device=device),
                      'text':semantics(rows,device)}
    return pool,result


def batch_data(pool,data,indices):
    return {key:value[indices] for key,value in data['text'].items()},pool[data['indices'][indices]],data['labels'][indices]


@torch.no_grad()
def evaluate(model,pool,data,config):
    model.eval();metrics={};prediction={}
    for task,item in data.items():
        logits=[]
        for start in range(0,len(item['rows']),config['batch_size']):
            indices=torch.arange(start,min(len(item['rows']),start+config['batch_size']),device=config['device'])
            text,source,labels=batch_data(pool,item,indices)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(text,source,task)
            logits.append(out.float().cpu())
        logits=torch.cat(logits);labels=item['labels'].cpu();p=softmax(logits.numpy(),axis=1)
        result={'n':len(labels),'accuracy':float((logits.argmax(1)==labels).float().mean()),
                'nll':float(torch.nn.functional.cross_entropy(logits,labels))}
        if task=='A1':result['AUROC']=float(roc_auc_score(labels,p[:,1]))
        metrics[task]=result
        prediction[task]={'logits':logits,'labels':labels,'group_ids':[r['group_id'] for r in item['rows']]}
    return metrics,prediction


def main(config):
    torch.set_num_threads(2);seed_everything(config['seed']);torch.cuda.set_device(config['device'])
    with recorded(config) as run:
        snapshot(run);cost=Cost(run,config);root=Path(config['output']);live(stage='learnability_'+config['kind'],gate_run=str(run.path))
        try:
            train=Bank(root,'reader_train',representation=config['representation'],load_sequences=False)
            val=Bank(root,'reader_val',representation=config['representation'],load_sequences=False)
            if len(train.entries)!=8000 or len(val.entries)!=1000:raise ValueError('C1 requires the full prescribed group counts')
            if {r['group_id'] for r in train.entries}&{r['group_id'] for r in val.entries}:raise ValueError('Training/validation group leakage')
            model=Reader(config['kind'],len(train.layer_ids)).to(config['device'])
            mean,scale=train.pool_normalization();model.mean.copy_(mean);model.scale.copy_(scale)
            train_pool,training=materialize(train,['A1','A4'],config['device'])
            val_pool,validation=materialize(val,['A1','A4'],config['device'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.01)
            rng=np.random.default_rng(config['seed']);history=[];best=float('inf');bad=0;exposure=0
            for epoch in range(1,config['epochs']+1):
                model.train();batches=[]
                for task,item in training.items():
                    permutation=rng.permutation(len(item['rows']))
                    batches.extend((task,permutation[start:start+config['batch_size']]) for start in range(0,len(permutation),config['batch_size']))
                rng.shuffle(batches)
                for bi,(task,ii) in enumerate(batches):
                    cost.check();indices=torch.as_tensor(ii,device=config['device'])
                    text,source,label=batch_data(train_pool,training[task],indices)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(text,source,task)
                    loss=torch.nn.functional.cross_entropy(logits.float(),label)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite C1 loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();exposure+=len(ii)
                    if bi%100==0:
                        cost.update();write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'batches':len(batches),'exposure':exposure,'gpu_hours':cost.elapsed()/3600})
                metrics,prediction=evaluate(model,val_pool,validation,config)
                score=np.mean([m['nll'] for m in metrics.values()]);history.append({'epoch':epoch,'metrics':metrics,'selection_nll':score,'exposure':exposure})
                if score<best:
                    best=score;bad=0;torch.save(model.state_dict(),run.path/'best.pt')
                    torch.save(prediction,run.path/'reader_val_predictions.pt');write_json(run.path/'best.json',history[-1])
                else:bad+=1
                write_json(run.path/'history.json',history);print(json.dumps(history[-1]),flush=True)
                if epoch>=config['min_epochs'] and bad>=config['patience']:break
            selected=json.loads((run.path/'best.json').read_text());m=selected['metrics']
            passed=m['A4']['accuracy']>=.45 and m['A1']['AUROC']>=.75
            folder=root/config['representation']/'gates';folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/(config['kind']+'.json'),{'run':str(run.path),'passed':passed,'metrics':m,
                       'checkpoint':str(run.path/'best.pt'),'reader_train_groups':len(train.entries),
                       'reader_val_groups':len(val.entries),'exposure':exposure,'epochs':epoch})
            cost.update(status='completed',test_opened=False,source_forward_calls=0,gate_passed=passed,epochs=epoch,exposure=exposure)
            live(stage='learnability_'+config['kind']+'_completed',gate_run=str(run.path),
                 **{config['representation']+'_'+config['kind']+'_gate_passed':passed})
        except Exception:
            cost.update(status='failed');live(stage='learnability_failed',gate_run=str(run.path));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

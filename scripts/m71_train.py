"""M7.1 four-task readers, 500-example overfit gate and matched exposure."""
import sys
sys.dont_write_bytecode=True
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m71_reader import Bank,Serializer
from clr.audit.m71_runtime import Cost,snapshot,live
from clr.audit.m7_evaluation import qa_summary
from clr.models.m71 import Reader
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded
from scripts.m71_gate import semantics


def source_for(model,bank,rows,device):
    if model.kind in ['linear','B3']:return bank.pool_batch(rows,device)
    if model.kind=='B0':return None
    return bank.batch(rows,device)


def text_for(model,serializer,rows,device):
    return semantics(rows,device) if model.kind in ['linear','B3'] else serializer.batch(rows,device)


@torch.no_grad()
def evaluate(model,bank,rows,serializer,c):
    model.eval();metrics={};predictions={}
    for task in ['A1','A2','A3','A4']:
        rs=[r for r in rows if r['task']==task]
        if not rs:continue
        logits=[]
        for start in range(0,len(rs),c['batch_size']):
            batch=rs[start:start+c['batch_size']]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(text_for(model,serializer,batch,c['device']),source_for(model,bank,batch,c['device']),task)
            logits.append(out.float().cpu())
        logits=torch.cat(logits);probs=logits.softmax(1).numpy()
        metrics[task]=qa_summary(rs,probs)
        predictions[task]={'logits':logits,'labels':torch.tensor([r['label'] for r in rs]),
                           'group_ids':[r['group_id'] for r in rs]}
    return metrics,predictions


def fixed_overfit_rows(bank,n=500):
    # Round robin over tasks and templates, using only reader_train messages.
    buckets={task:[r for r in bank.rows if r['task']==task] for task in ['A1','A2','A3','A4']}
    return [buckets[['A1','A2','A3','A4'][i%4]][i//4] for i in range(n)]


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);root=Path(c['output']);representation=c['representation']
        live(stage=c['mode']+'_'+c['kind'],training_run=str(run.path))
        try:
            gate=json.loads((root/representation/'C1.json').read_text())
            if not gate['passed']:raise ValueError('C1 not passed; CLR training forbidden')
            if c['mode']=='full' and not json.loads((root/representation/'C2.json').read_text())['passed']:
                raise ValueError('C2 not passed; full reader training forbidden')
            free,total=torch.cuda.mem_get_info(c['device'])
            if free/2**30<c['min_free_GiB']:raise RuntimeError('Reader GPU memory guard failed')
            sequence=c['kind'] not in ['linear','B3','B0']
            train=Bank(root,'reader_train',representation=representation,load_sequences=sequence,
                       limit=c.get('train_group_limit'))
            val=Bank(root,'reader_val',representation=representation,load_sequences=sequence,
                     limit=c.get('val_group_limit'))
            model=Reader(c['kind'],len(train.layer_ids)).to(c['device'])
            mean,scale=(train.pool_normalization() if c['kind'] in ['linear','B3'] else
                        train.sequence_normalization() if sequence else (model.mean.cpu(),model.scale.cpu()))
            model.mean.copy_(mean);model.scale.copy_(scale)
            serializer=None if c['kind'] in ['linear','B3'] else Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.01)
            rng=np.random.default_rng(c['seed']);history=[];exposure=0;bad=0;best=float('inf');minimum=0
            training=fixed_overfit_rows(train) if c['mode']=='overfit' else train.rows
            if c['kind']=='supervised':training=[r for r in training if r['task']=='A3']
            if c['kind']=='CLR' and c['mode']=='full':
                ref=json.loads((root/representation/'training'/f'B3_{c["seed"]}.json').read_text())
                minimum=ref['exposure']
            validation=training if c['mode']=='overfit' else val.rows
            if c['kind']=='supervised':validation=[r for r in validation if r['task']=='A3']
            frequencies={task:sum(row['task']==task for row in training) for task in ['A1','A2','A3','A4']}
            nonempty=sum(number>0 for number in frequencies.values())
            for epoch in range(1,c['epochs']+1):
                model.train();batches=[]
                for task in ['A1','A2','A3','A4']:
                    indices=[i for i,r in enumerate(training) if r['task']==task];rng.shuffle(indices)
                    batches.extend((task,indices[start:start+c['batch_size']]) for start in range(0,len(indices),c['batch_size']))
                rng.shuffle(batches)
                for bi,(task,ii) in enumerate(batches):
                    cost.check();rows=[training[i] for i in ii];target=torch.tensor([r['label'] for r in rows],device=c['device'])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(text_for(model,serializer,rows,c['device']),source_for(model,train,rows,c['device']),task)
                    loss=torch.nn.functional.cross_entropy(logits.float(),target)*len(training)/(nonempty*frequencies[task])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite M7.1 reader loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();exposure+=len(rows)
                    if bi%100==0:
                        cost.update();write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'batches':len(batches),
                            'exposure':exposure,'minimum_exposure':minimum,'GPUh':cost.elapsed()/3600})
                metrics,predictions=evaluate(model,train if c['mode']=='overfit' else val,validation,serializer,c)
                score=float(np.mean([m['nll'] for m in metrics.values()]))
                accuracy=sum(m['accuracy']*m['n'] for m in metrics.values())/sum(m['n'] for m in metrics.values())
                history.append({'epoch':epoch,'metrics':metrics,'selection_nll':score,'accuracy':accuracy,'exposure':exposure})
                # Exposure requirement applies to the returned checkpoint.
                if exposure>=minimum and score<best:
                    best=score;bad=0;torch.save(model.state_dict(),run.path/'best.pt')
                    torch.save(predictions,run.path/'validation_predictions.pt');write_json(run.path/'best.json',history[-1])
                elif exposure>=minimum:bad+=1
                write_json(run.path/'history.json',history);print(json.dumps(history[-1]),flush=True)
                if c['mode']=='overfit' and accuracy>=.98:break
                if c['mode']!='overfit' and epoch>=c['min_epochs'] and exposure>=minimum and bad>=c['patience']:break
            if not (run.path/'best.json').exists():raise RuntimeError('No checkpoint satisfies B3 exposure')
            selected=json.loads((run.path/'best.json').read_text())
            result={'run':str(run.path),'kind':c['kind'],'seed':c['seed'],'checkpoint':str(run.path/'best.pt'),
                    'exposure':exposure,'selected_exposure':selected['exposure'],'selected':selected,'test_opened':False}
            if c['mode']=='overfit':
                result['passed']=max(row['accuracy'] for row in history)>=.98
                model.load_state_dict(torch.load(run.path/'best.pt',map_location=c['device'],weights_only=True))
                diagnostic,_=evaluate(model,val,val.rows,serializer,c)
                result['reader_val_diagnostic']=diagnostic
                result['reader_val_used_for_overfit_stopping']=False
                write_json(root/representation/'C2.json',result)
            elif c['mode']=='full':
                if not json.loads((root/representation/'C2.json').read_text())['passed']:raise ValueError('C2 not passed')
                folder=root/representation/'training';folder.mkdir(exist_ok=True)
                write_json(folder/f'{c["kind"]}_{c["seed"]}.json',result)
            else:write_json(root/representation/'reader_smoke.json',result)
            cost.update(status='completed',test_opened=False,source_forward_calls=0,exposure=exposure)
            live(stage=c['mode']+'_'+c['kind']+'_completed',training_run=str(run.path))
        except Exception:
            cost.update(status='failed');live(stage='reader_training_failed',training_run=str(run.path));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

"""M6 single-source, shared-source and frozen-body adapter training; no test."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from clr.data.m6 import Bank,rows,nested_panel
from clr.data.m4.reader import targets,probe_inputs
from clr.models.m6 import NativeLite,NativeB3,TransferLite,MultiLite
from clr.models.clr_v0 import Serializer,QTYPES
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m6_runtime import Budget,live

def model_state(model,c):
    return model.trainable_state() if c['kind']=='b5b' else {k:v.detach().cpu() for k,v in model.state_dict().items()}

def restore(model,c,state):
    if c['kind']=='b5b':model.restore_trainable(state)
    else:model.load_state_dict(state)

def build(c):
    kind=c['kind'];spec=c['sources'];source=c['source'];d=spec[source]['dim']
    if kind=='lite':model=NativeLite(d)
    elif kind=='b3':model=NativeB3(d)
    elif kind=='transfer':
        donor=c['donor'];body=NativeLite(spec[donor]['dim']);body.load_state_dict(torch.load(c['donor_checkpoint'],map_location='cpu',weights_only=True));model=TransferLite(body,d)
    elif kind=='multi':model=MultiLite({s:x['dim'] for s,x in spec.items()})
    elif kind=='b5b':
        from clr.baselines.m6_b5b import B5b
        model=B5b(c['qwen_path'],d)
    else:raise ValueError(kind)
    return model

def forward(model,c,rs,bank,serializer,templates,norm,source):
    device=c['device'];prefix,mask=bank.batch(rs,device);q=rs[0]['qtype']
    if c['kind']=='b3':return model(*probe_inputs(rs,prefix,mask,templates,*norm[source]),q).float()
    if c['kind']=='b5b':return model.read(rs,prefix,mask,templates).float()
    text=serializer.batch(rs,device)
    if c['kind']=='multi':return model(text,prefix,mask,q,source).float()
    return model(text,prefix,mask,q).float()

@torch.no_grad()
def evaluate(model,c,groups,banks,serializer,templates,norm,budget):
    model.eval();out={};started=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['kind']!='b3'):
        for (source,q),rs in groups.items():
            loss=0.
            for i in range(0,len(rs),c['eval_batch']):
                batch=rs[i:i+c['eval_batch']];x=forward(model,c,batch,banks[source],serializer,templates,norm,source);loss+=float(-(targets(batch,c['device'])*x.log_softmax(-1)).sum())
                if i%(c['eval_batch']*100)==0:budget.check()
            out[source+':'+q]=loss/len(rs)
    budget.validation_seconds+=time.monotonic()-started;budget.validation_examples+=sum(map(len,groups.values()))
    return out

def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();budget=Budget(run,c,start);budget.training_seconds=0.;budget.validation_seconds=0.;budget.validation_examples=0;torch.set_num_threads(2);seed_everything(c['seed'])
    try:
        if c['mode']!='smoke':
            gate=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if gate.get('status')!='completed':raise RuntimeError('Training smoke required')
        used=list(c['sources']) if c['kind']=='multi' else [c['source']];train={};valid={};banks={};vb={};norm={};counts={}
        for s in used:
            spec=c['sources'][s];n=c.get('n_labels');panelpath=Path(spec['panels'])/f'{n}.parquet' if spec.get('panels') and n else None
            preselected=panelpath is not None and panelpath.exists()
            tr=pq.read_table(panelpath).to_pylist() if preselected else rows(spec['qa'],'reader_train','train')
            if c['mode']=='smoke':
                ids=set(sorted({r['problem_id'] for r in tr})[:c.get('smoke_problems',10)]);tr=[r for r in tr if r['problem_id'] in ids]
            if not preselected:tr=nested_panel(tr,c.get('n_labels'),c['panel_seed'])
            valsplit=c.get('validation_split','reader_val');vids=None
            if c.get('validation_problem_limit') and c['mode']!='smoke':
                from clr.data.questions.core import seed as panel_hash
                allids=set(pq.read_table(Path(spec['qa'])/f'{valsplit}.parquet',columns=['problem_id']).column('problem_id').to_pylist())
                vids=set(sorted(allids,key=lambda p:panel_hash(11,'M6-selection',p))[:c['validation_problem_limit']])
            vr=rows(spec['qa'],valsplit,'validate',ids=vids)
            if c['mode']=='smoke':
                ids=set(sorted({r['problem_id'] for r in vr})[:c.get('smoke_problems',10)]);vr=[r for r in vr if r['problem_id'] in ids]
            elif c.get('validation_problem_limit'):
                from clr.data.questions.core import seed as panel_hash
                ids=set(sorted({r['problem_id'] for r in vr},key=lambda p:panel_hash(11,'M6-selection',p))[:c['validation_problem_limit']]);vr=[r for r in vr if r['problem_id'] in ids]
            banks[s]=Bank(spec['root'],'reader_train','train',{r['problem_id'] for r in tr});vb[s]=Bank(spec['root'],valsplit,'validate',{r['problem_id'] for r in vr})
            # Unlabeled full training-distribution statistics are shared across label budgets.
            normpath=Path(spec['normalization'])
            if normpath.exists():
                no=np.load(normpath);mean,scale=no['mean'],no['scale']
            else:
                mean=banks[s].z.mean(0).numpy();scale=banks[s].z.std(0,correction=0).clamp_min(.01).numpy()
                if c.get('n_labels') is not None:raise ValueError('Small-N runs require frozen full-training normalization')
                normpath.parent.mkdir(parents=True,exist_ok=True);np.savez(normpath,mean=mean,scale=scale)
            norm[s]=[torch.from_numpy(x).to(c['device']) for x in [mean,scale]]
            for q in QTYPES:train[s,q]=[r for r in tr if r['qtype']==q];valid[s,q]=[r for r in vr if r['qtype']==q]
            counts[s]={q:len(train[s,q]) for q in QTYPES}
            if c.get('n_labels') or c['mode']=='smoke':
                write_json(run.path/f'{s}_label_panel.json',[{'problem_id':r['problem_id'],'state_ref':r['state_ref'],'qtype':r['qtype'],'template_id':r['template_id'],'options':r['options'],'replicate':r['replicate']} for r in tr])
            else:write_json(run.path/f'{s}_full_data.json',{'qa_path':spec['qa'],'instances':len(tr),'problem_ids':sorted({r['problem_id'] for r in tr}),'n_labels':'full'})
        from clr.sources.common import check_device
        check_device({'device':c['device'],'dtype':'float32' if c['kind']=='b3' else 'bfloat16','min_free_gib':14 if c['kind']=='b5b' else 6})
        model=build(c).to(c['device'])
        for s in used:
            if c['kind'] in ['lite','b5b']:model.state_mean.copy_(norm[s][0]);model.state_scale.copy_(norm[s][1])
            elif c['kind']=='transfer':model.mean.copy_(norm[s][0]);model.scale.copy_(norm[s][1])
            elif c['kind']=='multi':getattr(model,s+'_mean').copy_(norm[s][0]);getattr(model,s+'_scale').copy_(norm[s][1])
        serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True),load_config(c['templates'])) if c['kind'] not in ['b3','b5b'] else None
        templates=load_config(c['templates']);budget.check();batches_seen=0;examples={s:{q:0 for q in QTYPES} for s in used};rng=np.random.default_rng(c['seed']);history=[]
        trainable=[p for p in model.parameters() if p.requires_grad]
        write_json(run.path/'training_manifest.json',{'counts':counts,'n_labels':c.get('n_labels'),'normalization':'unlabeled full training bank, frozen across N','test_opened':False,'trainable_parameters':sum(p.numel() for p in trainable),'kind':c['kind'],'frozen_body':c['kind']=='transfer'})
        if c['kind']=='b3':
            # Match M3 per-type LR/epoch selection, retaining independent heads.
            selected={};beststates={}
            for s in used:
                for q in QTYPES:
                    group={(s,q):train[s,q]};vg={(s,q):valid[s,q]};initial=copy.deepcopy(model.models[q].state_dict());best=float('inf')
                    for lr in c['b3_lrs']:
                        model.models[q].load_state_dict(initial);opt=torch.optim.AdamW(model.models[q].parameters(),lr=lr,weight_decay=c['weight_decay']);bad=0
                        for epoch in range(1,c['epochs']+1):
                            training_start=time.monotonic()
                            model.train();rs=train[s,q];order=rng.permutation(len(rs))
                            for i in range(0,len(rs),c['b3_batch']):
                                batch=[rs[j] for j in order[i:i+c['b3_batch']]];opt.zero_grad(set_to_none=True);x=forward(model,c,batch,banks[s],serializer,templates,norm,s)
                                loss=-(targets(batch,c['device'])*x.log_softmax(-1)).sum(-1).mean();loss.backward();opt.step();examples[s][q]+=len(batch)
                            torch.cuda.synchronize(c['device']);budget.training_seconds+=time.monotonic()-training_start
                            if epoch%c.get('validation_every',1) and epoch!=c['epochs']:continue
                            vl=evaluate(model,c,vg,vb,serializer,templates,norm,budget)[s+':'+q]
                            if vl<best:
                                best=vl;beststates[q]=copy.deepcopy(model.models[q].state_dict());selected[q]={'lr':lr,'epoch':epoch,'val_nll':vl};bad=0
                            else:bad+=1
                            budget.check()
                            if epoch>=c['min_epochs'] and bad>=c['patience']:break
                    model.models[q].load_state_dict(beststates[q]);write_json(run.path/'selection.json',selected)
                    print(q,selected[q],flush=True)
            score=float(np.mean([x['val_nll'] for x in selected.values()]))
            torch.save({k:v.cpu() for k,v in model.state_dict().items()},run.path/'best.pt')
        else:
            opt=torch.optim.AdamW(trainable,lr=c['learning_rate'],weight_decay=c['weight_decay']);best=float('inf');bad=0
            total=sum(map(len,train.values()));weights={key:total/(len(train)*len(rs)) for key,rs in train.items()}
            for epoch in range(1,c['epochs']+1):
                training_start=time.monotonic()
                model.train();schedule=[];losses={key:0. for key in train}
                for key,rs in train.items():
                    order=rng.permutation(len(rs));schedule.extend((key,order[i:i+c['batch']]) for i in range(0,len(rs),c['batch']))
                rng.shuffle(schedule)
                for bi,((s,q),ii) in enumerate(schedule):
                    batch=[train[s,q][i] for i in ii];opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        x=forward(model,c,batch,banks[s],serializer,templates,norm,s);loss=-(targets(batch,c['device'])*x.log_softmax(-1)).sum(-1).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite M6 loss')
                    (loss*weights[s,q]).backward();torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step();examples[s][q]+=len(batch);losses[s,q]+=float(loss)*len(batch);batches_seen+=1
                    if bi%100==0:
                        budget.check();write_json(run.path/'progress.json',{'epoch':epoch,'batch':bi,'batches':len(schedule),'examples':examples,'gpu_hours':(time.monotonic()-start)/3600})
                torch.cuda.synchronize(c['device']);budget.training_seconds+=time.monotonic()-training_start
                if epoch%c.get('validation_every',1) and epoch!=c['epochs']:continue
                vl=evaluate(model,c,valid,vb,serializer,templates,norm,budget);score=float(np.mean(list(vl.values())));row={'epoch':epoch,'val_nll':vl,'selection_nll':score,'gpu_hours':(time.monotonic()-start)/3600}
                history.append(row);write_json(run.path/'history.json',history);print(json.dumps(row),flush=True)
                if epoch>=c['min_epochs']:
                    if score<best:
                        best=score;bad=0;torch.save(model_state(model,c),run.path/'best.pt');write_json(run.path/'best.json',row)
                    else:bad+=1
                if epoch>=c['min_epochs'] and bad>=c['patience']:break
            score=best
        (Path(c['output'])/'trained').mkdir(parents=True,exist_ok=True)
        write_json(Path(c['output'])/'trained'/f"{c['case']}_{c['seed']}.json",{'run':str(run.path),'checkpoint':str(run.path/'best.pt'),'sha256':hashlib.sha256((run.path/'best.pt').read_bytes()).hexdigest()})
        run.write_metrics(status='completed',best_val_nll=score,test_opened=False,examples=examples,optimizer_steps=batches_seen,trainable_parameters=sum(p.numel() for p in trainable),training_seconds=budget.training_seconds,validation_seconds=budget.validation_seconds,validation_examples=budget.validation_examples,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)
        live(stage='C_training',last_trained=c['case'],last_training_run=str(run.path))
    finally:
        run.write_metrics(gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

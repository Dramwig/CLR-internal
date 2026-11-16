"""Reader-train-only sibling generation for M5b-D, explicit sigma=0 + 7 pi."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m5b_runtime import Budget,snapshot
from clr.sources.codi_v2 import load
from clr.policy import prepare
from clr.data.labels import numeric_answer
from clr.search.m5a import Engine,seed_for
from clr.search.m5b import root_branch,children,oracle_scores


def train_rows(c):
    p=Path(c['train_manifest'])
    if p.resolve()!=Path('data/dumps/codi_gpt2_v3/reader_train/manifest.parquet').resolve():raise ValueError('D generation allowlists reader_train only')
    rows=pq.read_table(p,filters=[('kind','=','clean'),('t','=',0),('layer','=',12)]).to_pylist()
    if {r['split'] for r in rows}!={'reader_train'}:raise ValueError('Wrong D split')
    return [{'id':r['problem_id'],'question':r['question'],'answer':r['gold_answer'],'pad_to':r['pad_to'],'cohort_id':r['cohort_id'],'left_padding':r['left_padding']} for r in rows]


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();budget=Budget(run,c,start);source=None;e=None;status='failed'
    try:
        torch.set_num_threads(2);seed_everything(c['seed']);smoke=c['stage']=='smoke';budget.check();snapshot(run,c,['scripts/m5b_generate_d.py','clr/search/m5b.py','clr/search/m5a.py','clr/policy.py',c['protocol']])
        if not smoke:
            gate=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if not gate.get('gate_passed') or gate.get('status')!='completed':raise ValueError('D train-split generation smoke required')
        rows=train_rows(c);rng=np.random.default_rng(c['seed']);order=rng.permutation(len(rows))[:c['train_questions']+c['validation_questions']];rows=[rows[i] for i in order]
        if smoke:rows=rows[:c.get('smoke_questions',5)]
        source=load({**load_config(c['protocol'])['source'],'device':c['device'],'min_free_gib':c['min_free_gib']});e=Engine(source,c['batch'],c['sigma'])
        if c.get('merge_lora',True):source.model.codi=source.model.codi.merge_and_unload(safe_merge=True);source.model.codi.tie_weights()
        panels=[];shard=[];total=0;timings=[]
        for i,p in enumerate(rows):
            ps=time.monotonic();pre=prepare(p,source);ts=np.random.default_rng(seed_for(c['seed'],p['id'],'D-t')).choice(6,size=2,replace=False).tolist();gold=numeric_answer(p['answer'])
            if gold is None:raise ValueError('Invalid D gold')
            for t in ts:
                rb=root_branch(pre,t);ss=[seed_for(c['seed'],p['id'],'D-child',t,k) for k in range(8)];child=children(e,rb,8,ss)
                v,cost,hits=oracle_scores(e,child,p,c['seed'],4,f'D:{t}',gold)
                z=torch.zeros(8,7,768);z[:,:child.history.shape[1]]=child.history.cpu()
                shard.append({'history':z,'length':child.history.shape[1],'outcomes':torch.tensor(hits,dtype=torch.float32),'problem_id':p['id'],'t':t,'panel':'train' if i<c['train_questions'] else 'validation','source_forwards':cost});total+=1
            panels.append({'problem_id':p['id'],'ts':ts,'panel':'train' if i<c['train_questions'] else 'validation'})
            if len(shard)>=128 or i==len(rows)-1:
                torch.save(shard,run.path/f'siblings_{i:04d}.pt');shard=[]
            timings.append(time.monotonic()-ps);write_json(run.path/'progress.json',{'questions':i+1,'total_questions':len(rows),'groups':total,'gpu_hours':(time.monotonic()-start)/3600,'source_forwards':e.lanes});budget.check()
            if i%25==0:print(i+1,total,flush=True)
        write_json(run.path/'panels.json',panels);write_json(run.path/'timing.json',timings)
        status='completed';run.write_metrics(status=status,gate_passed=True,gpu_hours=(time.monotonic()-start)/3600,questions=len(rows),groups=total,source_forward_calls=source.forward_calls,lane_forwards=e.lanes,test_opened=False)
        write_json(Path(c['output'])/f"D_generation_{c['stage']}.json",{'run':str(run.path)})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls if source else 0,lane_forwards=e.lanes if e else 0,test_opened=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

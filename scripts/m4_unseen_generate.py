"""M1 policy on original M3 contexts; per-problem durable source generation."""
import argparse,json,time,hashlib,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.sources.codi_v2 import load
from clr.policy import prepare,rollout
from clr.data.labels import numeric_answer
from clr.data.loader import ShardWriter
from clr.data.questions.loader import StateStore
from clr.data.questions.core import seed

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);source=None
    prior=0.
    for file in Path('runs').glob('*/config.json'):
        old=json.loads(file.read_text())
        if file.parent.resolve()!=run.path.resolve() and old.get('milestone')=='M4' and old.get('stage')=='unseen_generate':
            prior+=json.loads((file.parent/'metrics.json').read_text()).get('gpu_hours',0.)
    if c['mode']=='full':
        gate=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
        if gate['status']!='completed':raise ValueError('Successful val-proxy source smoke required')
        estimate=gate['projected_full_gpu_hours']
        if prior+estimate>c['source_budget_hours']:raise RuntimeError(f'D projected {prior+estimate:.3f} GPUh exceeds cap; no reduced K/pool fallback')
    plan=json.loads(Path(c['output'], 'unseen_smoke_plan.json' if c['mode']=='smoke' else 'unseen_plan.json').read_text())
    out=Path(c['output'])/('unseen_smoke' if c['mode']=='smoke' else 'unseen');out.mkdir(exist_ok=True)
    source=load({**load_config(c['protocol'])['source'],'device':c['source_device'],'min_free_gib':c['source_min_free_gib']})
    stores={};earlycache={};timings=defaultdict(list);done=0
    def budget():
        elapsed=time.monotonic()-start
        if elapsed/3600+prior>=c['source_budget_hours']-.002:raise TimeoutError('D 6 GPU-hour budget exhausted; no silent scaling')
        if c['mode']=='smoke' and elapsed>c['smoke_seconds']:raise TimeoutError('D smoke exceeded 15 minutes')
    try:
      for split,rows in plan['splits'].items():
        for row in rows:
            budget();pid=row['id'];directory=out/split/'parts'/hashlib.sha256(pid.encode()).hexdigest()[:24]
            if (directory/'complete.json').exists():done+=1;continue
            if directory.exists():raise RuntimeError('Incomplete problem shard requires explicit recovery/version: '+str(directory))
            tick=time.monotonic();key=(row['origin_root'],row['origin_split'])
            if key not in stores:
                stores[key]=StateStore(*key)
                earlypath=Path(key[0])/'early_exit.parquet' if key[1]=='dumps' else Path(key[0])/key[1]/'early_exit.parquet'
                earlycache[key]=pq.read_table(earlypath).to_pylist()
            p=prepare(row,source);clean=torch.stack([z[0,0].cpu() for z in p.states]);saved=stores[key].prefix(pid,{'traj_id':-1,'t':6})
            if not torch.equal(clean.half(),saved.half()):raise ValueError('Recomputed context does not exactly reproduce saved fp16 states: '+pid)
            writer=ShardWriter(directory);gold=numeric_answer(row['answer']);metadata={k:row[k] for k in ['cohort_id','left_padding','pad_to','protocol_sha256']}
            metadata.update(problem_id=pid,source='codi_gpt2',split=split,pool=row['pool'],question=row['question'],gold_answer=row['answer'],intermediate_values=row.get('intermediate_values',[]),cot_parse_failed=row.get('cot_parse_failed',True))
            writer.add(clean[:,None,:],{**metadata,'kind':'clean','traj_id':-1,'t0':0,'steps':list(range(7)),'layers':[12],'model_answer':row['model_answer'],'correct':row['correct']})
            mc=[]
            if split=='unseen_train':
                starts=np.random.default_rng(seed(c['seed'],pid,'starts')).integers(0,6,size=c['trajectory_k'])
                for t0 in range(6):
                    indices=[j for j,t in enumerate(starts) if t==t0]
                    if not indices:continue
                    budget();seeds=[seed(c['seed'],pid,'traj',j) for j in indices]
                    results=rollout(row,p,t0,c['source_sigma'],0,seeds)
                    for j,result in zip(indices,results):
                        correct=numeric_answer(result['answer'])==gold
                        writer.add(result['states'][:,None,:],{**metadata,'kind':'traj','traj_id':j,'t0':t0,'steps':list(range(t0,7)),'layers':[12],'model_answer':result['answer'],'correct':correct,'rollout_seed':result['seed']},result['feedback'])
            else:
                for t in range(6):
                    budget();results=rollout(row,p,t,c['source_sigma'],0,[seed(c['seed'],pid,'mc',t,k) for k in range(c['mc_k'])])
                    correct=[int(numeric_answer(r['answer'])==gold) for r in results]
                    mc.append({'problem_id':pid,'t':t,'policy':'pi','answers':[r['answer'] for r in results],'correct':correct,'value':float(np.mean(correct))})
            writer.close();pq.write_table(pa.Table.from_pylist([r for r in earlycache[key] if r['problem_id']==pid]),directory/'early_exit.parquet')
            if mc:pq.write_table(pa.Table.from_pylist(mc),directory/'mc.parquet')
            duration=time.monotonic()-tick;timings[split].append(duration);done+=1
            write_json(directory/'complete.json',{'problem_id':pid,'seconds':duration,'origin':key,'fp16_exact':True})
            run.write_metrics(completed_questions=done,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls,progress=split)
      for split,rows in plan['splits'].items():
        manifests=[];early=[];mc=[]
        for row in rows:
            directory=out/split/'parts'/hashlib.sha256(row['id'].encode()).hexdigest()[:24]
            for r in pq.read_table(directory/'manifest.parquet').to_pylist():r['shard']=str(directory.relative_to(out/split)/r['shard']);manifests.append(r)
            early+=pq.read_table(directory/'early_exit.parquet').to_pylist()
            if (directory/'mc.parquet').exists():mc+=pq.read_table(directory/'mc.parquet').to_pylist()
        pq.write_table(pa.Table.from_pylist(manifests),out/split/'manifest.parquet');pq.write_table(pa.Table.from_pylist(early),out/split/'early_exit.parquet')
        if mc:pq.write_table(pa.Table.from_pylist(mc),out/split/'mc.parquet')
      estimate=sum(np.mean(timings[s])*n for s,n in [('unseen_train',996),('unseen_eval',997)] if timings[s])/3600
      write_json(run.path/'timings.json',dict(timings))
      run.write_metrics(status='completed',projected_full_gpu_hours=estimate,completed_questions=done,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls);print(run.path)
    finally:
      run.write_metrics(gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

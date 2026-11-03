"""SVAMP clean dumps/early exits only; capped five GPU hours across smoke/full."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.sources.codi_v2 import load
from clr.policy import prepare,early_exit
from clr.data.labels import numeric_answer
from clr.data.loader import ShardWriter
from clr.utils.stats_v2 import bootstrap_mean

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);prior=0
    if c['mode']=='full':
        smoke=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
        if smoke['status']!='completed':raise ValueError('Unseen smoke required')
        prior=smoke['gpu_hours']
    protocol=load_config(c['protocol']);source=load({**protocol['source'],'device':c['source_device'],'min_free_gib':c['source_min_free_gib']})
    rows=json.loads(Path(c['output'],c['pool_path']).read_text())
    for begin in range(0,len(rows),128):
        cohort=rows[begin:begin+128];width=max(len(source.tokenizer.encode(r['question'].strip().replace('  ',' '))) for r in cohort)
        for r in cohort:
            r['pad_to']=width;r['cohort_id']=f'{c["pool_name"]}:{begin//128}';r['left_padding']=width-len(source.tokenizer.encode(r['question'].strip().replace('  ',' ')))
    if c['mode']=='smoke':rows=rows[:max(1,int(len(rows)*c['smoke_fraction']))]
    out=run.path/'dumps';writer=ShardWriter(out);early=[]
    for row in rows:
        if (time.monotonic()-start)/3600+prior>=c['pool_budget_hours']:raise TimeoutError('D five GPU-hour cap reached')
        p=prepare(row,source);answers=[early_exit(row,p,t)['answer'] for t in range(7)]
        gold=numeric_answer(row['answer']);correct=[numeric_answer(a)==gold for a in answers]
        for t in range(7):early.append({'problem_id':row['id'],'t':t,'answer':answers[t],'correct':correct[t]})
        writer.add(torch.stack([z[0,0].cpu() for z in p.states])[:,None,:],{'problem_id':row['id'],'kind':'clean','source':'codi_gpt2','split':c['pool_name'],'traj_id':-1,'t0':0,'steps':list(range(7)),'layers':[12],
            'question':row['question'],'gold_answer':row['answer'],'model_answer':answers[-1],'label':correct[-1],'correct':correct[-1],
            'cohort_id':row['cohort_id'],'left_padding':row['left_padding'],'pad_to':row['pad_to'],'protocol_sha256':hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest()})
        if len(early)//7%64==0:writer.flush();write_json(run.path/'early_partial.json',early)
        run.write_metrics(completed_questions=len(early)//7,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls)
    storage=writer.close();pq.write_table(pa.Table.from_pylist(early),run.path/'early_exit.parquet')
    stats={str(t):bootstrap_mean([int(r['correct']) for r in early if r['t']==t]) for t in range(7)}
    run.write_metrics(status='completed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls,accuracy_by_step=stats,storage=storage)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

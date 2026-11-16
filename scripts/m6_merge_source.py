"""Merge immutable M6 problem chunks after all GPU shards completed."""
import argparse,hashlib,json,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import snapshot,live

def merge(c):
    planpath=Path(c['output'])/'plan.json';plan=json.loads(planpath.read_text());ph=hashlib.sha256(planpath.read_bytes()).hexdigest();summary={}
    for split in c['splits']:
        root=Path(c['dump_root'])/split;rs=plan['splits'][split];man=[];early=[];mc=[];storage=[]
        if (root/'completed.json').exists():raise FileExistsError(root/'completed.json')
        for first in range(0,len(rs),8):
            part=json.loads((root/f'chunk_{first:05d}.json').read_text())
            if part['ids']!=[r['id'] for r in rs[first:first+8]] or part['sigma']!=plan['sigma'] or part['plan_sha256']!=ph:raise ValueError('Chunk protocol/IDs differ from locked plan')
            man+=part['manifest'];early+=part['early'];mc+=part['mc'];storage.append(part['storage'])
        keys={(r['problem_id'],r['traj_id'],r['t'],r['layer']) for r in man}
        if len(keys)!=len(man) or {r['problem_id'] for r in man}!={r['id'] for r in rs}:raise ValueError('Duplicate/missing source states')
        if len(early)!=7*len(rs):raise ValueError('Missing early-exit labels')
        if not split.startswith('reader') and len(mc)!=12*len(rs):raise ValueError('Missing pi/pi_T K16 labels')
        for name,rows in [('manifest',man),('early_exit',early),('mc',mc)]:
            if rows:
                p=root/f'{name}.parquet';pq.write_table(pa.Table.from_pylist(rows),p.with_suffix('.tmp.parquet'),compression='zstd');p.with_suffix('.tmp.parquet').replace(p)
        write_json(root/'completed.json',{'problem_count':len(rs),'storage':storage,'sigma':plan['sigma'],'plan_sha256':ph,'merged_from_durable_chunks':True})
        summary[split]={'problems':len(rs),'states':len(man),'mc_panels':len(mc)}
    return summary

def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();snapshot(run,{**c,'section':'B'})
    result=merge(c);write_json(run.path/'data_counts.json',result);live(stage='B_data_completed',data_counts=result)
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

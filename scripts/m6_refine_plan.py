"""Cost-only refinement after fixed B data, before reader_train generation."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import hours,snapshot,live


def refine(c):
    root=Path(c['output']);path=root/'plan.json';original=path.read_bytes();plan=json.loads(original)
    for split in ['reader_val','val','cal','test']:
        done=json.loads(Path(c['dump_root'],split,'completed.json').read_text())
        if done['problem_count']!=len(plan['splits'][split]) or done['sigma']!=plan['sigma']:raise ValueError('Fixed-data counts/policy mismatch: '+split)
    trainroot=Path(c['dump_root'])/'reader_train'
    if trainroot.exists() and any(trainroot.iterdir()):raise ValueError('Cost refinement must precede all reader_train generation')
    used=sum(r['gpu_hours'] for r in hours() if r['section']=='B')
    seconds=float(np.mean(json.loads(Path(c['smoke_run'],'timing.json').read_text())['reader_train']))
    old_n=len(plan['splits']['reader_train']);available=max(0,30-used-c.get('loading_reserve_hours',.1))
    n=min(old_n,int(available*3600/(seconds*c['cost_safety_factor'])))
    if n<100:raise ValueError('Actual fixed data leave insufficient B budget for 100 reader_train problems')
    initial=root/'plan_initial.json'
    if initial.exists() and initial.read_bytes()!=original:raise ValueError('Initial plan already differs; explicit resume audit needed')
    if not initial.exists():initial.write_bytes(original)
    plan['splits']['reader_train']=plan['splits']['reader_train'][:n]
    plan['cost_estimate'].update(train_count=n,initial_train_count=old_n,actual_fixed_data_gpu_hours=used,
       projected_gpu_hours=used+c.get('loading_reserve_hours',.1)+n*seconds*c['cost_safety_factor']/3600,
       refinement='cost-only after complete fixed reader_val/val/cal/test, before any reader_train generation',
       initial_plan_sha256=hashlib.sha256(original).hexdigest())
    write_json(path,plan)
    return plan['cost_estimate']


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();snapshot(run,{**c,'section':'B'})
    result=refine(c);write_json(run.path/'cost_refinement.json',result);live(stage='B_train_plan_refined',cost_estimate=result)
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

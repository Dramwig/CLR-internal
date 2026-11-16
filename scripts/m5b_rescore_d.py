"""Score frozen val sibling observations with D; no source model or test access."""
import argparse,json,time
from pathlib import Path
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m5b_runtime import Budget,snapshot
from clr.search.m5a import State
from clr.search.m5b import Branch,ValueRunner


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();status='failed';budget=Budget(run,c,start);torch.set_num_threads(2)
    try:
        snapshot(run,c,['scripts/m5b_rescore_d.py','clr/search/m5b.py']);templates=load_config(c['templates']);count=0
        for seed in c['seeds']:
            value=ValueRunner('D-lite',seed,c,templates,c['device'])
            for f in sorted(Path(c['original_B_run']).glob(f'B_states_*_{seed}.pt')):
                item=torch.load(f,map_location='cpu',weights_only=True);rows=[]
                for t,z in enumerate(item['history']):
                    z=z.to(c['device']);score=value(Branch(State(z[:,-1:],(),t+1),z))
                    rows.extend({'problem_id':item['problem_id'],'seed':seed,'t':t,'child':k,'D-lite':float(v)} for k,v in enumerate(score))
                dest=f.name.replace('B_states_','B_').replace('.pt','.parquet');pq.write_table(pa.Table.from_pylist(rows),run.path/dest,compression='zstd');count+=1
                if count%25==0:budget.check()
        if count!=900:raise ValueError(f'D rescore incomplete: {count}/900')
        status='completed';run.write_metrics(status=status,questions=300,seeds=c['seeds'],gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=False);write_json(Path(c['output'])/'D_B_rescore.json',{'run':str(run.path)})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,test_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

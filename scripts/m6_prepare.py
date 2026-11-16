"""CPU-only M6 full-training normalization and frozen reference checkpoint index."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.data.m6 import Bank,rows,nested_panel
from clr.models.m6 import NativeB3
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.m6_runtime import snapshot
from clr.utils.idea_a import write_json

def main(c):
 with recorded(c) as run:
    snapshot(run,{**c,'section':c.get('section','CD')})
    print(run.path,flush=True)
    start=time.monotonic();torch.set_num_threads(2)
    spec=c['sources']['llama'];p=Path(spec['normalization'])
    if not p.exists():
        bank=Bank(spec['root'],'reader_train','train');p.parent.mkdir(parents=True,exist_ok=True)
        np.savez(p.with_suffix('.tmp.npz'),mean=bank.z.mean(0).numpy(),scale=bank.z.std(0,correction=0).clamp_min(.01).numpy());p.with_suffix('.tmp.npz').replace(p)
    refs={}
    for sd in [11,22,33]:
        model=NativeB3(768)
        for q in ['Q1','Q2','Q3','Q4']:model.models[q].load_state_dict(torch.load(Path(c['m3_run'])/f'B3_{q}_{sd}.pt',map_location='cpu',weights_only=True))
        dest=Path(c['output'])/f'gpt2_B3_reference_{sd}.pt'
        if not dest.exists():torch.save(model.state_dict(),dest)
        refs[f'single_gpt2_b3_{sd}']=str(dest);refs[f'single_gpt2_lite_{sd}']=str(Path(c['m41_runs'][str(sd)])/'best.pt')
    write_json(Path(c['output'])/'reference_checkpoints.json',refs)
    for source,spec in c['sources'].items():
        destination=Path(spec['panels']);destination.mkdir(parents=True,exist_ok=True)
        if not (destination/'3000.parquet').exists():
            panel=nested_panel(rows(spec['qa'],'reader_train','train'),3000,11)
            for n in [100,300,1000,3000]:pq.write_table(pa.Table.from_pylist(panel[:n]),destination/f'{n}.parquet',compression='zstd')
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start)
    print(run.path,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

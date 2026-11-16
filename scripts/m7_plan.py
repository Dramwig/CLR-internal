"""Storage/compute plan using original frozen validation inputs only."""
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from clr.sources.kv_relay import verify_idea_a,sender_text
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live

def main(c):
    verify_idea_a(c['idea_a'])
    from latdef.data.schema import Example
    with recorded(c) as run:
        snapshot(run)
        tok=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True)
        path=Path(c['idea_a'])/'data/synthetic/m1-v2-r2/val.jsonl'
        rows=[Example(**r) for r in map(json.loads,path.open()) if r['meta']['alpha']==1 and r['meta']['d_pub']==0]
        lengths=[len(tok.encode(sender_text(r),add_special_tokens=False)) for r in rows]
        # Raw Qwen4 message: H=8,D=128,K+V,fp16; two distinct real messages/episode.
        per_layer=2*8*128*2
        mean=float(np.mean(lengths));target=12000
        by_layers={str(n):mean*per_layer*n*2*target/2**30 for n in [1,4,8]}
        # 12 GiB KV allowance leaves 3 GiB for metadata/checkpoints/statistics.
        count=int(12*2**30/(mean*per_layer*2*1.15))
        count=count//12*12
        counts={'reader_train':count*8//12,'reader_val':count//12,'cal_A':count//24,'cal_B':count//24,'test':count*2//12}
        hidden=Path('/home/me/datasets/HiddenBench/1e3c25b1fd798c6717f4df0463edd3825c8e37f9/benchmark.json')
        obj=json.loads(hidden.read_text())
        hidden_info={'available':True,'sha256':hashlib.sha256(hidden.read_bytes()).hexdigest(),
                     'top_type':type(obj).__name__,'top_keys':list(obj)[:20] if isinstance(obj,dict) else [],
                     'scope':'availability and schema only; not included in M7 comparison'}
        result={'mode':'val_only_plan','val_groups':len(rows),'token_mean':mean,'token_min':min(lengths),
                'token_max':max(lengths),'target_KV_GiB_by_layers':by_layers,'chosen_layers':[35],
                'chosen_episode_counts':counts,'planned_KV_allowance_GiB':12,'safety_factor':1.15,
                'changed_settings':['saved layers 4->1','episode counts reduced for 15GiB cap'],
                'test_opened':False,'source_forward_calls':0,'gpu_hours':0,'hiddenbench':hidden_info}
        run.write_metrics(status='completed',**result)
        write_json(Path(c['output'])/'plan.json',{'run':str(run.path),**result});live(storage_plan=str(run.path))
        print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

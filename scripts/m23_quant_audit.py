"""Recompute a frozen random panel from interrupted M1 segments (no new labels)."""
import argparse,json,time
from pathlib import Path
import torch
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.questions.core import seed
from clr.data.questions.loader import StateStore
from clr.sources.codi_v2 import load
from clr.policy import prepare

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);plan=json.loads(Path(c['plan']).read_text())
    if c['mode']=='full' and json.loads(Path(c['smoke_run'],'metrics.json').read_text())['status']!='completed':raise ValueError('Audit smoke required')
    protocol=load_config(c['protocol']);source=load({**protocol['source'],'device':c['source_device'],'min_free_gib':c['source_min_free_gib']})
    records=[];panels={}
    for split,bound in [('reader_train',10112),('test',768)]:
        selected=sorted(plan['splits'][split][:bound],key=lambda r:seed(c['seed'],'quant_audit',r['id']))[:100]
        panels[split]=[r['id'] for r in selected]
        if c['mode']=='smoke':selected=selected[:5]
        store=StateStore(c['root'],split)
        for row in selected:
            prefix=prepare(row,source)
            for t,z in enumerate(prefix.states):
                x=z[0,0].cpu();stored=store.state(row['id'],-1,t)
                diff=(x-stored).abs();roundtrip=x.half().float()
                records.append({'problem_id':row['id'],'split':split,'t':t,'max_abs':diff.max().item(),
                     'relative_l2':(torch.linalg.vector_norm(x-stored)/torch.linalg.vector_norm(x)).item(),
                     'fp32_exact':bool(torch.equal(x,stored)),'fp16_exact':bool(torch.equal(roundtrip,stored)),
                     'fp16_mismatch_elements':int((roundtrip!=stored).sum())})
            run.write_metrics(completed_problems=len(records)//7,source_forward_calls=source.forward_calls)
    write_json(run.path/'panel.json',panels);write_json(run.path/'quantization.json',records)
    run.write_metrics(status='completed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls,
        max_abs=max(r['max_abs'] for r in records),max_relative_l2=max(r['relative_l2'] for r in records),fp16_exact_all=all(r['fp16_exact'] for r in records))
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

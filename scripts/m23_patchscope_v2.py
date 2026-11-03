"""Versioned B5a throughput change; validate on fixed reader_val, never tune on test."""
import argparse, functools, json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.special import softmax
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.sources.codi_v2 import load
from clr.data.questions.loader import StateStore
from clr.data.questions.core import seed
from scripts import m23_patchscope as original

def main(c):
    if c['stage']=='patchscope_batch_gate':
        with recorded(c) as run:
            start=time.monotonic()
            source=load({**load_config(c['protocol'])['source'],'device':c['source_device'],'min_free_gib':c['source_min_free_gib']})
            templates=load_config(c['templates']);store=StateStore(c['root'],'reader_val')
            rows=pq.read_table(Path(c['output'])/'full/reader_val.parquet').to_pylist()
            ids=set(sorted({r['problem_id'] for r in rows},key=lambda p:seed(11,'b5-batch-gate',p))[:50])
            result=[]
            for q in ['Q1','Q2','Q3','Q4']:
                panel=[r for r in rows if r['problem_id'] in ids and r['qtype']==q and not r['is_heldout_template'] and r['state_ref']['kind']=='clean']
                panel=sorted(panel,key=lambda r:seed(11,r['problem_id'],r['state_ref']['t'],tuple(r['options'])))[:128]
                a=original.read(source,store,panel,templates,batch_size=8)
                b=original.read(source,store,panel,templates,batch_size=c['source_batch_size'])
                result.append({'qtype':q,'n':len(panel),'max_logit_error':float(np.max(abs(a-b))),'max_probability_error':float(np.max(abs(softmax(a,axis=1)-softmax(b,axis=1)))),'answer_agreement':float(np.mean(a.argmax(1)==b.argmax(1)))})
            passed=all(r['max_probability_error']<=1e-4 and r['answer_agreement']==1 for r in result)
            write_json(run.path/'equivalence.json',{'fixed_problem_ids':sorted(ids),'rows':result,'passed':passed,'criteria':{'max_probability_error':1e-4,'answer_agreement':1}})
            run.write_metrics(status='completed',passed=passed,source_forward_calls=source.forward_calls,gpu_hours=(time.monotonic()-start)/3600)
            if not passed:raise RuntimeError('B5a batch equivalence gate failed; preserve batch8 run')
            print(run.path)
    else:
        gate=json.loads(Path(c['batch_gate_run'],'metrics.json').read_text())
        if gate['status']!='completed' or not gate['passed']:raise RuntimeError('Batch gate required')
        original.read=functools.partial(original.read,batch_size=c['source_batch_size'])
        original.main(c)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

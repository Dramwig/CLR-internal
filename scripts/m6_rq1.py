"""Final pre-specified RQ1 check: small sigma, frozen sigma1.5 values, val only."""
import argparse,json,time
import os
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.m5b_bc import val_rows
from clr.search.m5a import Engine,seed_for
from clr.search.m5b import root_branch,children,continue_branch,ValueRunner
from clr.sources.codi_v2 import load
from clr.policy import prepare,rollout
from clr.data.labels import numeric_answer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import Budget,live

def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();budget=Budget(run,c,start);source=None;e=None
    try:
        torch.set_num_threads(2)
        search=c['search'];N=search['candidate_N'];beam=search['beam'];horizon=search['horizon']
        if search!={'candidate_sigmas':[.3,.75],'candidate_N':8,'beam':4,'horizon':6,'value_names':['B3','CLR-lite'],'include_clean':True,'value_training_sigma':1.5}:raise ValueError('M6 RQ1 frozen protocol differs from explicit YAML')
        if c['mode']=='full' and not json.loads(Path(c['smoke_run'],'metrics.json').read_text()).get('gate_passed'):raise ValueError('RQ1 smoke required')
        source=load({**load_config(c['protocol'])['source'],'device':c['device'],'min_free_gib':c['min_free_gib']})
        rs=val_rows(c);indices=c['smoke_indices'] if c['mode']=='smoke' else range(300)
        if c['mode']=='smoke' and len(indices)>15:raise ValueError('Smoke >5%')
        if c['merge_lora']:
            p=rs[list(indices)[0]];pre=prepare(p,source);ss=[seed_for('M6-merge',k) for k in range(4)];before=rollout(p,pre,0,1.5,0,ss)
            source.model.codi=source.model.codi.merge_and_unload(safe_merge=True);source.model.codi.tie_weights();pre=prepare(p,source);after=rollout(p,pre,0,1.5,0,ss)
            delta=max(float((a['states']-b['states']).abs().max()) for a,b in zip(before,after));agree=[numeric_answer(o['answer']) for o in before]==[numeric_answer(o['answer']) for o in after]
            write_json(run.path/'merge_gate.json',{'max_abs_state':delta,'same_answers':agree,'threshold':.001})
            if not agree or delta>.001:raise RuntimeError('GPT2 LoRA fusion equivalence gate failed')
        timings=[]
        e=Engine(source,c['batch'],1.5)
        if c.get('resume_run'):
            old=Path(c['resume_run']);oldc=json.loads((old/'config.json').read_text())
            for key in ['seeds','protocol','m3_baseline_run','m41_runs','batch']:
                if oldc[key]!=c[key]:raise ValueError('RQ1 resume changed scientific settings: '+key)
            for p in old.glob('pred_*.parquet'):os.link(p,run.path/p.name)
        for sd in c['seeds']:
            values={name:ValueRunner(name,sd,c,load_config(c['templates']),source.device) for name in search['value_names']}
            for i in indices:
                if (run.path/f'pred_{i:03d}_{sd}.parquet').exists():continue
                ts=time.monotonic();p=rs[i];pre=prepare(p,source);gold=numeric_answer(p['answer']);rows=[]
                greedy,_,_=e.decode(root_branch(pre,horizon).state);greedy=greedy[0]
                if greedy!=numeric_answer(p['expected']):raise RuntimeError('Frozen greedy replay mismatch')
                for sigma in search['candidate_sigmas']:
                    e.sigma=sigma;root=root_branch(pre)
                    ss=[seed_for(sd,p['id'],'M6-pass8',k) for k in range(N)];pool=continue_branch(e,root.select([0]*N),ss,stop=horizon,clean_indices=[0]);answers,_,_=e.decode(pool.state)
                    if answers[0]!=greedy:raise RuntimeError('sigma0 independent candidate lost')
                    for name,vr in values.items():
                        branch=root;before=e.lanes
                        while branch.state.t<horizon:
                            t=branch.state.t;seeds=[seed_for(sd,p['id'],'M6-beam',t,parent,k) for parent in range(len(branch.state.z)) for k in range(N)]
                            branch=children(e,branch,N,seeds);scores=vr(branch);branch=branch.select(np.argsort(-scores,kind='stable')[:(1 if branch.state.t==horizon else beam)])
                        a,_,_=e.decode(branch.state)
                        rows.append({'problem_id':p['id'],'seed':sd,'sigma':sigma,'value':name,'N':N,'b':beam,'horizon':horizon,'correct':a[0]==gold,'greedy_correct':greedy==gold,'pass_at_8':any(x==gold for x in answers),'source_forwards':1+e.lanes-before})
                        budget.check()
                pq.write_table(pa.Table.from_pylist(rows),run.path/f'pred_{i:03d}_{sd}.parquet',compression='zstd');timings.append(time.monotonic()-ts)
                write_json(run.path/'progress.json',{'seed':sd,'completed_problem_index':i,'seconds_per_problem_seed':timings[-1],'gpu_hours':(time.monotonic()-start)/3600});print(sd,i,timings[-1],flush=True);budget.check()
        run.write_metrics(status='completed',gate_passed=True,test_opened=False,seconds_per_problem_seed=float(np.mean(timings)),gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls)
        live(rq1_run=str(run.path),rq1_stage='smoke_passed' if c['mode']=='smoke' else 'completed')
    finally:
        run.write_metrics(gpu_hours=(time.monotonic()-start)/3600 if source else 0,source_forward_calls=source.forward_calls if source else 0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

"""YAML-only M5b B/C; data access hard-allowlisted to the frozen val300 manifest."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m5b_runtime import Budget,snapshot
from clr.sources.codi_v2 import load
from clr.policy import prepare,rollout
from clr.data.labels import numeric_answer
from clr.search.m5a import Engine,seed_for,majority
from clr.search.m5b import Branch,root_branch,children,continue_branch,ValueRunner,oracle_scores,group_metrics


def val_rows(c):
    p=Path(c['val_manifest'])
    if p.resolve()!=Path('data/dumps/codi_gpt2_v3/val/manifest.parquet').resolve():raise ValueError('B/C requires frozen val-only manifest')
    rs=pq.read_table(p,filters=[('kind','=','clean'),('t','=',0),('layer','=',12)]).to_pylist()
    if len(rs)!=300 or len({r['problem_id'] for r in rs})!=300 or {r['split'] for r in rs}!={'val'}:raise ValueError('Require exact val300')
    return [{'id':r['problem_id'],'question':r['question'],'answer':r['gold_answer'],'pad_to':r['pad_to'],'cohort_id':r['cohort_id'],'left_padding':r['left_padding'],'expected':r['model_answer']} for r in rs]


def source_profile(source):
    from transformers.pytorch_utils import Conv1D
    lw=sum(m.weight.numel() for m in source.model.codi.modules() if isinstance(m,(torch.nn.Linear,Conv1D)))
    lw+=sum(m.weight.numel() for m in source.model.prj.modules() if isinstance(m,torch.nn.Linear))
    return {'source_linear_flops':2*lw,'attention_flops_per_context_token':4*12*768,'value_profile':'torch.utils.flop_counter; matrix/attention FLOPs; embedding, activations, normalization and serializer CPU excluded',
            'source_equivalent':'value FLOPs/(source linear FLOPs + attention FLOPs per token*(pad_to+4)); executed source lane-token forwards count=1 per lane-call'}


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();status='failed';e=None;source=None;budget=Budget(run,c,start)
    try:
        torch.set_num_threads(2);seed_everything(c['seed']);smoke=c['stage']=='smoke';budget.check()
        snapshot(run,c,['scripts/m5b_bc.py','clr/search/m5b.py','clr/search/m5a.py','clr/policy.py','clr/models/clr_v1.py',c['protocol'],c['templates']])
        if not smoke:
            gate=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if gate.get('status')!='completed' or not gate.get('gate_passed'):raise RuntimeError('Passed B/C val-only smoke required')
            if c.get('budget_plan'):
                plan=json.loads(Path(c['budget_plan']).read_text())
                if not plan['gate_passed']:raise RuntimeError('Budget estimate did not pass')
        rows=val_rows(c);indices=c.get('smoke_indices',[0]) if smoke else c.get('problem_indices',list(range(300)))
        if smoke and len(indices)>15:raise ValueError('Smoke exceeds 5% val IDs')
        source=load({**load_config(c['protocol'])['source'],'device':c['device'],'min_free_gib':c['min_free_gib']})
        e=Engine(source,c['batch'],c['sigma'])
        if c.get('merge_lora',True):
            p=rows[indices[0]];pre=prepare(p,source);ss=[seed_for('merge-gate',k) for k in range(4)];a=rollout(p,pre,0,1.5,0,ss)
            source.model.codi=source.model.codi.merge_and_unload(safe_merge=True);source.model.codi.tie_weights()
            pre=prepare(p,source);b=rollout(p,pre,0,1.5,0,ss)
            delta=max((x['states']-y['states']).abs().max().item() for x,y in zip(a,b));agree=[numeric_answer(x['answer']) for x in a]==[numeric_answer(x['answer']) for x in b]
            write_json(run.path/'merge_gate.json',{'max_abs_state':delta,'answers_equal':agree,'threshold':.001,'val_only':True})
            if not agree or delta>.001:raise RuntimeError('LoRA fusion equivalence gate failed')
            del a,b,pre
        prof=source_profile(source);templates=load_config(c['templates']);write_json(run.path/'compute_profile.json',prof)
        counts={'B':0,'C':0};timing=[];passed=True
        for seed in c['seeds']:
            values={v:ValueRunner(v,seed,c,templates,source.device) for v in c['values']}
            # All seven lengths occur in search; profile each actual input length once.
            for v,vr in values.items():
                for length in range(2,8):vr.flop_cost(length)
            for pi in indices:
                budget.check();ps=time.monotonic();p=rows[pi];gold=numeric_answer(p['answer'])
                if gold is None:raise ValueError('Invalid gold')
                pre=prepare(p,source);root=root_branch(pre);final=root_branch(pre,6)
                before=e.lanes;aa,_,limited=e.decode(final.state);s0cost=7+e.lanes-before;greedy=aa[0]
                if greedy!=numeric_answer(p['expected']):passed=False
                brow=[];history=[]
                if 'B' in c['parts']:
                    for t in range(6):
                        rb=root_branch(pre,t);N=17;ss=[seed_for(seed,p['id'],'B-child',t,k) for k in range(N)]
                        child=children(e,rb,N,ss)
                        # The sigma=0 child reproduces the next saved clean observation.
                        delta=(child.state.z[0]-pre.states[t+1][0]).abs().max().item()
                        if delta>1e-4:raise RuntimeError('Clean-child protocol mismatch')
                        vv,oc,hits=oracle_scores(e,child,p,seed,16,f'B:{t}',gold)
                        scores={v:vr(child) for v,vr in values.items()}
                        for k in range(N):
                            brow.append({'problem_id':p['id'],'seed':seed,'t':t,'child':k,'sigma':0. if k==0 else c['sigma'],'Vhat':float(vv[k]),'outcomes':hits[k],'oracle_source_forwards':oc,**{v:float(s[k]) for v,s in scores.items()}})
                        history.append(child.history.cpu());budget.check()
                    pq.write_table(pa.Table.from_pylist(brow),run.path/f'B_{pi:03d}_{seed}.parquet',compression='zstd')
                    torch.save({'history':history,'problem_id':p['id'],'seed':seed,'t':list(range(6))},run.path/f'B_states_{pi:03d}_{seed}.pt');counts['B']+=1
                out=[{'method':'S0','value':'none','N':1,'b':1,'horizon':0,'answer':greedy,'correct':greedy==gold,'source_forwards':s0cost,'value_source_equivalent':0.}]
                denom=prof['source_linear_flops']+prof['attention_flops_per_context_token']*(p['pad_to']+4)
                if 'C' in c['parts']:
                    for horizon in c['horizons']:
                        for N in c['Ns']:
                            before=e.lanes;ss=[seed_for(seed,p['id'],'S2-pool',horizon,N,k) for k in range(N)]
                            pool=continue_branch(e,root.select([0]*N),ss,stop=horizon,clean_indices=[0])
                            if horizon<6:pool=continue_branch(e,pool,[seed_for(s,'clean-tail') for s in ss],stop=6,clean_indices=range(N))
                            latent_cost=1+e.lanes-before
                            for name,vr in values.items():
                                scores=vr(pool);best=int(np.argmax(scores));eq=N*vr.flop_cost(7)/denom
                                # Value selection needs observed z_T only. Decode
                                # exactly the selected branch for production S2.
                                decode_before=e.lanes;chosen,_,_=e.decode(pool.state.select([best]));decode_cost=e.lanes-decode_before
                                out.append({'method':'S2','value':name,'N':N,'b':1,'horizon':horizon,'answer':chosen[0],'correct':chosen[0]==gold,'source_forwards':latent_cost+decode_cost,'value_source_equivalent':eq})
                            decode_before=e.lanes;answers,_,limited=e.decode(pool.state);cost=latent_cost+e.lanes-decode_before
                            # Full decodes belong to voting/pass@N reference,
                            # not to the production value selector's cost.
                            if answers[0]!=greedy:raise RuntimeError('S2 clean trajectory lost')
                            out.append({'method':'S1-clean','value':'none','N':N,'b':1,'horizon':horizon,'answer':majority(answers),'correct':majority(answers)==gold,'source_forwards':cost,'value_source_equivalent':0.})
                            # At the deterministic terminal, oracle K16 is exactly the gold
                            # indicator. Compute it analytically rather than executing 16
                            # identical decodes; privileged reference is also pass@N.
                            hit=any(a==gold for a in answers)
                            for name in ['oracle','pass@N']:
                                out.append({'method':'S2','value':name,'N':N,'b':1,'horizon':horizon,'answer':gold if hit else answers[0],'correct':hit,'source_forwards':cost,'value_source_equivalent':0.,'privileged':True})
                        for N in c['Ns']:
                            for b in c['beams']:
                                for name,vr in values.items():
                                    before=e.lanes;branch=root;eq=0.;value_evals=0
                                    while branch.state.t<horizon:
                                        t=branch.state.t;ss=[seed_for(seed,p['id'],'S3',horizon,N,b,t,i,k) for i in range(len(branch.state.z)) for k in range(N)]
                                        branch=children(e,branch,N,ss);scores=vr(branch);eq+=len(scores)*vr.flop_cost(branch.history.shape[1])/denom;value_evals+=len(scores)
                                        keep=np.argsort(-scores,kind='stable')[:(1 if branch.state.t==horizon else b)];branch=branch.select(keep)
                                    if horizon<6:branch=continue_branch(e,branch,[seed_for(seed,p['id'],'S3-clean-tail',horizon,N,b)],clean_indices=[0])
                                    ans,_,limited=e.decode(branch.state)
                                    out.append({'method':'S3','value':name,'N':N,'b':b,'horizon':horizon,'answer':ans[0],'correct':ans[0]==gold,'source_forwards':1+e.lanes-before,'value_source_equivalent':eq,'value_evaluations':value_evals})
                                    budget.check()
                    # Matched independent pi voting costs, preserved as M5a S1 baseline.
                    # Candidate-set clean requirements apply to S2/S3; S1 stays independent pi.
                    n=c.get('vote_max',128);answers=[];ds=[]
                    for vs in range(0,n,e.batch):
                        count=min(e.batch,n-vs);ss=[seed_for(seed,p['id'],'S1-iid',k) for k in range(vs,vs+count)]
                        vote_final=e.continue_pi(root.state.select([0]*count),ss);ab,db,_=e.decode(vote_final);answers.extend(ab);ds.extend(db)
                    for n1 in c.get('vote_Ns',[1,2,4,8,16,32,64,128]):
                        a=majority(answers[:n1]);out.append({'method':'S1','value':'none','N':n1,'b':1,'horizon':6,'answer':a,'correct':a==gold,'source_forwards':1+6*n1+sum(ds[:n1]),'value_source_equivalent':0.})
                    for r in out:r.update(problem_id=p['id'],seed=seed)
                    pq.write_table(pa.Table.from_pylist(out),run.path/f'C_{pi:03d}_{seed}.parquet',compression='zstd');counts['C']+=1
                tm={'problem':pi,'seed':seed,'seconds':time.monotonic()-ps,'gpu_hours':(time.monotonic()-start)/3600,'lane_forwards':e.lanes,'counts':dict(counts)};timing.append(tm);write_json(run.path/'timing.json',timing);write_json(run.path/'progress.json',tm);budget.check();print(json.dumps(tm),flush=True)
            write_json(run.path/'value_profiles.json',{v:vr.flops_by_length for v,vr in values.items()});del values;torch.cuda.empty_cache()
        status='completed';run.write_metrics(status=status,gate_passed=passed,gpu_hours=(time.monotonic()-start)/3600,questions=len(indices),seeds=c['seeds'],source_forward_calls=source.forward_calls,lane_forwards=e.lanes,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,test_opened=False,parts=c['parts'])
        write_json(Path(c['output'])/f"bc_{c['stage']}_{'-'.join(c['parts'])}_{c['seed']}.json",{'run':str(run.path)})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=source.forward_calls if source else 0,lane_forwards=e.lanes if e else 0,test_opened=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

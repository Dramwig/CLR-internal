"""Val300 continuation/batch gate; all failures saved, no test-set inspection."""
import argparse,json,time,hashlib,copy
from pathlib import Path
import torch
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.sources.codi_v2 import load
from clr.policy import prepare,rollout,early_exit
from clr.data.labels import numeric_answer
from clr.data.m1 import quantization

REFERENCE=Path('runs/20261028T131534016226Z_83bfd1aab8dd_d4e55b')
def m01_seed(seed,pid,k):return int.from_bytes(hashlib.sha256(f'{seed}:{pid}:{k}'.encode()).digest()[:8],'big')%(2**63-1)
def main(c):
 with recorded(c) as run:
    start=time.perf_counter();source=None
    try:
        smoke=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
        if smoke['status']!='completed' or smoke['questions']!=200:raise ValueError('Successful 200-problem smoke required')
        plan=json.loads(Path(c['candidate_path']).read_text());protocol=load_config(c['protocol'])
        if smoke['candidate_sha256']!=hashlib.sha256(Path(c['candidate_path']).read_bytes()).hexdigest():raise ValueError('Candidate manifest changed')
        torch.set_num_threads(c['threads']);seed_everything(c['seed']);source=load({**protocol['source'],'device':c['device'],'min_free_gib':c['min_free_gib']})
        references={}
        for i,item in enumerate(json.loads((REFERENCE/'summary.json').read_text())):
            if item['setting']=={'sigma':1.5,'from_step':3,'temperature':0}:
                references[item['seed']]={'summary':item,'panel':{r['id']:r for r in json.loads((REFERENCE/f'panel_{i+1:03d}.json').read_text())}}
        tol={key:max(x['summary']['metrics'][key]['mean'] for x in references.values())-min(x['summary']['metrics'][key]['mean'] for x in references.values()) for key in ['accuracy','nondegenerate']}
        write_json(run.path/'gate_definition.json',{'tolerances':tol,'rule':'each matched seed AND three-seed average absolute delta <= M0.1 max-minus-min; zero-noise all t native and persisted-state answers must match 100%; early T exact', 'context':'original batch1 KV replicated; persisted fp16 z_t separately tested with original KV'})
        failures=[];checks=[];noise={s:[] for s in c['stochastic_seeds']};early=[];max_abs=0.;max_rel=0.;disk_abs=0.;disk_rel=0.
        for i,row in enumerate(plan['splits']['val']):
            prefix=prepare(row,source);normal=source.run(row);expected=numeric_answer(normal['answer'])
            e=early_exit(row,prefix,6);same=e['answer']==normal['answer'];early.append(same)
            if not same:
                write_json(run.path/'early_T_mismatch.json',{'id':row['id'],'normal':normal['answer'],'early':e['answer']})
                raise RuntimeError('Val early exit T mismatch; stopped as required')
            gold=numeric_answer(row['answer'])
            for t in range(6):
                native=rollout(row,prefix,t,0,0,list(range(16)))
                stored=copy.copy(prefix);stored.states=[x.half().float() for x in prefix.states]
                quant=rollout(row,stored,t,0,0,list(range(16)))
                for mode,outputs in [('native_fp32',native),('saved_fp16',quant)]:
                    agrees=all(numeric_answer(o['answer'])==expected for o in outputs)
                    if not agrees:failures.append({'id':row['id'],'t':t,'mode':mode,'expected':expected,'answers':[numeric_answer(o['answer']) for o in outputs],'cohort_id':row['cohort_id'],'left_padding':row['left_padding']})
                    target=prefix.states[t+1][0,0].cpu() if mode=='native_fp32' else prefix.states[t+1][0,0].cpu().half().float()
                    d=torch.stack([o['states'][1] for o in outputs])-target
                    absdiff=d.abs().max().item();rel=(d.norm(dim=-1)/target.norm().clamp_min(1e-30)).max().item()
                    if mode=='native_fp32':max_abs=max(max_abs,absdiff);max_rel=max(max_rel,rel)
                    else:disk_abs=max(disk_abs,absdiff);disk_rel=max(disk_rel,rel)
                    checks.append({'id':row['id'],'t':t,'mode':mode,'same_answer':agrees,'max_abs':absdiff,'relative_l2':rel})
            for seed in c['stochastic_seeds']:
                outputs=rollout(row,prefix,3,1.5,0,[m01_seed(seed,row['id'],k) for k in range(16)])
                correct=[numeric_answer(o['answer'])==gold for o in outputs]
                ref=references[seed]['panel'][row['id']]
                noise[seed].append({'id':row['id'],'correct':correct,'answers':[numeric_answer(o['answer']) for o in outputs], 'reference_correct':ref['correct']})
            if (i+1)%10==0:
                write_json(run.path/'zero_noise_disagreements.json',failures);write_json(run.path/'continuation_checks.json',checks);write_json(run.path/'noise_panel.json',noise)
            run.write_metrics(completed_questions=i+1,zero_noise_disagreements=len(failures),wall_seconds=time.perf_counter()-start,source_forward_calls=source.forward_calls)
        comparisons=[]
        for seed,panel in noise.items():
            got={'accuracy':sum(sum(r['correct'])/16 for r in panel)/300,'nondegenerate':sum(0<sum(r['correct'])<16 for r in panel)/300}
            ref={key:references[seed]['summary']['metrics'][key]['mean'] for key in tol}
            comparisons.append({'seed':seed,'batched':got,'batch1':ref,'abs_delta':{key:abs(got[key]-ref[key]) for key in tol},'passed':all(abs(got[key]-ref[key])<=tol[key]+1e-12 for key in tol)})
        write_json(run.path/'zero_noise_disagreements.json',failures);write_json(run.path/'continuation_checks.json',checks);write_json(run.path/'noise_panel.json',noise);write_json(run.path/'batch_comparison.json',comparisons)
        passed=not failures and all(x['passed'] for x in comparisons) and all(early)
        run.write_metrics(status='completed',gate_passed=passed,questions=300,early_T_agreement=sum(early)/300,zero_noise_disagreements=len(failures),noise_comparison=comparisons,tolerances=tol,native_next_state_max_abs=max_abs,native_next_state_max_relative_l2=max_rel,saved_next_state_max_abs=disk_abs,saved_next_state_max_relative_l2=disk_rel)
        if not passed:
            with Path('docs/BLOCKERS.md').open('a') as f:f.write(f'\n## M1 batch/continuation gate NOT passed — {run.path.name}\nZero-noise discrepancies: {len(failures)}; three-seed comparisons: {comparisons}. No full generation launched; exact per-problem evidence in run.\n')
        print(run.path,flush=True)
    finally:
        elapsed=time.perf_counter()-start;run.write_metrics(wall_seconds=elapsed,gpu_hours=elapsed/3600 if source else 0,source_forward_calls=source.forward_calls if source else 0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

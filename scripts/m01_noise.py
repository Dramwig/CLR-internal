"""Fixed-panel CODI v2 noise study. Individual calls obey batch=1 fp32."""
import argparse,hashlib,json,signal,time
from pathlib import Path
import torch
from clr.sources.codi_v2 import load
from clr.sources.common import max_latent_difference
from clr.data.labels import numeric_answer
from clr.data.noise_stats_v2 import summarize,recommend
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded
from clr.utils.stats import mean_ci

def seed_for(seed,pid,k):
    return int.from_bytes(hashlib.sha256(f'{seed}:{pid}:{k}'.encode()).digest()[:8],'big')%(2**63-1)

def deadline(*args):raise TimeoutError('M0.1 noise smoke exceeded 900 seconds')
def main(c):
 with recorded(c) as run:
    started=time.perf_counter();source=None
    if c['mode']=='smoke':signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
    try:
        protocol=load_config(c['protocol']);p_hash=hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest()
        if protocol['generation_batch_size']!=1 or protocol['source']['dtype']!='float32':raise ValueError('This runner implements only the explicit fp32 batch=1 fallback')
        rows=json.loads(Path(c['data']).read_text());splits=json.loads(Path(c['splits']).read_text())
        if [r['id'] for r in rows]!=splits['val'] or len(rows)<300:raise ValueError('Require frozen clean val panel >=300')
        if c['mode']=='smoke':
            if not 0<c['limit']<=len(rows)*.05:raise ValueError('Smoke limit exceeds 5%')
        else:
            proof=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if proof.get('status')!='completed' or proof.get('protocol_sha256')!=p_hash:raise RuntimeError('Matching successful smoke required')
            if not (protocol['official_alignment_passed'] or protocol.get('review',{}).get('fp32_noise_authorized',False)):
                raise RuntimeError('Source protocol agreement gate has not passed and no explicit review exception authorizes fp32 noise')
        torch.set_num_threads(c['threads']);seed_everything(11);source=load(protocol['source'])
        for start in range(0,len(rows),128):
            group=rows[start:start+128]
            width=max(len(source.tokenizer.encode(r['question'].strip().replace('  ',' '))) for r in group)
            for r in group:r['pad_to']=width
        write_json(run.path/'panel.json',[{'id':r['id'],'pad_to':r['pad_to']} for r in rows]);write_json(run.path/'protocol.json',protocol)
        if c['mode']=='smoke':rows=rows[:c['limit']]
        # Repeated input, identical dtype/context/batch, elementwise latent check.
        a=source.run(rows[0]);b=source.run(rows[0])
        run.write_metrics(determinism={'max_abs_latent_difference':max_latent_difference(a['latent_steps'],b['latent_steps']),'same_answer':a['answer']==b['answer']})
        settings=[{'sigma':s,'from_step':f,'temperature':0} for f in c['from_steps'] for s in c['sigmas']]
        settings += [{'sigma':0,'from_step':0,'temperature':t} for t in c['temperatures']]
        results=[]
        for setting in settings:
            for seed in c['seeds']:
                samples=[]
                for row in rows:
                    answers=[];correct=[]
                    for k in range(c['K']):
                        rng=seed_for(seed,row['id'],k)
                        out=source.run(row,{'from_step':setting['from_step'],'sigma':setting['sigma'],'seed':rng},decode={'temperature':setting['temperature'],'seeds':[rng]},capture_latents=False)
                        ans=numeric_answer(out['answer']);answers.append(ans);correct.append(ans==numeric_answer(row['answer']))
                        run.logger.log('rollout',problem_id=row['id'],setting=setting,seed=seed,k=k,answer=out['answer'],pred=ans,correct=correct[-1],meta=out['meta'])
                    samples.append({'id':row['id'],'answers':answers,'correct':correct})
                    elapsed=time.perf_counter()-started
                    run.write_metrics(progress={'setting':setting,'seed':seed,'completed_questions':len(samples),'questions_per_setting':len(rows),'completed_setting_seeds':len(results)},
                        wall_seconds=elapsed,gpu_hours=elapsed/3600,source_forward_calls=source.forward_calls)
                result={'setting':setting,'seed':seed,'metrics':summarize(samples,c['K'])}
                results.append(result);write_json(run.path/f"panel_{len(results):03d}.json",samples)
                write_json(run.path/'summary.json',results)
                run.write_metrics(completed_setting_seeds=len(results),wall_seconds=time.perf_counter()-started,source_forward_calls=source.forward_calls)
                print('finished',setting,'seed',seed,flush=True)
        # Mean of per-seed nondegeneracy (do not pool K across seeds).
        baseline=sum(r['metrics']['accuracy']['mean'] for r in results if r['setting']=={'sigma':0,'from_step':0,'temperature':0})/len(c['seeds'])
        candidates=[]
        seed_summary=[]
        for setting in settings:
            subset=[r for r in results if r['setting']==setting]
            candidates.append({**setting,'accuracy_drop':baseline-sum(r['metrics']['accuracy']['mean'] for r in subset)/len(subset),'nondegenerate':sum(r['metrics']['nondegenerate']['mean'] for r in subset)/len(subset)})
            if len(subset)>=3:
                seed_summary.append({'setting':setting,'statistics':'mean and 95% Student-t CI across stochastic seed means; question bootstrap is separately in summary.json',
                  'metrics':{key:mean_ci([r['metrics'][key]['mean'] for r in subset]) for key in ['accuracy','nondegenerate','invalid_rate','distinct_answers','distinct_answers_including_invalid']}})
        write_json(run.path/'seed_summary.json',seed_summary)
        recommendation=recommend(candidates) if c['mode']=='full' else {'selected':None,'reason':'Smoke is not eligible for sigma selection'}
        write_json(run.path/'recommendation.json',recommendation)
        run.write_metrics(status='completed',protocol_sha256=p_hash,questions=len(rows),K=c['K'],seeds=c['seeds'],recommendation=recommendation)
        print(run.path,flush=True)
    finally:
        signal.alarm(0);elapsed=time.perf_counter()-started
        run.write_metrics(wall_seconds=elapsed,gpu_hours=elapsed/3600 if source else 0,source_forward_calls=source.forward_calls if source else 0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

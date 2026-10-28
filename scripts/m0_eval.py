"""Config-only smoke, final reproduction, determinism, and validation noise study."""
import argparse,hashlib,importlib,json,signal,time
from pathlib import Path
import torch
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.stats import mean_ci
from clr.sources.common import max_latent_difference
from clr.data.labels import score_answer,canonical_answer

def timeout(signum,frame):raise TimeoutError('M0 smoke exceeded 900 seconds; full run forbidden')

def main(c):
 with recorded(c) as run:
    t=time.perf_counter();source=None
    if c['mode']=='smoke':signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:
        rows=json.loads(Path(c['data_path']).read_text())
        splits=json.loads(Path(c['split_path']).read_text())
        allowed=set(splits[c['split']])
        if len({r['id'] for r in rows})!=len(rows) or any(r['id'] not in allowed for r in rows):raise ValueError('Split membership/uniqueness failed')
        if c['mode']=='smoke' and (c['split']=='test' or c['limit']>len(rows)*.05):raise ValueError('Smoke must use <=5% non-test problems')
        if c['experiment']=='noise' and c['split'] not in ['val','cal']:raise ValueError('Noise selection may not use test')
        if c['mode']=='full':
            proof=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            smoke_cfg=json.loads(Path(c['smoke_run'],'config.json').read_text())
            if smoke_cfg.get('baseline_batch_size',1)!=c.get('baseline_batch_size',1) or (c['experiment']=='noise' and smoke_cfg.get('rollout_batch_size',1)!=c.get('rollout_batch_size',1)):raise ValueError('Batch protocol requires matching smoke')
            if proof['status']!='completed' or smoke_cfg['mode']!='smoke' or smoke_cfg['source']!=c['source']:raise ValueError('Matching successful smoke required')
        if c['limit'] is not None:rows=sorted(rows,key=lambda r:r['id'])[:c['limit']]
        write_json(run.path/'selection.json',{'problem_ids':[r['id'] for r in rows],
            'split_sha256':hashlib.sha256(Path(c['split_path']).read_bytes()).hexdigest(),
            'data_sha256':hashlib.sha256(Path(c['data_path']).read_bytes()).hexdigest()})
        write_json(run.path/'resources.json', {'source':c['source'], 'official_code_commit': __import__('subprocess').check_output(['git','-C',c['source'].get('code_path','third_party/huginn'),'rev-parse','HEAD'],text=True).strip()})
        seed_everything(c['seeds'][0]);source=importlib.import_module('clr.sources.'+c['source_name']).load(c['source'])
        source.model.eval()
        seed_results=[]
        for seed in c['seeds']:
            seed_everything(seed);base=[];diffs=[];noise_result={}
            predictions={}
            batch_size=c.get('baseline_batch_size',1)
            if batch_size>1:
                groups={}
                for row in rows:
                    length=len(source.tokenizer.encode(row['question'].strip().replace('  ',' ')))
                    groups.setdefault(length,[]).append(row)
                for group in groups.values():
                    for offset in range(0,len(group),batch_size):
                        batch=group[offset:offset+batch_size]
                        outs=source.run_batch([r['question'] for r in batch])
                        predictions.update({r['id']:o for r,o in zip(batch,outs)})
            for index,row in enumerate(rows):
                a=predictions[row['id']] if batch_size>1 else source.run(row['question'])
                if index<c.get('determinism_limit',len(rows)):
                    # Repeat the identical single-input call; batch/single numerical drift is separate.
                    single=source.run(row['question']) if batch_size>1 else a
                    b=source.run(row['question'])
                    diffs.append(max_latent_difference(single['latent_steps'],b['latent_steps']))
                    zero=source.run(row['question'],{'from_step':0,'sigma':0.,'seed':seed})
                    if single['answer']!=zero['answer'] or max_latent_difference(single['latent_steps'],zero['latent_steps'])!=0:
                        raise RuntimeError('Zero-noise identity failed')
                    run.logger.log('determinism',seed=seed,problem_id=row['id'],max_abs=diffs[-1],batch_vs_single_max_abs=max_latent_difference(a['latent_steps'],single['latent_steps']),batch_vs_single_answer_equal=a['answer']==single['answer'])
                base.append(score_answer(a['answer'],row['answer'],c['task']))
                run.logger.log('prediction',seed=seed,problem_id=row['id'],answer=a['answer'],correct=base[-1],meta=a['meta'])
            if c['experiment']=='noise':
                for sigma in c['sigma']:
                    acc=[];div=[];invalid=[]
                    for row in rows:
                        answers=[];correct=[]
                        perts=[{'from_step':c['from_step'],'sigma':sigma,'seed':int(hashlib.sha256(f'{seed}/{row["id"]}/{k}'.encode()).hexdigest()[:8],16)} for k in range(c['rollouts'])]
                        if c.get('rollout_batch_size',1)==c['rollouts']:
                            outputs=source.run_batch([row['question']]*c['rollouts'],perts)
                        elif c.get('rollout_batch_size',1)==1:
                            outputs=[source.run(row['question'],p) for p in perts]
                        else:raise ValueError('Supported rollout batch sizes: 1 or K')
                        for k,out in enumerate(outputs):
                            answer=canonical_answer(out['answer'],c['task']);answers.append(answer)
                            correct.append(score_answer(out['answer'],row['answer'],c['task']))
                            run.logger.log('rollout',seed=seed,problem_id=row['id'],sigma=sigma,k=k,answer=out['answer'],canonical=answer,correct=correct[-1],meta=out['meta'])
                        acc.append(sum(correct)/len(correct));div.append(len(set(x for x in answers if x is not None)))
                        invalid.append(sum(x is None for x in answers)/len(answers))
                    noise_result[str(sigma)]={'accuracy':sum(acc)/len(acc),'distinct_answers':sum(div)/len(div),'invalid_fraction':sum(invalid)/len(invalid)}
            seed_results.append({'seed':seed,'accuracy':sum(base)/len(base),'max_latent_abs_difference':max(diffs),'noise':noise_result})
            run.write_metrics(seed_results=seed_results)
        aggregate={'accuracy':mean_ci([r['accuracy'] for r in seed_results])}
        curves={str(s):{m:mean_ci([r['noise'][str(s)][m] for r in seed_results]) for m in ['accuracy','distinct_answers','invalid_fraction']} for s in c['sigma']} if c['experiment']=='noise' else {}
        run.write_metrics(status='completed',aggregate=aggregate,noise=curves,problems=len(rows),max_latent_abs_difference=max(r['max_latent_abs_difference'] for r in seed_results))
        print(run.path)
    finally:
        signal.alarm(0)
        if source and source.device.type=='cuda':torch.cuda.synchronize(source.device)
        elapsed=time.perf_counter()-t
        run.write_metrics(wall_seconds=elapsed,gpu_hours=elapsed/3600 if c['source']['device'].startswith('cuda') else 0,
            source_forward_calls=source.forward_calls if source else 0,cost_note='Allocated single-device wall hours, includes loading; shared GPU, not utilization-adjusted')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

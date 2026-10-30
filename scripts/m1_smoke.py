"""200-problem representative workload, capped at 15 minutes, never full data."""
import argparse,json,time,signal,hashlib,random
from pathlib import Path
import torch
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.sources.codi_v2 import load
from clr.policy import prepare,rollout,early_exit
from clr.data.labels import numeric_answer
from clr.data.m1 import rng_seed
from clr.data.loader import ShardWriter

def timeout(*a):raise TimeoutError('M1 200-question smoke exceeded 900 seconds; full run prohibited')
def main(c):
 with recorded(c) as run:
    started=time.perf_counter();source=None;writers={};times={'prepare':0.,'early':0.,'traj':0.,'mc':0.};mismatches=[]
    try:
        signal.signal(signal.SIGALRM,timeout);signal.alarm(c['smoke_seconds'])
        plan=json.loads(Path(c['candidate_path']).read_text());protocol=load_config(c['protocol'])
        torch.set_num_threads(c['threads']);seed_everything(c['seed'])
        source=load({**protocol['source'],'device':c['device'],'min_free_gib':c['min_free_gib']})
        # 100 train-distribution +100 fixed val; each ID sampled before results.
        rows=[('reader_train',r) for r in plan['splits']['reader_train'][:100]]+[('val',r) for r in plan['splits']['val'][:100]]
        if len(rows)!=c['smoke_questions'] or len(rows)>sum(len(v) for v in plan['splits'].values())*.05:raise ValueError('Invalid smoke size')
        labels=[];comparison=[]
        for index,(split,row) in enumerate(rows):
            t=time.perf_counter();prefix=prepare(row,source);times['prepare']+=time.perf_counter()-t
            t=time.perf_counter();early=[early_exit(row,prefix,j) for j in range(7)];times['early']+=time.perf_counter()-t
            normal=source.run(row)
            if normal['answer']!=early[-1]['answer']:
                mismatches.append({'id':row['id'],'normal':normal['answer'],'early_T':early[-1]['answer']})
                write_json(run.path/'early_T_mismatches.json',mismatches)
                raise RuntimeError('Early exit T differs from normal inference; stop')
            if index<3:
                for seed in c['stochastic_seeds']:
                    old=source.run(row,{'sigma':1.5,'from_step':3,'seed':seed});new=rollout(row,prefix,3,1.5,0,seed)
                    diff=max((old['latent_steps'][j]-new['feedback'][j-3]).abs().max().item() for j in range(3,6))
                    comparison.append({'id':row['id'],'seed':seed,'same_answer':old['answer']==new['answer'],'max_feedback_diff':diff})
                    if old['answer']!=new['answer'] or diff!=0:raise RuntimeError('M0.1 exact policy continuation equivalence failed')
            if split not in writers:writers[split]=ShardWriter(run.path/'smoke_dumps'/split,c['fp16_relative_l2_limit'])
            w=writers[split];gold=numeric_answer(row['answer']);model=early[-1]['answer']
            meta={'problem_id':row['id'],'source':'codi_gpt2','split':split,'question':row['question'],'gold_answer':row['answer'],'gold_cot':row['cot'],
                  'intermediate_values':row['intermediate_values'],'cot_parse_failed':row['cot_parse_failed'],'cohort_id':row['cohort_id'],
                  'left_padding':row['left_padding'],'pad_to':row['pad_to'],'protocol_sha256':row['protocol_sha256'],'model_answer':model,'correct':numeric_answer(model)==gold}
            all_layers=index<10
            states=torch.stack(prefix.layers) if all_layers else torch.stack([z[0,0].cpu() for z in prefix.states])[:,None,:]
            w.add(states,{**meta,'kind':'clean','traj_id':-1,'t0':0,'label':numeric_answer(model)==gold,'steps':list(range(7)),'layers':list(range(13)) if all_layers else [12]})
            t=time.perf_counter();groups={}
            for k in range(c['trajectory_K']):
                seed=rng_seed(c['stochastic_seeds'][k%3],row['id'],'traj',k=k)
                t0=random.Random(seed).randrange(6);groups.setdefault(t0,[]).append((k,seed))
            for t0,items in groups.items():
                out=rollout(row,prefix,t0,c['sigma'],0,[seed for k,seed in items])
                for (k,seed),o in zip(items,out):
                    w.add(o['states'][:,None,:],{**meta,'kind':'traj','traj_id':k,'t0':t0,'label':numeric_answer(o['answer'])==gold,'model_answer':o['answer'],'correct':numeric_answer(o['answer'])==gold,'steps':list(range(t0,7)),'layers':[12]},o['feedback'])
            times['traj']+=time.perf_counter()-t
            # Both policies at one uniformly cycled step per smoke problem; full MC runs all six.
            step=index%6;t=time.perf_counter()
            for policy,sigma,temp in [('pi',c['sigma'],0),('pi_T',0,1.)]:
                seeds=[rng_seed(c['stochastic_seeds'][k%3],row['id'],policy,step,k) for k in range(c['mc_K'])]
                out=rollout(row,prefix,step,sigma,temp,seeds)
                labels.append({'id':row['id'],'t':step,'policy':policy,'answers':[o['answer'] for o in out],'correct':[numeric_answer(o['answer'])==gold for o in out]})
            times['mc']+=time.perf_counter()-t
            if (index+1)%20==0:w.flush()
            run.write_metrics(completed_questions=index+1,stage_seconds=times,wall_seconds=time.perf_counter()-started,source_forward_calls=source.forward_calls)
        stats={k:w.close() for k,w in writers.items()}
        write_json(run.path/'mc_labels.json',labels);write_json(run.path/'m01_equivalence.json',comparison);write_json(run.path/'early_T_mismatches.json',mismatches)
        run.write_metrics(status='completed',questions=len(rows),stage_seconds=times,storage=stats,protocol_sha256=plan['protocol_sha256'],candidate_sha256=hashlib.sha256(Path(c['candidate_path']).read_bytes()).hexdigest(),scope='200 problems clean + all early exits + K8 trajectories + two K16 policies at one step per problem; all-layer first10; not full MC')
        print(run.path,flush=True)
    finally:
        signal.alarm(0);elapsed=time.perf_counter()-started
        run.write_metrics(wall_seconds=elapsed,gpu_hours=elapsed/3600 if source else 0,source_forward_calls=source.forward_calls if source else 0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

"""Append independent voting trajectories; preserve original failed-gate evidence."""
import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path
import torch
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.policy import prepare,rollout
from clr.sources.codi_v2 import load
from clr.search.m5a import Engine,State,seed_for
from clr.data.labels import numeric_answer
from scripts.m5a_search import val_rows


def main(c):
    with recorded(c) as run:
        print(run.path,flush=True);start=time.perf_counter();source=None;engine=None
        original=Path(c['search_run']);base=json.loads((original/'config.json').read_text())
        try:
            def deadline(*_):raise TimeoutError('Vote extension exceeded remaining authorized cost')
            signal.signal(signal.SIGALRM,deadline)
            signal.alarm(min(c['smoke_seconds'],int((c['max_gpu_hours']-c['prior_gpu_hours'])*3600)))
            for name in ['scripts/m5a_extend_votes.py','clr/search/m5a.py','scripts/m5a_search.py',c['config_path']]:
                dest=run.path/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
            if os.environ.get('CUDA_VISIBLE_DEVICES')!=c['gpu_uuid']:
                raise ValueError('Explicit GPU UUID mismatch')
            rows=val_rows(base);torch.set_num_threads(base['threads']);seed_everything(base['seed'])
            source=load({**load_config(base['protocol'])['source'],'device':'cuda:0','min_free_gib':c['min_free_gib']})
            engine=Engine(source,base['batch'],base['sigma'],base['noise_mode'])
            p=rows[0];prefix=prepare(p,source);ss=[seed_for('merge_gate',k) for k in range(16)]
            raw=rollout(p,prefix,0,base['sigma'],0,ss)
            source.model.codi=source.model.codi.merge_and_unload(safe_merge=True);source.model.codi.tie_weights()
            prefix=prepare(p,source);merged=rollout(p,prefix,0,base['sigma'],0,ss)
            delta=max((a['states']-b['states']).abs().max().item() for a,b in zip(raw,merged))
            agrees=[numeric_answer(a['answer']) for a in raw]==[numeric_answer(a['answer']) for a in merged]
            write_json(run.path/'merge_gate.json',{'max_state_abs':delta,'answers_identical':agrees})
            if not agrees or delta>base['merge_state_tolerance']:raise RuntimeError('Merge gate failed')
            del raw,merged,prefix
            recovered=[]
            for item in c['pools']:
                pi,seed=item['problem_index'],item['seed'];problem=rows[pi]
                old_path=original/f'vote_{pi:03d}_{seed}.json';old=json.loads(old_path.read_text())
                if old['target_reached'] or old['problem_id']!=problem['id']:
                    raise ValueError('Extension must target an actually uncovered frozen val pool')
                if len(old['answers'])%base['batch']!=0:
                    raise ValueError('Preserve original batch iid block addressing')
                pool={**old,'answers':list(old['answers']),
                      'cumulative_source_forwards':list(old['cumulative_source_forwards'])}
                prefix=prepare(problem,source);root=State(prefix.states[0],prefix.caches[0],0)
                while pool['cumulative_source_forwards'][-1]<old['target']:
                    count=len(pool['answers']);n=min(base['batch'],c['vote_max']-count)
                    if n<=0:raise RuntimeError('Explicit extended cap reached, no implicit shrinking')
                    ss=[seed_for(seed,problem['id'],'matched_vote',k) for k in range(count,count+n)]
                    state=engine.continue_pi(root.select([0]*n),ss);answers,costs,_=engine.decode(state)
                    pool['answers'].extend(answers);total=pool['cumulative_source_forwards'][-1]
                    for cost in costs:total+=6+cost;pool['cumulative_source_forwards'].append(total)
                pool.update(target_reached=True,status='completed',original_pool=str(old_path),
                            original_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                            original_trajectory_count=len(old['answers']))
                assert pool['answers'][:len(old['answers'])]==old['answers']
                assert pool['cumulative_source_forwards'][:len(old['answers'])]==old['cumulative_source_forwards']
                write_json(run.path/old_path.name,pool);recovered.append(old_path.name)
            write_json(run.path/'recovery.json',{'original_search_run':str(original),'pools':recovered,
                       'original_results_sha256':hashlib.sha256((original/'results.json').read_bytes()).hexdigest(),
                       'gate_passed':True,'test_access':False})
            run.write_metrics(status='completed',gate_passed=True,recovered_pools=len(recovered),
                              questions=len(c['pools']),test_access=False)
        finally:
            signal.alarm(0);elapsed=time.perf_counter()-start
            run.write_metrics(wall_seconds=elapsed,gpu_hours=elapsed/3600 if source else 0,
                              source_forward_calls=source.forward_calls if source else 0,
                              lane_forwards=engine.lanes if engine else 0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*')
    a=p.parse_args();c=load_config(a.config,a.overrides);c['config_path']=a.config;main(c)

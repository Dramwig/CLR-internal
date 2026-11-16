"""Explicit YAML parallel GPU job runner; aggregate budgets live in each child."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def main(c):
    state={'stage':'waiting','completed':[]};journal=Path(c['journal']);write_json(journal,state)
    while True:
        m=json.loads(Path(c['wait_run'],'metrics.json').read_text())
        if m.get('status')=='completed':break
        if m.get('status')=='failed':raise RuntimeError('Parallel prerequisite failed')
        time.sleep(15)
    procs=[]
    assigned=set()
    for job in c['jobs']:
        if c.get('gpu_candidates'):
            while True:
                result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                free={int(line.split(',')[0]):float(line.split(',')[1])/1024 for line in result.strip().splitlines()}
                choices=[g for g in c['gpu_candidates'] if g not in assigned and free.get(g,0)>=c['minimum_free_gib']]
                if choices:
                    gpu=max(choices,key=lambda g:free[g]);job={**job,'preferred_gpu':job['gpu'],'gpu':gpu,'free_gib_at_assignment':free[gpu]};assigned.add(gpu);break
                state.update(stage='waiting_resources',free_gib=free);write_json(journal,state);time.sleep(20)
        out=Path(job['log']).open('w');env={**os.environ,'CUDA_VISIBLE_DEVICES':str(job['gpu'])};p=subprocess.Popen([sys.executable,job['script'],job['config']],env=env,stdout=out,stderr=subprocess.STDOUT);procs.append((p,out,job));state.update(stage='starting',assigned_jobs=[entry[2] for entry in procs]);write_json(journal,state)
        if c.get('gpu_candidates'):time.sleep(10)
    state.update(stage='running',jobs=c['jobs']);write_json(journal,state)
    while procs:
        for entry in procs[:]:
            p,out,job=entry
            if p.poll() is not None:
                out.close();procs.remove(entry)
                if p.returncode:state.update(stage='failed',error=f'Job failed: {job}');write_json(journal,state);raise RuntimeError(state['error'])
                state['completed'].append(job);write_json(journal,state)
        if procs:time.sleep(15)
    state.update(stage='completed');write_json(journal,state)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Small YAML job queue. Waiting controllers consume no GPU budget."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def main(c):
    state={'stage':'waiting','completed':[]};path=Path(c['journal']);path.parent.mkdir(parents=True,exist_ok=True)
    def wait_metrics(root):
        while True:
            p=Path(root)/'metrics.json'
            if p.exists():
                m=json.loads(p.read_text())
                if m.get('status')=='completed':return
                if m.get('status') in ['failed','interrupted']:raise RuntimeError(f'Prerequisite failed: {root}')
            time.sleep(10)
    try:
        write_json(path,state)
        if c.get('wait_run'):wait_metrics(c['wait_run'])
        for job in c['jobs']:
            state.update(stage='running',current=job);write_json(path,state)
            env={**os.environ,**{k:str(v) for k,v in c.get('env',{}).items()}}
            with Path(job['log']).open('w') as out:
                r=subprocess.run([sys.executable,job['script'],job['config']],env=env,stdout=out,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f"Job exit {r.returncode}: {job}")
            state['completed'].append(job);write_json(path,state)
        state.update(stage='completed',current=None);write_json(path,state)
    except Exception as e:
        state.update(stage='failed',error=str(e));write_json(path,state);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

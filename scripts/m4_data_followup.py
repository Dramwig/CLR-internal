"""Wait for D production completion, then construct the frozen typed questions."""
import argparse,json,time,subprocess,os
from pathlib import Path
from clr.utils.config import load_config

def main(c):
    source=Path(c['source_run']);metrics=source/'metrics.json'
    while not metrics.exists() or json.loads(metrics.read_text()).get('status')=='running':time.sleep(30)
    if json.loads(metrics.read_text()).get('status')!='completed':raise RuntimeError(f'Source production failed: {source}')
    smoke=json.loads(Path(c['qa_smoke_run'],'metrics.json').read_text())
    if smoke.get('status')!='completed':raise ValueError('Question smoke not complete')
    if not Path(c['output'],'qa/summary.json').exists():
        subprocess.run(['.venv/bin/python','scripts/m4_unseen_qa.py',c['source_config'],'stage=unseen_qa','mode=full'],check=True,env={**os.environ,'PYTHONPATH':'.','OMP_NUM_THREADS':'2'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

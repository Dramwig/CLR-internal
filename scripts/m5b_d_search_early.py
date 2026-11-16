"""Run already-trained D search on an available GPU alongside original C."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 root=Path(c['output']);runs=json.loads((root/'D_training_complete.json').read_text());journal=root/'D_pipeline.json'
 state={'stage':'C_full','training_runs':runs,'schedule':'three seeds sequentially on GPU6; original C concurrently; scientific settings unchanged'};write_json(journal,state)
 try:
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(c['gpu'])}
    with Path(c['log']).open('w') as out:r=subprocess.run([sys.executable,'scripts/m5b_bc.py',c['config']],env=env,stdout=out,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('D early C search failed: '+c['log'])
    path=Path(c['log']).read_text().splitlines()[0];met=json.loads(Path(path,'metrics.json').read_text())
    if met['status']!='completed' or not met['gate_passed']:raise RuntimeError('D full protocol gate failed')
    state.update(stage='completed',C_runs=[path]);write_json(journal,state)
 except Exception as e:state.update(stage='failed',error=str(e));write_json(journal,state);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

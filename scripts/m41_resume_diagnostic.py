"""Explicit lower-LR continuation of original architecture, preserving first run."""
import argparse,json,time,subprocess
from pathlib import Path
import yaml
from clr.utils.config import load_config
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
 gate=Path(c.pop('wait_for_gate'))
 while not gate.exists():time.sleep(5)
 prior=json.loads(gate.read_text());c['initial_checkpoint']=str(Path(prior['run'])/'best.pt');dest=Path(c['output'])/'resume_config.yaml';dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(yaml.safe_dump(c,sort_keys=False))
 subprocess.run(['.venv/bin/python','scripts/m41_learnability.py',str(dest)],check=True)

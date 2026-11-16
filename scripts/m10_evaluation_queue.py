"""Sequential final-only M10 evaluator. Requires an immutable training lock."""
import argparse,json,subprocess,sys
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 lock=json.loads(Path(c['eval_lock']).read_text())
 if not lock['all_training_locked']:raise ValueError('Training lock required')
 for j in c['jobs']:
  if Path(j['completion']).exists():
   ref=json.loads(Path(j['completion']).read_text());met=json.loads((Path(ref['run'])/'metrics.json').read_text())
   if met['status']=='completed':continue
   raise ValueError('Incomplete existing final evaluation')
  p=Path(j['log']);attempt=1
  while p.exists():attempt+=1;p=Path(j['log']+'.retry'+str(attempt))
  with p.open('x') as f:subprocess.run([sys.executable,j['script'],j['config']],stdout=f,stderr=subprocess.STDOUT,check=True)
 write_json(c['completion_file'],{'jobs':c['jobs'],'test_opened':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

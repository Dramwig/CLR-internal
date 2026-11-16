"""Use released M10 GPUs for unchanged, not-yet-started training tasks."""
import argparse,json,subprocess,time,fcntl
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 while True:
  if not Path(c['dependency']).exists():time.sleep(15);continue
  ref=json.loads(Path(c['dependency']).read_text())
  status=json.loads((Path(ref['run'])/'metrics.json').read_text())['status'] if 'run' in ref else 'completed'
  if status=='completed':break
  if status=='failed':raise ValueError('Released-GPU dependency failed')
  time.sleep(15)
 first=load_config(load_config(c['queue'])['jobs'][0]['config'])
 # A completed metrics file can precede process teardown by a few moments.
 lease_path=Path(first['output'])/'runtime_locks'/(first['device'].replace(':','_')+'.lock')
 with lease_path.open('a') as lease:
  while True:
   try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:time.sleep(2)
  fcntl.flock(lease,fcntl.LOCK_UN)
 log=Path(c['log'])
 if log.exists():raise FileExistsError(log)
 with log.open('x') as out:subprocess.run(['.venv/bin/python','scripts/m10_train_queue.py',c['queue']],stdout=out,stderr=subprocess.STDOUT,check=True)
 write_json(c['completion'],{'dependency':c['dependency'],'queue':c['queue'],'test_opened':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Preserve the existing GSM worker; distribute only unstarted jobs."""
import argparse,json,os,subprocess,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 run=Path(c['preserved_run']);pid=c['preserved_pid'];groups=c['queues'];children=[]
 for queue in groups:
  log=Path(queue['log'])
  if log.exists():raise FileExistsError(log)
  out=log.open('x');children.append((subprocess.Popen(['.venv/bin/python','scripts/m10_train_queue.py',queue['config']],stdout=out,stderr=subprocess.STDOUT),out))
 # The original CPU controller was stopped without signalling its child.
 # Wait for the original exact-config worker and verify its completion ref.
 while Path(f'/proc/{pid}/cmdline').exists():
  raw=Path(f'/proc/{pid}/cmdline').read_bytes()
  if not raw:break
  if b'm10_train_gsm.py' not in raw:raise ValueError('Preserved PID was reused')
  time.sleep(15)
 if json.loads((run/'metrics.json').read_text())['status']!='completed':raise ValueError('Preserved worker failed')
 log=Path(c['remaining_gpu4_log'])
 if log.exists():raise FileExistsError(log)
 out=log.open('x');children.append((subprocess.Popen(['.venv/bin/python','scripts/m10_train_queue.py',c['remaining_gpu4_queue']],stdout=out,stderr=subprocess.STDOUT),out))
 for child,out in children:
  status=child.wait();out.close()
  if status:raise RuntimeError('GSM partition failed: '+str(status))
 original=load_config(c['original_queue'])
 for j in original['jobs']:
  cfg=load_config(j['config']);ref=Path(cfg['output'])/'trained'/f'{cfg["case"]}_{cfg["seed"]}_full.json';r=json.loads(ref.read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text())
  if m['status']!='completed':raise ValueError('Incomplete original matrix')
 write_json('data/M10_v1/gsm_training_completed.json',{'jobs':original['jobs'],'scheduling_supplement':c['supplement'],'test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

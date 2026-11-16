"""Wait for all selected training, lock it, then final evaluation and CPU exports."""
import argparse,json,subprocess,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import live

def execute(python,script,config,log):
 p=Path(log)
 attempt=1
 while p.exists():
  attempt+=1;p=Path(log+'.retry'+str(attempt))
 with p.open('x') as f:subprocess.run([python,script,config],stdout=f,stderr=subprocess.STDOUT,check=True)
def ready(c):
 for p in c['completion_gates']:
  if not Path(p).exists():return False
 ref=json.loads(Path('data/M10_v1/trained/transfer_full.json').read_text());m=json.loads((Path(ref['run'])/'metrics.json').read_text())
 return m['status']=='completed'
def main(c):
 while not ready(c):
  live(stage='training_before_final_lock',test_opened=False);write_json('data/M10_v1/final_controller_progress.json',{'stage':'waiting_for_all_training','test_opened':False});time.sleep(30)
 lock_path=Path(c['lock_config']).read_text() if False else Path(load_config(c['lock_config']).get('output','data/M10_v1'))/'final_lock.json'
 if lock_path.exists():
  live(stage='locked_final_evaluation',test_opened=False)
 else:
  execute('.venv/bin/python','scripts/m10_lock.py',c['lock_config'],'reports/M10_evidence/final_lock.log');live(stage='locked_final_evaluation',test_opened=False)
 jobs=[]
 for script,config,log in c['evaluation_jobs']:
  p=Path(log);attempt=1
  while p.exists():
   attempt+=1;p=Path(log+'.retry'+str(attempt))
  out=p.open('x');jobs.append((subprocess.Popen(['.venv/bin/python',script,config],stdout=out,stderr=subprocess.STDOUT),out))
 # Source needs the communication environment. Source test generation and its
 # reader evaluation are sequential; all model/checkpoint choices were locked.
 execute('.venv-comm/bin/python','scripts/m10_source_hiddenbench.py',c['hidden_source'],'reports/M10_evidence/hiddenbench_source_test.log')
 execute('.venv/bin/python','scripts/m10_evaluate_hiddenbench.py',c['hidden_eval'],'reports/M10_evidence/hiddenbench_eval_full.log')
 for p,out in jobs:
  rc=p.wait();out.close()
  if rc:raise RuntimeError('Final evaluator failed with exit '+str(rc))
 write_json('data/M10_v1/all_evaluation_completed.json',{'test_opened':True,'status':'final_evaluation_completed'})
 for script,config,log in c['exports']:execute('.venv/bin/python',script,config,log)
 write_json('data/M10_v1/exports_completed.json',{'model_experiments_complete':True,'paper_reports_freeze_pending':True});live(stage='exports_complete_paper_pending',test_opened=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

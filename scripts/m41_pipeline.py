"""Persistent gated continuation, never treating missing gates as approval."""
import argparse,json,time,subprocess
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def execute(script,config):subprocess.run(['.venv/bin/python',script,str(config)],check=True)
def find_run(stage,case):
 found=[]
 for p in Path('runs').glob('*/config.json'):
  r=json.loads(p.read_text())
  if r.get('milestone')=='M4.1' and r.get('stage')==stage and r.get('case')==case and r.get('mode')=='smoke':found.append(p.parent)
 if len(found)!=1:raise ValueError('Ambiguous/missing smoke run '+case)
 if json.loads((found[0]/'metrics.json').read_text())['status']!='completed':raise ValueError('Smoke incomplete')
 return str(found[0])
def main(c):
 root=Path(c['output']);gate=root/'gate_lock.json';start=time.monotonic()
 while not gate.exists():
  if time.monotonic()-start>c['wait_seconds']:raise TimeoutError('M4.1 gate timeout, no main experiment started')
  time.sleep(10)
 if not json.loads(gate.read_text()).get('both_variants_passed'):raise ValueError('Candidate learnability failed; C remains blocked')
 configs=root/'pipeline_configs';configs.mkdir(exist_ok=True);tr=load_config(c['training_config']);ev=load_config(c['evaluation_config']);smokes={}
 if c.get('resume_smokes'):
  smokes=json.loads(Path(c['resume_smokes']).read_text())
  for directory in smokes.values():
   m=json.loads(Path(directory,'metrics.json').read_text())
   if m['status']!='completed' or m.get('test_opened'):raise ValueError('Invalid resumed val smoke')
 for v in ([] if c.get('resume_smokes') else ['v1','lite','b3_multi']):
  tc={**tr,'variant':v,'mode':'smoke','case':'C_smoke','reader_device':'cuda:1','batch_size':256 if v=='v1' else 1024,'eval_batch':256 if v=='v1' else 1024,'epochs':1,'min_epochs':1,'run_budget_hours':.25};p=configs/f'train_smoke_{v}.yaml';p.write_text(yaml.safe_dump(tc,sort_keys=False));execute('scripts/m41_train.py',p)
  trainrun=json.loads((root/f'trained_C_smoke_{v}_11.json').read_text())['run']
  if v=='b3_multi':continue
  ec={**ev,'variant':v,'mode':'smoke','case':f'eval_smoke_{v}','checkpoint':str(Path(trainrun)/'best.pt'),'reader_device':'cuda:1','model_kind':'clr','eval_batch':256 if v=='v1' else 1024,'run_budget_hours':.25};p=configs/f'eval_smoke_{v}.yaml';p.write_text(yaml.safe_dump(ec,sort_keys=False));execute('scripts/m41_evaluate.py',p);smokes[v]=find_run('evaluate',ec['case'])
 write_json(root/'pipeline_smokes.json',smokes);execute('scripts/m41_train_suite.py',c['training_config'])
 ev['evaluation_smokes']=smokes;p=configs/'evaluation_full.yaml';p.write_text(yaml.safe_dump(ev,sort_keys=False));execute('scripts/m41_eval_suite.py',p)
 execute('scripts/m41_report.py',c['report_config']);execute('scripts/m41_finish.py',c['report_config'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
 try:main(load_config(a.config,a.overrides))
 except Exception as e:
  write_json(Path('docs/M41_LIVE.json'),{'stage':'blocked_error','error':f'{type(e).__name__}: {e}','review':'reports/M4.1_REVIEW.md'})
  Path('docs/BLOCKERS.md').open('a').write(f'\n## M4.1 自动接续暂停\n{type(e).__name__}: {e}；详细日志reports/M41_pipeline.log。未静默更改预算或跳过关卡。\n');raise

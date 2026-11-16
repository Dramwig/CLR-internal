"""Separate M10 sub-budget ledgers; no old runtime state is changed."""
import hashlib,json,shutil,time,fcntl
from datetime import datetime,timezone
from pathlib import Path
from clr.utils.idea_a import write_json

def ledger(exclude=None):
 rows=[]
 for p in Path('runs').glob('*/config.json'):
  c=json.loads(p.read_text())
  if c.get('milestone')!='M10' or (exclude and p.parent.resolve()==Path(exclude).resolve()):continue
  m=json.loads((p.parent/'metrics.json').read_text()) if (p.parent/'metrics.json').exists() else {}
  b=json.loads((p.parent/'budget.json').read_text()) if (p.parent/'budget.json').exists() else {}
  rows.append(dict(run=str(p.parent),section=c.get('section'),stage=c.get('stage'),gpu_hours=max(m.get('gpu_hours',0),b.get('own_gpu_hours',0)),status=m.get('status'),source_forward_calls=m.get('source_forward_calls')))
 return rows

def snapshot(run,c):
 paths=set(Path('clr').rglob('*.py'))|set(Path('scripts').glob('m10*.py'))|set(Path('configs/M10').rglob('*.yaml'))|{Path('docs/M10_TASK.md')}
 if Path('tests/test_m10.py').exists():paths.add(Path('tests/test_m10.py'))
 hashes={}
 for p in sorted(paths):
  dst=run.path/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 write_json(run.path/'source_sha256.json',hashes)

def live(**kw):
 p=Path('docs/M10_LIVE.json');d=json.loads(p.read_text()) if p.exists() else {}
 write_json(p,{**d,**kw,'updated_utc':datetime.now(timezone.utc).isoformat()})

class Budget:
 def __init__(self,run,c):
  self.run,self.c,self.start=run,c,time.monotonic()
  self.test_opened=False;self.source_calls=0
  folder=Path('data/M10_v1/runtime_locks');folder.mkdir(exist_ok=True)
  self.lease=(folder/(c['device'].replace(':','_')+'.lock')).open('a')
  try:fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('M10 GPU already reserved by an active run: '+c['device'])
  write_json(run.path/'gpu_lease.json',{'device':c['device'],'pid':__import__('os').getpid(),'run':str(run.path)})
  snapshot(run,c)
 def check(self,status='running',**kw):
  own=(time.monotonic()-self.start)/3600;rs=ledger(self.run.path);total=sum(r['gpu_hours'] for r in rs)+own;section=sum(r['gpu_hours'] for r in rs if r['section']==self.c['section'])+own
  write_json(self.run.path/'budget.json',dict(own_gpu_hours=own,total_gpu_hours=total,section_gpu_hours=section))
  self.test_opened=self.test_opened or kw.pop('test_opened',False)
  self.source_calls=kw.pop('source_forward_calls',self.source_calls)
  if self.test_opened and not (self.run.path/'test_access.json').exists():write_json(self.run.path/'test_access.json',{'test_opened':True,'UTC':datetime.now(timezone.utc).isoformat()})
  payload=dict(status=status,gpu_hours=own,source_forward_calls=self.source_calls,test_opened=self.test_opened)
  payload.update(kw)
  self.run.write_metrics(**payload)
  if total>=40-.025 or section>=self.c['section_budget']-.025:raise TimeoutError('M10 GPU hard budget guard reached')
  if self.c.get('mode')=='smoke' and own>.25:raise TimeoutError('M10 smoke exceeds 15 minutes')
  return own

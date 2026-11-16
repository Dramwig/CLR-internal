"""Independent M10b budget and run ledger; never reads or mutates M10 freeze."""
import hashlib,json,shutil,time,fcntl,os
from datetime import datetime,timezone
from pathlib import Path
from clr.utils.idea_a import write_json
ROOT=Path('data/M10b_v1')

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ledger(exclude=None):
 out=[]
 for p in Path('runs').glob('*/config.json'):
  c=json.loads(p.read_text())
  if c.get('milestone')!='M10b' or (exclude and p.parent.resolve()==Path(exclude).resolve()): continue
  m=json.loads((p.parent/'metrics.json').read_text()) if (p.parent/'metrics.json').exists() else {}
  b=json.loads((p.parent/'budget.json').read_text()) if (p.parent/'budget.json').exists() else {}
  charge=max(float(m.get('gpu_hours',0)),float(b.get('own_gpu_hours',0)))
  if m.get('status')=='failed' and not charge: charge=.001 # conservative 3.6s charge for pre-model snapshot failure
  out.append({'run':str(p.parent),'gpu_hours':charge, 'status':m.get('status'), 'section':c.get('section')})
 return out

def snapshot(run,c):
 paths=[Path('scripts/m10b_train_b5c.py'),Path('clr/models/m10_b5c.py'),Path('clr/data/m10.py'),Path('clr/utils/m10b_runtime.py')]
 d={}
 for p in paths:
  if not p.exists(): continue
  dst=run.path/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);d[str(p)]=_sha(p)
 write_json(run.path/'source_sha256.json',d)

class Budget:
 def __init__(self,run,c):
  self.run,self.c,self.start=run,c,time.monotonic();self.test_opened=False
  ROOT.joinpath('runtime_locks').mkdir(parents=True,exist_ok=True)
  self.lease=(ROOT/'runtime_locks'/(c['device'].replace(':','_')+'.lock')).open('a')
  try: fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError: raise RuntimeError('M10b GPU already reserved: '+c['device'])
  write_json(run.path/'gpu_lease.json',{'device':c['device'],'pid':os.getpid(),'run':str(run.path)})
  snapshot(run,c)
 def check(self,status='running',**kw):
  own=(time.monotonic()-self.start)/3600
  rs=ledger(self.run.path); total=sum(x['gpu_hours'] for x in rs)+own; section=sum(x['gpu_hours'] for x in rs if x.get('section')==self.c.get('section'))+own
  self.test_opened=self.test_opened or bool(kw.pop('test_opened',False))
  write_json(self.run.path/'budget.json',{'own_gpu_hours':own,'total_gpu_hours':total,'section_gpu_hours':section,'hard_budget_gpu_hours':22.0})
  payload={'status':status,'gpu_hours':own,'test_opened':self.test_opened};payload.update(kw);runmetrics=self.run.write_metrics(**payload)
  if total>=22-.01 or section>=22-.01: raise TimeoutError('M10b GPU hard budget reached')
  if self.c.get('mode')=='smoke' and own>.25: raise TimeoutError('M10b smoke exceeded 15 minutes')
  return own

def live(**kw):
 p=Path('docs/M10b_LIVE.json');d=json.loads(p.read_text()) if p.exists() else {};write_json(p,{**d,**kw,'updated_utc':datetime.now(timezone.utc).isoformat()})

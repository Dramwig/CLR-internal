"""Audit fixed-seed matrices/exposure and immutable final artifacts."""
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main(c):
 root=Path(c['output']);freeze=json.loads((root/'freeze.json').read_text());lock=json.loads((root/'final_lock.json').read_text());errors=[]
 def check(item):
  p,h=item
  try:return None if sha(p)==h else p
  except Exception:return p
 with ThreadPoolExecutor(max_workers=2) as pool:errors.extend(p for p in pool.map(check,freeze['sha256'].items()) if p)
 for p,h in lock['checkpoint_sha256'].items():
  if sha(p)!=h:errors.append('locked checkpoint '+p)
 expected=[]
 for name in ['gsm','audit']:
  queue=load_config(f'configs/M10/full_{name}_queue.yaml')
  for j in queue['jobs']:
   tc=load_config(j['config']);path=root/'evaluated'/f'{tc["case"]}_{tc["seed"]}_full.json';expected.append(str(path))
   if not path.exists():errors.append('missing eval '+str(path))
   else:
    ref=json.loads(path.read_text());m=json.loads((Path(ref['run'])/'metrics.json').read_text())
    if m['status']!='completed' or not m.get('test_opened'):errors.append('nonfinal eval '+str(path))
 for name in ['transfer_full','hiddenbench']:
  path=root/'evaluated/transfer_full.json' if name=='transfer_full' else root/'hiddenbench/evaluation.json';expected.append(str(path))
  if not path.exists():errors.append('missing eval '+str(path))
  else:
   ref=json.loads(path.read_text());m=json.loads((Path(ref['run'])/'metrics.json').read_text())
   if m['status']!='completed' or not m.get('test_opened'):errors.append('nonfinal eval '+str(path))
 for p in (root/'trained').glob('*CLR-per-type*full.json'):
  r=json.loads(p.read_text());c=r['config'];target=json.loads(Path(c['reference_exposure']).read_text())[c['source']][str(c['seed'])]['exposure']
  for q,cp in r['checkpoints'].items():
   if cp['selected']['exposure'][q]!=target[q]:errors.append('wrong type exposure '+str(p)+q)
 for p in (root/'trained').glob('audit_B6*full.json'):
  r=json.loads(p.read_text())
  for t,v in r['parameters'].items():
   if v['parameters']<v['B3_parameters']:errors.append('B6 params '+str(p)+t)
 write_json(root/'completion_audit.json',{'passed':not errors,'errors':errors,'frozen_files':len(freeze['sha256']),'final_evaluation_references':expected,'checkpoint_hashes':len(lock['checkpoint_sha256']),'freeze_sha256':sha(root/'freeze.json')})
 print(len(freeze['sha256']),'frozen files;',len(errors),'errors')
 if errors:raise ValueError('Completion audit failed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

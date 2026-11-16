"""Freeze all selected training and explicit omissions before any M10 test."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import ledger

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(c):
 root=Path(c['output']);dest=root/'final_lock.json'
 if dest.exists():raise FileExistsError('Final lock cannot be overwritten')
 for p in c['completion_gates']:
  if not Path(p).exists():raise ValueError('Training gate incomplete: '+p)
 refs=[]
 for cfg in load_config(c['gsm_queue'])['jobs']+load_config(c['audit_queue'])['jobs']:
  tc=load_config(cfg['config']);refs.append(root/'trained'/f'{tc["case"]}_{tc["seed"]}_full.json')
 refs+=[root/'trained/transfer_full.json',root/'hiddenbench/training.json'];checkpoints={};hashes={};run_status={}
 for p in refs:
  ref=json.loads(p.read_text());run=Path(ref['run']);m=json.loads((run/'metrics.json').read_text())
  if m['status']!='completed' or m.get('test_opened',False):raise ValueError('Training not complete/isolated: '+str(p))
  hashes[str(p)]=sha(p);run_status[str(run)]=m
  for cp in ref['checkpoints'].values():
   h=sha(cp['path'])
   if h!=cp['sha256']:raise ValueError('Training checkpoint hash changed')
   checkpoints[cp['path']]=h
 costs=ledger();budget={s:sum(r['gpu_hours'] for r in costs if r['section']==s) for s in c['limits']}
 for entry in costs:
  run=Path(entry['run']);metrics=json.loads((run/'metrics.json').read_text())
  if metrics.get('test_opened') or (run/'test_access.json').exists():raise ValueError('M10 test was accessed before the final lock: '+str(run))
  if entry['status'] not in ['completed','failed']:raise ValueError('GPU run still active before final lock: '+str(run))
 for s,limit in c['limits'].items():
  if budget[s]+c['evaluation_reserve'][s]>=limit:raise RuntimeError(f'Final evaluation admission fails {s}: spent={budget[s]}, reserve={c["evaluation_reserve"][s]}')
 omission=json.loads((root/'B5c_budget_admission.json').read_text())
 write_json(dest,{'all_training_locked':True,'UTC':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':checkpoints,'training_refs_sha256':hashes,'training_status':run_status,'budget_GPUh_before_test':budget,'evaluation_reserve_GPUh':c['evaluation_reserve'],'limits':c['limits'],'B5c':'NR: full training exceeds assigned 12GPUh; no scientific protocol reduction','B5c_admission_sha256':sha(root/'B5c_budget_admission.json'),'hiddenbench_protocol_sha256':sha(root/'hiddenbench/protocol.json'),'hiddenbench_public_protocol_sha256':sha(root/'hiddenbench/protocol_public_v2.json'),'transfer_panels':'nested labels; N100/300/1000/full; all seeds fixed','test_opened_before_lock':False})
 print(len(checkpoints),'checkpoints locked')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

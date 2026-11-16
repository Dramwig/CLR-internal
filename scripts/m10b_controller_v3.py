"""Sequential fixed full-data/three-seed pipeline with pre-test checkpoint lock."""
import argparse,json,subprocess,hashlib,time,os
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10b_runtime import ledger,live

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(c):
 root=Path(c['output']); device=c['execution_device']; used=lambda:sum(r['gpu_hours'] for r in ledger())
 if not json.loads((root/'budget_admission.json').read_text())['passed']: raise ValueError('Budget admission failed')
 for seed in c['seeds']:
  ref=root/'trained'/f'B5c_gpt2_{seed}_full.json'
  if not ref.exists():
   live(stage='training',seed=seed,test_opened=False)
   with (root/f'train_{seed}.log').open('a') as log:
    subprocess.run([c['python'],'-u','scripts/m10b_train_b5c.py',f'configs/M10b/B5c_gpt2_{seed}.yaml',f'device={device}','min_free_GiB=0'],stdout=log,stderr=subprocess.STDOUT,check=True)
  r=json.loads(ref.read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text());mf=json.loads((Path(r['run'])/'training_manifest.json').read_text())
  if m['status']!='completed' or mf['counts']!={'Q3':860228,'Q4':279440} or sha(r['checkpoint'])!=r['sha256']:raise ValueError('Training integrity failure')
  history=json.loads((Path(r['run'])/'history.json').read_text())
  if history[-1]['exposure']!=mf['counts']:raise ValueError('Full one-pass exposure missing')
  remaining=sum(not (root/'trained'/f'B5c_gpt2_{other}_full.json').exists() for other in c['seeds'])
  projected=used()+remaining*max(json.loads((Path(json.loads(p.read_text())['run'])/'metrics.json').read_text())['gpu_hours'] for p in (root/'trained').glob('B5c_gpt2_*_full.json'))+c['evaluation_reserve_GPUh']
  write_json(root/'projection_live.json',{'used_GPUh':used(),'projected_total_GPUh':projected,'completed_seed':seed,'remaining_seeds':remaining,'evaluation_reserve_GPUh':c['evaluation_reserve_GPUh']})
  if projected>22:live(stage='stopped_budget_projection',projection=projected,test_opened=False);raise RuntimeError('M10b projected full cost exceeds 22 GPUh; no seed/labels reduced')
 cps={};refs={}
 for seed in c['seeds']:
  p=root/'trained'/f'B5c_gpt2_{seed}_full.json';r=json.loads(p.read_text());cps[r['checkpoint']]=sha(r['checkpoint']);refs[str(p)]=sha(p)
 if not (root/'final_lock.json').exists():write_json(root/'final_lock.json',{'all_training_locked':True,'checkpoint_sha256':cps,'training_refs_sha256':refs,'seeds':c['seeds'],'qtypes':['Q3','Q4'],'epochs':1,'test_opened_before_lock':False,'budget_GPUh_before_test':used(),'evaluation_reserve_GPUh':c['evaluation_reserve_GPUh'],'hard_limit_GPUh':22,'LLaMA_and_audit':'excluded by user scope'})
 for seed in c['seeds']:
  if not (root/'evaluated'/f'B5c_gpt2_{seed}_full.json').exists():
   live(stage='final_evaluation',seed=seed)
   with (root/f'eval_{seed}.log').open('a') as log:subprocess.run([c['python'],'-u','scripts/m10b_evaluate_b5c.py',f'configs/M10b/eval_{seed}.yaml',f'device={device}'],stdout=log,stderr=subprocess.STDOUT,check=True)
 write_json(root/'experiments_completed.json',{'completed':True,'GPUh':used(),'seeds':c['seeds']});live(stage='experiments_completed',test_opened=True)
 if c.get('finish_config'):
  subprocess.run([c['python'],'-u','scripts/m10b_finish.py',c['finish_config']],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

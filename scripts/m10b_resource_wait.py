"""CPU-only resource wait; restore exact scientific protocol once memory is available."""
import argparse,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10b_runtime import live,ledger

def main(c):
 root=Path(c['output']);stable={};live(stage='waiting_gpu_memory',seed=33,test_opened=False)
 while True:
  if (root/'freeze.json').exists() or (root/'budget_stopped.json').exists():return
  result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  gpus=[tuple(map(int,line.split(','))) for line in result.stdout.strip().splitlines()]
  for idx,free,util in gpus:stable[idx]=stable.get(idx,0)+1 if free>=c['minimum_free_MiB'] else 0
  eligible=[(idx,free,util) for idx,free,util in gpus if stable[idx]>=c['stable_checks']]
  observation={'stage':'waiting_gpu_memory','GPUh_used':sum(r['gpu_hours'] for r in ledger()),'minimum_free_MiB':c['minimum_free_MiB'],'gpus':[{'index':i,'free_MiB':f,'utilization':u,'stable_checks':stable[i]} for i,f,u in gpus],'UTC':datetime.now(timezone.utc).isoformat(),'wait_is_CPU_only':True}
  write_json(root/'resource_wait.json',observation)
  if eligible:
   device='cuda:'+str(max(eligible,key=lambda g:(g[1],-g[2]))[0]);cfg=load_config(c['controller_config']);cfg['execution_device']=device
   import yaml
   launch=root/'resource_selected_controller.yaml';launch.write_text(yaml.safe_dump(cfg,sort_keys=False));write_json(root/'execution_resume_v2.json',{**observation,'device':device,'scientific_changes':False,'reason':'seed33 GPU3 external memory contention OOM; resume only failed seed, preserve completed11/22 and all failed costs'})
   live(stage='resuming_training',seed=33,execution_device=device,test_opened=False)
   with (root/'controller_resume_v2.log').open('a') as log:subprocess.run([c['python'],'-u','scripts/m10b_controller_v2.py',str(launch)],stdout=log,stderr=subprocess.STDOUT,check=True)
   return
  time.sleep(c['poll_seconds'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Wait for startup memory only; running tasks have no free-memory stop guard."""
import argparse,fcntl,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10b_runtime import live,ledger


def eligible_devices(gpus,stable,minimum,checks):
 for idx,free,util in gpus:stable[idx]=stable.get(idx,0)+1 if free>=minimum else 0
 return [g for g in gpus if stable[g[0]]>=checks]


def main(c):
 root=Path(c['output']);stable={};attempt=0
 with (root/'startup_monitor.lock').open('a') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   if (root/'freeze.json').exists() or (root/'budget_stopped.json').exists():return
   costs=ledger();used=sum(r['gpu_hours'] for r in costs)
   test_already_opened=any(json.loads((Path(r['run'])/'metrics.json').read_text()).get('test_opened',False) for r in costs if (Path(r['run'])/'metrics.json').exists())
   completed=[]
   for p in (root/'trained').glob('B5c_gpt2_*_full.json'):
    ref=json.loads(p.read_text());completed.append(json.loads((Path(ref['run'])/'metrics.json').read_text())['gpu_hours'])
   missing=sum(not (root/'trained'/f'B5c_gpt2_{seed}_full.json').exists() for seed in [11,22,33])
   projected=used+missing*max(completed)+c['evaluation_reserve_GPUh']
   if projected>22:
    live(stage='stopped_budget_projection',projection=projected,test_opened=test_already_opened)
    write_json(root/'startup_monitor_stopped.json',{'reason':'remaining full protocol projected above22GPUh','used_GPUh':used,'projected_GPUh':projected})
    return
   result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
   if result.returncode:
    write_json(root/'startup_query_error.json',{'stderr':result.stderr,'UTC':datetime.now(timezone.utc).isoformat()});time.sleep(c['poll_seconds']);continue
   gpus=[tuple(map(int,line.split(','))) for line in result.stdout.strip().splitlines()]
   eligible=eligible_devices(gpus,stable,c['minimum_free_MiB'],c['stable_checks'])
   observation={'stage':'waiting_startup_memory','GPUh_used':used,'projected_GPUh':projected,'startup_minimum_free_MiB':c['minimum_free_MiB'],'runtime_free_memory_guard':False,'gpus':[{'index':i,'free_MiB':f,'utilization':u,'stable_checks':stable[i]} for i,f,u in gpus],'UTC':datetime.now(timezone.utc).isoformat(),'wait_is_CPU_only':True}
   write_json(root/'resource_wait_v2.json',observation)
   live(stage='waiting_startup_memory',seed=33,current_cost_GPUh=used,current_projected_GPUh=projected,runtime_free_memory_guard=False,test_opened=test_already_opened)
   if not eligible:time.sleep(c['poll_seconds']);continue
   device='cuda:'+str(max(eligible,key=lambda g:(g[1],-g[2]))[0]);cfg=load_config(c['controller_config']);cfg['execution_device']=device
   import yaml
   attempt+=1;launch=root/f'monitored_launch_{attempt}_{time.time_ns()}.yaml';launch.write_text(yaml.safe_dump(cfg,sort_keys=False))
   before={r['run'] for r in costs}
   write_json(root/'startup_monitor_launch.json',{**observation,'device':device,'controller_config':str(launch),'min_free_GiB_during_training':0})
   live(stage='startup_monitor_launched',execution_device=device,runtime_free_memory_guard=False)
   # Deliberately block here. No GPU-memory polling or memory-based signals
   # occur while the child runs; controller_v3 forces min_free_GiB=0.
   with (root/'controller_monitored.log').open('a') as log:
    child=subprocess.run([c['python'],'-u','scripts/m10b_controller_v3.py',str(launch)],stdout=log,stderr=subprocess.STDOUT)
   if child.returncode==0:return
   new=[r for r in ledger() if r['run'] not in before]
   failures=[json.loads((Path(r['run'])/'metrics.json').read_text()) for r in new if (Path(r['run'])/'metrics.json').exists()]
   if failures and failures[-1].get('error_type')=='OutOfMemoryError':
    live(stage='actual_OOM_returning_to_startup_monitor',runtime_free_memory_guard=False)
    stable={};time.sleep(c['poll_seconds']);continue
   live(stage='controller_failed_non_OOM',controller_returncode=child.returncode)
   return

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

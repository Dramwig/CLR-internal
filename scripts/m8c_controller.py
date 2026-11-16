"""Run the fixed test matrix; scheduling decisions depend only on time and cost."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from clr.audit.m8c_runtime import snapshot,Cost,total_cost,live
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def find_run(mode):
    found=[]
    for p in Path('runs').glob('*/config.json'):
        c=json.loads(p.read_text())
        if c.get('milestone')=='M8c' and c.get('mode')==mode:found.append(p.parent)
    return max(found,key=lambda p:p.name) if found else None

def terminate(jobs):
    for job in jobs:
        if job['process'].poll() is None:job['process'].send_signal(signal.SIGINT)
    deadline=time.monotonic()+30
    while time.monotonic()<deadline and any(j['process'].poll() is None for j in jobs):time.sleep(.25)
    for job in jobs:
        if job['process'].poll() is None:job['process'].kill()
        job['process'].wait();job['handle'].close()

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output'])
        lock=json.loads((root/'final_lock.json').read_text());assert lock['n']==200 and lock['budget_gpu_hours']==21
        jobs=[];records=[];primary_complete=False
        try:
            for job_config in c['primary_configs']:
                cfg=load_config(job_config);log=Path(c['log_folder'])/(cfg['mode']+'.log');handle=log.open('x')
                process=subprocess.Popen([sys.executable,'-u','-m','scripts.m8c_evaluate',job_config],stdout=handle,stderr=subprocess.STDOUT)
                jobs.append({'process':process,'handle':handle,'config':job_config,'mode':cfg['mode'],'run':None,'device':cfg['device']})
            while True:
                time.sleep(c['poll_seconds']);rates=[];done=[];status=[]
                for job in jobs:
                    if job['run'] is None:job['run']=find_run(job['mode'])
                    p=job['run'];timing=p/'timings.jsonl' if p else None
                    ts=[json.loads(l) for l in timing.read_text().splitlines()] if timing and timing.exists() else []
                    n=len(ts);done.append(n)
                    # A completed-prefix mean is shrunk toward the pre-test val estimate.
                    prior=c['prior_equivalent_groups'];rate=(sum(t['seconds'] for t in ts)/3600+prior*c['prior_gpu_hours_per_seed_group'])/(n+prior)
                    rates.append(rate)
                    status.append({'mode':job['mode'],'run':str(p) if p else None,'pid':job['process'].pid,'device':job['device'],'groups':n,'rate_gpu_hours':rate,'exit_code':job['process'].poll()})
                spent=total_cost();forecast=spent+sum(rate*(200-n) for rate,n in zip(rates,done))+c['forecast_shutdown_reserve_gpu_hours']
                state={'primary':status,'spent_gpu_hours':spent,'forecast_total_gpu_hours':forecast,'forecast_uses':'only elapsed group durations and pre-test val prior, no QA/attack outcomes'}
                write_json(root/'controller_progress.json',state);live(stage='primary_test_running',test_opened=any(p['run'] for p in status),background_tasks_active=True,primary_groups=done,gpu_hours_conservative=spent,primary_forecast_gpu_hours=forecast)
                print('primary groups',done,'spent',round(spent,3),'forecast',round(forecast,3),flush=True)
                errors=[j for j in jobs if j['process'].poll() not in [None,0]]
                if errors:raise RuntimeError('Primary child failed: '+str([j['mode'] for j in errors]))
                if all(j['process'].poll()==0 for j in jobs):
                    assert done==[200,200,200];primary_complete=True;break
                if min(done)>=c['forecast_min_groups'] and forecast>c['budget_gpu_hours']:
                    write_json(root/'primary_forecast_stop.json',state);raise RuntimeError('Primary fixed200 forecast exceeds21 GPUh; stopped without reducing groups or epsilon')
            for job in jobs:job['handle'].close()
            # Priorities and all 100 secondary IDs were locked before first test access.
            summary={};details=[]
            for job_config in c['secondary_configs']:
                cfg=load_config(job_config);mode=cfg['mode'];remaining=c['budget_gpu_hours']-total_cost()
                if remaining<=c['secondary_start_reserve_gpu_hours']:
                    summary[mode]='未执行：剩余预算不足启动预留';continue
                log=Path(c['log_folder'])/(mode+'.log')
                with log.open('x') as handle:
                    child=subprocess.Popen([sys.executable,'-u','-m',c['secondary_modules'][mode],job_config],stdout=handle,stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        time.sleep(c['poll_seconds']);live(stage=mode,test_opened=True,background_tasks_active=True,gpu_hours_conservative=total_cost())
                p=find_run(mode);timing=p/'timings.jsonl';complete=len(timing.read_text().splitlines()) if timing.exists() else 0
                detail={'mode':mode,'run':str(p),'completed_groups':complete,'planned_groups':100,'exit_code':child.returncode,'gpu_hours_after':total_cost()};details.append(detail)
                summary[mode]=f'{complete}/100完整组'+('，完成' if child.returncode==0 else '，预算停止或失败；部分行保留')
                if mode=='minimum_test':write_json(root/'minimum_available.json',{'run':str(p),'groups':complete,'planned_groups':100})
                if child.returncode!=0:
                    trace=(p/'traceback.txt').read_text() if (p/'traceback.txt').exists() else log.read_text()
                    if 'budget shutdown reserve reached' not in trace and 'wall-time limit' not in trace:raise RuntimeError('Unexpected secondary failure: '+str(p))
            write_json(root/'secondary_status.json',{'summary':summary,'details':details,'execution_order':['minimum_test','test_blind','test_sensitivity'],'primary_complete':True})
            write_json(root/'controller_complete.json',{'run':str(run.path),'primary_complete':True,'gpu_hours':total_cost(),'secondary':summary})
            cost.update(status='completed');live(stage='primary_and_budgeted_secondary_completed',background_tasks_active=False,test_opened=True,primary_complete=True)
        except BaseException:
            terminate(jobs);cost.update(status='failed');live(stage='primary_budget_or_runtime_stop',background_tasks_active=False,milestone_complete=False,primary_complete=primary_complete)
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

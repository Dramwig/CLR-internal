"""Budget-ordered secondary work after exact primary continuation/merge."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time
from pathlib import Path
from clr.audit.m8c_runtime import snapshot,Cost,total_cost,live
from scripts.m8c_controller import find_run
from clr.utils.records import recorded
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output'])
        for seed in [11,22,33]:
            ref=json.loads((root/f'final_test_seed{seed}_part0.json').read_text());assert ref['groups']==200
        summary={};details=[]
        for config in c['secondary_configs']:
            cfg=load_config(config);mode=cfg['mode']
            if 21-total_cost()<=.08:summary[mode]='未执行：预算余量不足启动';continue
            log=Path(c['log_folder'])/(mode+'_continued.log')
            with log.open('x') as f:
                child=subprocess.Popen([sys.executable,'-u','-m',c['secondary_modules'][mode],config],stdout=f,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    time.sleep(10);live(stage=mode,test_opened=True,background_tasks_active=True,gpu_hours_conservative=total_cost())
            p=find_run(mode);timing=p/'timings.jsonl';n=len(timing.read_text().splitlines()) if timing.exists() else 0
            details.append({'mode':mode,'run':str(p),'completed_groups':n,'planned_groups':100,'exit_code':child.returncode})
            summary[mode]=f'{n}/100完整组'+('，完成' if child.returncode==0 else '，预算停止或资源失败，部分结果保留')
            if mode=='minimum_test':write_json(root/'minimum_available.json',{'run':str(p),'groups':n,'planned_groups':100})
            if child.returncode!=0:
                for later in c['secondary_configs'][c['secondary_configs'].index(config)+1:]:summary[load_config(later)['mode']]='未执行：前序未完成或预算耗尽'
                break
        write_json(root/'secondary_status.json',{'summary':summary,'details':details,'execution_order':['minimum_test','test_blind','test_sensitivity'],'primary_complete':True})
        write_json(root/'controller_complete.json',{'run':str(run.path),'primary_complete':True,'gpu_hours':total_cost(),'secondary':summary,'continued_after_single_run_timeout':True})
        cost.update(status='completed');live(stage='primary_and_budgeted_secondary_completed',primary_complete=True,test_opened=True,background_tasks_active=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

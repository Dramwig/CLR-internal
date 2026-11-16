"""CPU-only, val-derived M8c feasibility gate; never loads test examples or IDs."""
import sys
sys.dont_write_bytecode=True
import argparse,json,math,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import nnls
from clr.audit.m8c_runtime import snapshot,Cost,total_cost,live
from clr.audit.features import sha256
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def throughput(rows_by_group,seconds,epsilons):
    """Fit nonnegative fixed/D0-step/R4-step costs, then remove dropped work."""
    original=[];planned=[];ys=[]
    for i,rows in enumerate(rows_by_group,1):
        original.append([1,*[sum(r['steps'] for r in rows if r['attack_design']==d) for d in ['D0','R4']]])
        planned.append([1,*[sum(r['steps'] for r in rows if r['attack_design']==d and r['epsilon'] in epsilons) for d in ['D0','R4']]])
        ys.append(seconds[i])
    x=np.asarray(original,dtype=float);y=np.asarray(ys,dtype=float);p=np.asarray(planned,dtype=float)
    coefficient,_=nnls(x,y);prediction=p@coefficient
    return coefficient,prediction,y-x@coefficient

def reconcile_interruption(c):
    folder=Path(c['interrupted_run']);m=folder/'metrics.json';met=json.loads(m.read_text())
    if met['status'] not in ['running','evaluating']:return
    # Preserve the real last heartbeat; do not pretend it was a process finalizer.
    backup=folder/'metrics_before_user_interrupt.json'
    if not backup.exists():backup.write_bytes(m.read_bytes())
    started=datetime.fromisoformat(json.loads((folder/'events.jsonl').read_text().splitlines()[0])['timestamp'])
    ended=datetime.fromisoformat(c['interrupted_end_upper_utc'])
    upper=(ended-started).total_seconds()/3600+5/3600
    addition=max(0,upper-met['gpu_hours'])
    path=Path(c['output'])/'cost_upper_additions.json';doc=json.loads(path.read_text())
    assert not any(r['run']==str(folder) for r in doc['rows'])
    doc['rows'].append({'run':str(folder),'reason':c['interrupted_reason'],'gpu_hours_recorded':met['gpu_hours'],
                        'gpu_hours_upper_addition':addition,'end_upper_utc':c['interrupted_end_upper_utc']})
    write_json(path,doc)
    write_json(folder/'interrupt_reconciliation.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'reason':c['interrupted_reason'],'original_metrics_sha256':sha256(backup),'gpu_hours_upper':upper,
        'test_opened':met['test_opened'],'original_status':met['status'],'reconciled_status':'failed'})
    write_json(m,{**met,'status':'failed','status_reason':'administrative reconciliation of user-directed interruption',
                  'reconciliation':'interrupt_reconciliation.json','original_heartbeat':'metrics_before_user_interrupt.json'})

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output'])
        assert not (root/'final_lock.json').exists(),'Budget review must precede test unlocking'
        reconcile_interruption(c)
        source=Path(c['val_run']);met=json.loads((source/'metrics.json').read_text())
        assert met['status'] in ['completed','failed'],'Wait for the val job to finish or stop'
        assert not met['test_opened']
        grouped={}
        for row in map(json.loads,(source/'results.jsonl').read_text().splitlines()):
            grouped.setdefault(row['group_id'],[]).append(row)
        expected={(eps,d,s) for eps in c['val_original_epsilons'] for d in c['designs'] for s in c['reader_seeds']}
        full=[];partial={}
        for gid,rows in grouped.items():
            keys={(r['epsilon'],r['attack_design'],r['attack_seed']) for r in rows}
            if len(rows)==len(expected) and keys==expected:full.append(rows)
            else:partial[gid]=len(rows)
        seconds={}
        for line in Path(c['val_log']).read_text().splitlines():
            match=re.search(r'group\s+(\d+)\s+\d+\s+seconds\s+([0-9.]+)',line)
            if match:seconds[int(match.group(1))]=float(match.group(2))
        assert len(full)==len(seconds) and len(full)>1
        coefficients,prediction,residual=throughput(full,seconds,c['main_epsilons'])
        rng=np.random.default_rng(c['seed']);draws=[]
        for i in range(0,c['bootstrap_resamples'],500):
            n=min(500,c['bootstrap_resamples']-i)
            draws.extend(prediction[rng.integers(len(prediction),size=(n,len(prediction)))].mean(1))
        point=float(prediction.mean()/3600);interval=[float(np.quantile(draws,q)/3600) for q in [.025,.975]]
        rate=interval[1]*(1+c['throughput_margin']);spent=total_cost();reserve=c['model_loading_reserve_gpu_hours']
        available=max(0,c['budget_gpu_hours']-spent-reserve)
        maximum=min(500,max(0,math.floor(available/rate)))
        optimistic=min(500,max(0,math.floor(available/point)))
        required=spent+reserve+c['minimum_main_groups']*rate
        expected_required=spent+reserve+c['minimum_main_groups']*point
        doc={'run':str(run.path),'test_opened':False,'passed':maximum>=c['minimum_main_groups'],
             'maximum_feasible_groups':maximum,'maximum_at_point_rate':optimistic,'minimum_main_groups':c['minimum_main_groups'],
             'budget_gpu_hours':c['budget_gpu_hours'],'spent_gpu_hours':spent,'remaining_gpu_hours':c['budget_gpu_hours']-spent,
             'primary_200_gpu_hours':c['minimum_main_groups']*rate,'total_200_gpu_hours':required,
             'point_primary_200_gpu_hours':c['minimum_main_groups']*point,'point_total_200_gpu_hours':expected_required,
             'additional_gpu_hours_for_200':max(0,required-c['budget_gpu_hours']),
             'primary_fraction_of_remaining_when_planning':1.0,'loading_reserve_gpu_hours':reserve,
             'rate_gpu_hours_per_group':rate,'point_rate_gpu_hours_per_group':point,'point_rate_bootstrap_ci':interval,
             'throughput_margin':c['throughput_margin'],'bootstrap_resamples':c['bootstrap_resamples'],
             'val_complete_groups':len(full),'val_expected_groups':c['val_expected_groups'],'val_partial_groups':partial,
             'val_run':str(source),'val_status':met['status'],'val_gpu_hours':met['gpu_hours'],
             'new_main_matrix':{'kind':'E3','lambdas':[1.0],'epsilons':c['main_epsilons'],'designs':c['designs'],'seeds':c['reader_seeds'],'steps':200},
             'model_seconds_coefficients':dict(zip(['fixed_per_group','D0_per_PGD_step','R4_per_PGD_step'],coefficients.tolist())),
             'model_residual_RMSE_seconds':float(np.sqrt(np.mean(residual**2))),
             'forecast_note':'Val five-epsilon group durations fit by nonnegative fixed/D0-step/R4-step costs; predict only the three requested epsilons. Lock rate is 97.5% bootstrap upper mean plus 20% timing margin. All remaining budget is offered to primary E3 before secondary items. Forecast is not a runtime guarantee.',
             'secondary_protocol':{'groups':100,'order':c.get('secondary_order',['lambda=.1/10 at epsilon=.1','minimum search at most 6 bisections','E1/E2 at epsilon=.1']),'executed':False},
             'cropped_settings':{'epsilons':[.03,.3],'old_test60_plan':'superseded by user; never locked or executed'},
             'input_sha256':{str(p):sha256(p) for p in [source/'results.jsonl',source/'metrics.json',Path(c['val_log']),Path('docs/M8c_TASK.md')]}}
        write_json(root/c.get('feasibility_name','budget_feasibility.json'),doc)
        cost.update(status='completed');live(stage='budget_gate_passed' if doc['passed'] else 'awaiting_budget_decision',
              milestone_complete=False,test_opened=False,background_tasks_active=False,budget_feasibility=str(root/c.get('feasibility_name','budget_feasibility.json')),
              maximum_feasible_groups=maximum,total_200_gpu_hours=required,gpu_hours_conservative=spent)
        print(json.dumps(doc,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args();main(load_config(args.config))

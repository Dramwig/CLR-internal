"""Complete only the already authorized, gate-passing M1 pipeline; no retries."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
status=Path('reports/evidence_M1/full_job.json');write_json(status,{'status':'waiting_for_preflight'})
while True:
    correction=json.loads(Path('reports/evidence_M1/revised_job.json').read_text())
    if correction['status'] in ['blocked_non_timeout_failure','revised_smoke_failed']:
        write_json(status,{'status':'blocked_smoke','evidence':correction});sys.exit(1)
    job=json.loads(Path('reports/evidence_M1/job.json').read_text())
    # Ignore the first oversized smoke's failure while the explicit correction runs.
    ready=correction['status'] in ['gate_pipeline_finished','original_smoke_passed_no_retry']
    if ready and job['status'] in ['blocked_gate_failed','gate_execution_failed','budget_failed','budget_completed']:break
    time.sleep(15)
paths={k:job[k] for k in ['smoke_run','gate_run','budget_run'] if k in job}
overrides=[x for x in a.overrides if x.split('=',1)[0] not in paths]+[f'{k}={v}' for k,v in paths.items()]
if job['status']=='budget_completed':
    write_json(status,{'status':'full_generation_running',**paths});before=set(Path('runs').iterdir())
    result=subprocess.run([sys.executable,'scripts/m1_generate.py',a.config,*overrides,'stage=generation'])
    candidates=[r for r in set(Path('runs').iterdir())-before if (r/'config.json').exists() and json.loads((r/'config.json').read_text()).get('stage')=='generation']
    if len(candidates)==1:paths['generation_run']=str(candidates[0]);overrides.append(f'generation_run={candidates[0]}')
    final='generation_completed' if result.returncode==0 else 'generation_failed'
else:final='blocked_preflight'
write_json(status,{'status':final,**paths,'preflight_status':job['status']})
subprocess.run([sys.executable,'scripts/m1_report.py',a.config,*overrides,'stage=report'],check=True)
with Path('reports/M1_data.md').open('a') as f:
    f.write('\n## 自动运行最新状态\n\n'+final+'。详见 [实测结果](M1_results.md) 和 `reports/evidence_M1/full_job.json`；全量状态以generation_run/metrics.json为准。\n')
for name in ['EXPERIMENTS','BLOCKERS']:
    with Path(f'docs/{name}.md').open('a') as f:f.write(f'\n## M1 pipeline — {final}\n\nRuns: {json.dumps(paths)}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job.json`. No failed gate was bypassed.\n')

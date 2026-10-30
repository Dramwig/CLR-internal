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
report_path=Path('reports/M1_data.md')
history=report_path.read_text()
report_path.write_text('# M1 数据报告：'+final+'\n\n'+Path('reports/M1_results.md').read_text()+'\n\n# 协议定义和运行历史\n\n'+history)
if final=='generation_completed':
    import hashlib
    files=[Path('reports/M1_data.md'),Path('reports/M1_results.md'),Path(c['split_path']),Path('clr/policy.py'),Path('clr/data/loader.py'),Path('clr/data/m1.py')]
    write_json(Path('reports/M1_completed_v1_manifest.json'),{'status':'completed_data_generation; awaiting milestone review','runs':paths,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'data_manifest':str(Path(paths['generation_run'])/'artifacts_sha256.json')})
for name in ['EXPERIMENTS','BLOCKERS']:
    with Path(f'docs/{name}.md').open('a') as f:f.write(f'\n## M1 pipeline — {final}\n\nRuns: {json.dumps(paths)}. Evidence: `reports/M1_results.md`, `reports/evidence_M1/full_job.json`. No failed gate was bypassed.\n')

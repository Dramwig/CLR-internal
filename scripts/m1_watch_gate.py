"""Resume the authorized pipeline after an already running smoke; no retries."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
status=Path('reports/evidence_M1/job.json');write_json(status,{'status':'waiting_for_smoke','smoke_run':c['smoke_run']})
while True:
    m=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
    if m['status']!='running':break
    time.sleep(10)
if m['status']!='completed':
    write_json(status,{'status':'blocked_smoke_failed','smoke_run':c['smoke_run']});sys.exit(1)
before=set(Path('runs').iterdir())
cmd=[sys.executable,'scripts/m1_gate.py',a.config,*a.overrides,'stage=batch_gate']
write_json(status,{'status':'batch_gate_running','command':cmd,'smoke_run':c['smoke_run']})
result=subprocess.run(cmd)
new=[r for r in set(Path('runs').iterdir())-before if (r/'config.json').exists() and json.loads((r/'config.json').read_text()).get('stage')=='batch_gate']
if result.returncode or len(new)!=1:
    write_json(status,{'status':'gate_execution_failed','returncode':result.returncode,'runs':[str(r) for r in new]});sys.exit(1)
r=new[0];m=json.loads((r/'metrics.json').read_text());write_json(status,{'status':'gate_passed' if m['gate_passed'] else 'blocked_gate_failed','gate_run':str(r),'smoke_run':c['smoke_run']})
if m['gate_passed']:
    result=subprocess.run([sys.executable,'scripts/m1_budget.py',a.config,*a.overrides,f'gate_run={r}','stage=budget'])
    if result.returncode:write_json(status,{'status':'budget_failed','gate_run':str(r),'smoke_run':c['smoke_run']})

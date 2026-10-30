"""One explicit correction of an oversized smoke workload; no model fallback."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
status=Path('reports/evidence_M1/revised_job.json')
write_json(status,{'status':'waiting_for_original_smoke','run':c['smoke_run']})
while True:
 m=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
 if m['status']!='running':break
 time.sleep(10)
if m['status']=='completed':write_json(status,{'status':'original_smoke_passed_no_retry'});sys.exit(0)
trace=Path(c['smoke_run'],'traceback.txt')
if not trace.exists() or 'smoke exceeded 900 seconds' not in trace.read_text():
 write_json(status,{'status':'blocked_non_timeout_failure','run':c['smoke_run']});sys.exit(1)
before=set(Path('runs').iterdir());write_json(status,{'status':'revised_smoke_running','reason':'Match production split roles; keep200 problems,K8 train,K16 each policy heldout,all early exits,900s'})
result=subprocess.run([sys.executable,'scripts/m1_smoke_v2.py',a.config,*a.overrides,'stage=smoke_v2'])
runs=[r for r in set(Path('runs').iterdir())-before if (r/'config.json').exists() and json.loads((r/'config.json').read_text()).get('stage')=='smoke_v2']
if result.returncode or len(runs)!=1:
 write_json(status,{'status':'revised_smoke_failed','runs':[str(r) for r in runs]});sys.exit(1)
r=runs[0];write_json(status,{'status':'revised_smoke_completed','run':str(r)})
overrides=[x for x in a.overrides if not x.startswith('smoke_run=')]
result=subprocess.run([sys.executable,'scripts/m1_watch_gate.py',a.config,*overrides,f'smoke_run={r}'])
write_json(status,{'status':'gate_pipeline_finished','smoke_run':str(r),'returncode':result.returncode})

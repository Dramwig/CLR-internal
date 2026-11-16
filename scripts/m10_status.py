"""Read actual M10 process handles and durable per-run budget observations."""
import argparse,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import ledger,live

def main(c):
 processes=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
  except (OSError,UnicodeError):continue
  if len(argv)>1 and Path(argv[0]).name.startswith('python') and Path(argv[1]).name.startswith('m10_') and Path(argv[1]).name!='m10_status.py':processes.append({'pid':int(p.name),'argv':argv})
 costs=ledger();trained=[]
 for p in Path(c['output'],'trained').glob('*full.json'):
  r=json.loads(p.read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text())
  if m.get('status')=='completed':trained.append({'reference':str(p),'run':r['run']})
 summary={'processes':processes,'costs':costs,'GPUh_by_section':{s:sum(r['gpu_hours'] for r in costs if r['section']==s) for s in ['P1','P2-4','P2-5','P2-6']},'completed_full_training':trained,'test_opened':any(json.loads((Path(r['run'])/'metrics.json').read_text()).get('test_opened',False) for r in costs if (Path(r['run'])/'metrics.json').exists())}
 write_json(Path(c['output'])/'status_observation.json',summary)
 root=Path(c['output']);stage='complete' if (root/'freeze.json').exists() else 'exports_complete_paper_pending' if (root/'exports_completed.json').exists() else 'locked_final_evaluation' if (root/'final_lock.json').exists() else 'training_in_progress'
 live(stage=stage,active_processes=processes,completed_full_training_count=len(trained),GPUh_by_section=summary['GPUh_by_section'],test_opened=summary['test_opened'])
 print(json.dumps({k:v for k,v in summary.items() if k not in ['costs','completed_full_training']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""M9-A CPU aggregation from saved final predictions; zero model calls."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m9_runtime import snapshot
from scripts.m6_report import collect

def main(c):
 with recorded(c) as run:
  snapshot(run,c);root=Path(c['output']);dest=root/'analysis_A';dest.mkdir(exist_ok=True)
  old=json.loads(Path('data/M6_v1/training_lock.json').read_text());idx=json.loads(Path('data/M6_v1/evaluation_index.json').read_text())
  keys=[k for k in idx if k.startswith('single_')];index={k:idx[k] for k in keys};models={k:old['models'][k] for k in keys}
  new=json.loads((root/'training_lock.json').read_text());models.update(new['models']);index.update(json.loads((root/'evaluation_index.json').read_text()))
  collect(c,{'models':models},index,dest)
  write_json(root/'statistics_complete.json',{'run':str(run.path),'analysis':str(dest)});run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

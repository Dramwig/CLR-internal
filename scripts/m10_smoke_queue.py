"""Execute the frozen list of val-only smoke configs; stop on any error."""
import argparse,json,subprocess,sys
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import ledger,live

def main(c):
 for path in c['configs']:
  job=load_config(path);ref=Path(job['output'])/'trained'/f'{job["case"]}_{job["seed"]}_smoke.json'
  if ref.exists():
   r=json.loads(ref.read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text())
   if m.get('status')=='completed':continue
  live(stage='smoke_running',config=path,test_opened=False)
  with Path('reports/M10_evidence',Path(path).stem+'.log').open('x') as out:
   subprocess.run([sys.executable,'scripts/m10_train_gsm.py',path],stdout=out,stderr=subprocess.STDOUT,check=True)
 write_json('data/M10_v1/gsm_smoke_completed.json',{'configs':c['configs'],'cost':ledger(),'test_opened':False})
 live(stage='gsm_smokes_completed',test_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Sequential YAML-driven M10 training queue; never performs test inference."""
import argparse,json,subprocess,sys
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import ledger,live

def main(c):
 for job in c['jobs']:
  p=job['config'];config=load_config(p);case=config.get('case',config.get('kind'));ref=Path(config['output'])/'trained'/f'{case}_{config["seed"]}_full.json'
  if ref.exists():
   r=json.loads(ref.read_text());m=json.loads((Path(r['run'])/'metrics.json').read_text())
   if m['status']=='completed':continue
   raise RuntimeError('Existing incomplete ref: '+str(ref))
  log=Path('reports/M10_evidence',Path(p).stem+'_full.log')
  attempt=1
  while log.exists():
   attempt+=1;log=Path('reports/M10_evidence',Path(p).stem+f'_full_retry{attempt}.log')
  live(stage='P1_full_training',active_config=p,test_opened=False)
  with log.open('x') as out:subprocess.run([sys.executable,job['script'],p],stdout=out,stderr=subprocess.STDOUT,check=True)
 write_json(c['completion_file'],{'jobs':c['jobs'],'cost':ledger(),'test_opened':False})
 live(stage=c['completed_stage'],test_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

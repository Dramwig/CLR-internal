"""Complete every GSM inference branch using <=5% val proxies only."""
import argparse,json,subprocess,sys
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 for cfg in c['configs']:
  log=Path('reports/M10_evidence')/(Path(cfg).stem+'_branch.log')
  if log.exists():raise FileExistsError(log)
  with log.open('x') as f:subprocess.run([sys.executable,'scripts/m10_evaluate_gsm.py',cfg,*c['overrides']],stdout=f,stderr=subprocess.STDOUT,check=True)
 write_json(c['completion_file'],{'configs':c['configs'],'test_opened':False,'passed':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

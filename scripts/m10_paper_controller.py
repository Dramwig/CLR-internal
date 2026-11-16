"""Complete derived paper products after all experiment exports finish."""
import argparse,json,subprocess,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def execute(script,config,log):
 p=Path(log)
 attempt=1
 while p.exists():
  attempt+=1;p=Path(log+'.retry'+str(attempt))
 with p.open('x') as f:subprocess.run(['.venv/bin/python',script,config],stdout=f,stderr=subprocess.STDOUT,check=True)
def main(c):
 while not Path(c['exports_gate']).exists():time.sleep(30)
 for script,config,log in c['jobs']:execute(script,config,log)
 write_json('data/M10_v1/paper_reports_ready.json',{'experiment_exports_complete':True,'reproduction_checked':True,'reports_written':True,'requires_final_visual_review_and_freeze':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

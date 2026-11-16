"""CPU continuation starts only after every locked final GPU evaluation completes."""
import sys
sys.dont_write_bytecode=True
import subprocess,time
from pathlib import Path
from datetime import datetime
from clr.utils.config import load_config

def main(c,config):
    root=Path(c['output'])
    while not (root/'b_completed.json').exists():
        if datetime.now().astimezone()>=datetime.fromisoformat(c['deadline']):raise TimeoutError('Deadline before B completion')
        time.sleep(20)
    with (root/'analysis_b.log').open('w') as stream:subprocess.run([sys.executable,'-m','supplemental.stats_20261116_v1.analysis_b',config],stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('B final analysis completed',flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]),sys.argv[1])

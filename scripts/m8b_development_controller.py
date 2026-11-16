"""Detached ordered continuation; every dependent experiment has a YAML/run."""
import sys,time,subprocess
from pathlib import Path

def execute(module,config,log):
    with Path(log).open('x') as f:subprocess.run([sys.executable,'-m',module,config],stdout=f,stderr=subprocess.STDOUT,check=True)

while not Path('data/M8b_v1/training/R2_33.json').exists():time.sleep(10)
execute('scripts.m8b_train_pipeline','configs/M8b/train_R3_all.yaml','reports/M8b_evidence/train_R3_pipeline.log')
execute('scripts.m8b_calibrate','configs/M8b/calibrate_all.yaml','reports/M8b_evidence/calibrate_all.log')

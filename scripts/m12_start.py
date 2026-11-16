"""Recorded gate recheck after exact numerical parity fix, then full pipeline."""
import argparse,subprocess,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c,config):
    subprocess.run(['.venv/bin/python','-m','pytest','tests/test_m12.py','-q'],check=True,stdout=Path('reports/M12_evidence/unit_launch.log').open('w'),stderr=subprocess.STDOUT)
    for name in ['smoke_train.json','smoke.json']:
        p=Path(c['output'],name)
        if p.exists():p.rename(p.with_name(p.stem+'_initial.json'))
    subprocess.run(['.venv/bin/python','-m','scripts.m12_smoke',config],check=True)
    subprocess.run(['.venv/bin/python','-m','scripts.m12_protocol_smoke',config],check=True)
    write_json('docs/M12_LIVE.json',{'stage':'smoke_passed_starting_full','deadline':c['deadline'],'budget_gpu_hours':30,'test_opened':False})
    subprocess.run(['.venv/bin/python','-m','scripts.m12_pipeline',config],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config),a.config)

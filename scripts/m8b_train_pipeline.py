"""Sequential training/calibration controller, no test access."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.audit.m8b_runtime import live

def main(c):
    for design in c['designs']:
        for seed in c['reader_seeds']:
            cfg={**c,'design':design,'seed':seed};tag='_warm_v2' if c.get('warm',False) else '';p=Path(f'configs/M8b/train_{design}_{seed}{tag}.yaml');p.write_text(yaml.safe_dump(cfg,sort_keys=False))
            with Path(f'reports/M8b_evidence/train_{design}_{seed}{tag}.log').open('x') as log:
                subprocess.run([sys.executable,'-m','scripts.m8b_train',str(p)],stdout=log,stderr=subprocess.STDOUT,check=True)
    live(stage='training_all_completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

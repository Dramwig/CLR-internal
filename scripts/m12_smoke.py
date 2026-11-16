"""Five training variants + disjoint reader_val calibration/evaluation proxies."""
import argparse,json,time,subprocess
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    started=time.monotonic();rows=[]
    for a in ['A1','A2','A3','A4','A5']:
        cfg={**c,'ablation':a,'case':'smoke_'+a,'mode':'smoke','epochs':1,'min_epochs':1}
        p=Path('configs/M12')/f'smoke_{a}.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False))
        with Path(f'reports/M12_evidence/smoke_{a}.log').open('w') as log:
            subprocess.run(['.venv/bin/python','-m','scripts.m12_train',str(p)],check=True,stdout=log,stderr=subprocess.STDOUT)
        run=json.loads(Path(c['output'],f'trained_smoke_{a}_lite_11.json').read_text())['run']
        rows.append({'ablation':a,'run':run})
        if time.monotonic()-started>=850:raise TimeoutError('Smoke suite too slow for 900s gate')
    write_json(Path(c['output'])/'smoke_train.json',{'passed':True,'jobs':rows,'elapsed_seconds':time.monotonic()-started,'test_opened':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

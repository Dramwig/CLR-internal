"""Rebuild paper products in the isolated plotting environment and compare."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from clr.utils.m9_runtime import snapshot


def main(c):
    with recorded(c) as run:
        snapshot(run,c)
        subprocess.run([c['python'],'-I','scripts/paper/make_all.py',c['repeat_config']],check=True)
        original,repeat=Path(c['primary_root']),Path(c['repeat_root'])
        records=[]
        for sub in ['tables','figures','derived']:
            left=sorted((original/sub).glob('*'))
            right=sorted((repeat/sub).glob('*'))
            if {p.name for p in left}!={p.name for p in right}:raise ValueError('Artifact inventory differs')
            for a in left:
                b=repeat/sub/a.name;identical=a.read_bytes()==b.read_bytes()
                numerical=None
                if a.suffix=='.csv':
                    x,y=pd.read_csv(a),pd.read_csv(b)
                    try:pd.testing.assert_frame_equal(x,y,check_exact=False,rtol=1e-12,atol=1e-12);numerical=True
                    except AssertionError:numerical=False
                records.append({'artifact':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
                                'repeat_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),
                                'byte_identical':identical,'numerically_identical':numerical})
        if not all(r['byte_identical'] for r in records):
            raise ValueError('Paper build was not byte-identical; inspect the recorded environment before freezing')
        probe=subprocess.run([c['python'],'-I','-c',
            'import importlib.util,json; print(json.dumps({k:importlib.util.find_spec(k) is not None for k in ["torch","transformers","numpy","pandas","matplotlib","yaml"]}))'],
            capture_output=True,text=True,check=True)
        environment=json.loads(probe.stdout)
        if environment['torch'] or environment['transformers']:raise ValueError('Plotting environment is not independent')
        write_json(Path(c['output'])/'reproducibility.json',{'run':str(run.path),'passed':True,
                   'environment_modules':environment,'artifacts':records,'tolerance':1e-12,
                   'protocol':'Independent clean plotting venv; second build in isolated Python processes, separate output directory'})
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,byte_identical=len(records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');main(load_config(p.parse_args().config))

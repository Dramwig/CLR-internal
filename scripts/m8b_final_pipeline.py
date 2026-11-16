"""Locked final evaluation only; no parameter changes or outcome selection."""
import sys
sys.dont_write_bytecode=True
import argparse,subprocess
from pathlib import Path
from clr.utils.config import load_config

def main(c):
    for spec in c['jobs']:
        with Path(spec['log']).open('x') as f:
            subprocess.run([sys.executable,'-m',spec['module'],spec['config']],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

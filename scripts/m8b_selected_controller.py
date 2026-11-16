"""YAML-controlled val-only dependency queue."""
import sys
sys.dont_write_bytecode=True
import argparse,subprocess,time
from pathlib import Path
from clr.utils.config import load_config

def main(c):
    started=time.monotonic()
    while not all(Path(p).exists() for p in c['wait_for']):
        if time.monotonic()-started>c['max_wait_seconds']:raise RuntimeError('M8b development dependency wait timed out')
        time.sleep(10)
    for j in c['jobs']:
        with Path(j['log']).open('x') as f:subprocess.run([sys.executable,'-m',j['module'],j['config']],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

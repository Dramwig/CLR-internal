"""Run explicit diagnostic config sequence without silently restarting failures."""
import argparse,subprocess
from clr.utils.config import load_config
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
 for cfg in c['configs']:
  subprocess.run(['.venv/bin/python','scripts/m41_learnability.py',cfg],check=True)

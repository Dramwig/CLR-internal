"""Durable continuation after training: locked evaluation, report, final verification.

No automatic recovery from failed training/evaluation; errors retain their runs.
"""
import argparse,json,time,subprocess,os
from pathlib import Path
from clr.utils.config import load_config

def main(c):
    root=Path(c['output']);lock=root/'training_complete.json'
    while not lock.exists():
        # The tmux controller's exit is a real blocker, not a reason to alter settings.
        alive=subprocess.run(['tmux','has-session','-t',c['training_session']],capture_output=True).returncode==0
        if not alive:raise RuntimeError('Training controller exited before lock; inspect its log and failed run')
        time.sleep(30)
    env={**os.environ,'PYTHONPATH':'.','OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1'}
    subprocess.run(['.venv/bin/python','scripts/m4_eval_suite.py',c['eval_config']],check=True,env=env)
    subprocess.run(['.venv/bin/python','scripts/m4_report.py',c['report_config']],check=True,env=env)
    subprocess.run(['.venv/bin/python','scripts/m4_finish.py',c['report_config']],check=True,env=env)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

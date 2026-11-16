"""Wait for final inference to close, then verify, report and freeze M7.1."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time,shutil
from pathlib import Path
from clr.utils.config import load_config


def run(script,config,label):
    with (Path('reports/M71_evidence')/(label+'.log')).open('x') as handle:
        subprocess.run([sys.executable,script,config],stdout=handle,stderr=subprocess.STDOUT,check=True)


def main(c):
    root=Path(c['output'])
    while True:
        live=json.loads(Path('docs/M71_LIVE.json').read_text())
        if (root/'freeze.json').exists():return
        if live.get('stage')=='main_comparison_complete_pending_statistics':break
        time.sleep(20)
    previous=root/'upstream_verification_startup.json'
    if not previous.exists():shutil.copyfile(root/'upstream_verification.json',previous)
    run('scripts/m71_verify.py',c['verify_config'],'upstream_verify_final')
    run('scripts/m71_report.py',c['report_config'],'report_final')
    run('scripts/m71_freeze.py',c['freeze_config'],'freeze_final')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('overrides',nargs='*')
    args=parser.parse_args();main(load_config(args.config,args.overrides))

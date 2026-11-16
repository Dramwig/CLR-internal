"""Durable, no-GPU continuation: wait, analyze, then freeze only passing evidence."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def main(c):
    root=Path(c['search_run']);last=-1
    while subprocess.run(['tmux','has-session','-t',c['search_session']],capture_output=True).returncode==0:
        metrics=json.loads((root/'metrics.json').read_text())
        live=json.loads(Path('docs/M5a_LIVE.json').read_text())
        progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
        n=metrics.get('completed_questions',0)
        live.update(updated_utc=datetime.now(timezone.utc).isoformat(),stage='full_running',
                    active_run=str(root),completed_questions=n,total_questions=300,progress=progress,
                    conservative_total_gpu_hours_so_far=c['prior_gpu_hours']+progress.get('gpu_hours',0))
        write_json(Path('docs/M5a_LIVE.json'),live)
        if n!=last:
            print(json.dumps({'completed_questions':n,'questions':300,
                              'gpu_hours_so_far':live['conservative_total_gpu_hours_so_far']},ensure_ascii=False),flush=True)
            last=n
        time.sleep(c['poll_seconds'])
    metrics=json.loads((root/'metrics.json').read_text())
    if metrics['status']!='completed' or not metrics.get('gate_passed'):
        live=json.loads(Path('docs/M5a_LIVE.json').read_text())
        live.update(stage='search_failed_needs_explicit_recovery',milestone_complete=False,metrics=metrics)
        write_json(Path('docs/M5a_LIVE.json'),live)
        raise RuntimeError('Search failed; no automatic settings/model reduction')
    for script,config,log in [('scripts/m5a_report.py',c['report_config'],c['report_log']),
                              ('scripts/m5a_finish.py',c['finish_config'],c['finish_log'])]:
        with Path(log).open('w') as f:
            subprocess.run([sys.executable,script,config],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('M5a completed, analyzed, frozen; review ready.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*')
    a=p.parse_args();main(load_config(a.config,a.overrides))

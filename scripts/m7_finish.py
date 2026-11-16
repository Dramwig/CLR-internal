"""Finalization controller: completed training -> calibration lock -> val -> final test -> report/freeze."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,subprocess,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live

def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);start=time.monotonic();jobs=[]
        def wait_file(path):
            while not path.exists():
                if time.monotonic()-start>c['max_wait_seconds']:raise RuntimeError('M7 finish wait deadline exceeded')
                time.sleep(10)
        def execute(label,script,config,overrides=()):
            log=Path('reports/M7_evidence')/(label+'.log')
            live(stage=label,finish_controller=str(run.path),active_log=str(log))
            env={**os.environ,'PYTHONPATH':'.','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2',
                 'LD_LIBRARY_PATH':'/home/me/anaconda3/envs/latdefer/lib'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')}
            with log.open('w') as f:proc=subprocess.run([sys.executable,script,config,*overrides],stdout=f,stderr=subprocess.STDOUT,env=env)
            if proc.returncode:raise RuntimeError(f'M7 {label} failed; see {log}')
            jobs.append({'stage':label,'log':str(log),'completed':True});write_json(root/'finish_jobs.json',jobs)
        wait_file(root/'all_trained.json')
        for split in ['cal_A','cal_B','reader_val']:
            wait_file(root/'source'/split/'completed.json')
        wait_file(root/'source/reader_val/natural_commitment_completed.json')
        # Calibration and final settings are exercised by val-only smoke beforehand.
        if not (root/'final_lock.json').exists():execute('calibration_full','scripts/m7_evaluate.py',c['evaluation_config'])
        if not (root/'evaluated/reader_val.json').exists():execute('evaluate_reader_val','scripts/m7_evaluate.py',c['evaluation_config'],['phase=evaluate','split=reader_val'])
        if not (root/'source/test/natural_commitment_completed.json').exists():execute('source_test_final','scripts/m7_generate.py',c['test_source_config'])
        if not (root/'evaluated/test.json').exists():execute('evaluate_test_final','scripts/m7_evaluate.py',c['evaluation_config'],['phase=evaluate','split=test'])
        execute('report_final','scripts/m7_report.py',c['report_config'])
        execute('freeze_final','scripts/m7_freeze.py',c['freeze_config'])
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=True,jobs=jobs,
                          cost_basis='controller only; child runs account all GPU residency')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

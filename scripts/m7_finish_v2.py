"""Locked M7 finalization with independent val evaluation/test generation in parallel."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,signal,subprocess,time,threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);start=time.monotonic();jobs=[];mutex=threading.Lock()
        pid=c['superseded_waiter_pid'];old=Path(c['superseded_waiter_run'])
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/m7_finish.py ' not in cmd:raise ValueError('Unexpected old finish-controller identity')
        children=subprocess.check_output(['ps','--ppid',str(pid),'-o','pid='],text=True).split() if subprocess.run(['ps','--ppid',str(pid),'-o','pid='],capture_output=True).returncode==0 else []
        if children:raise ValueError('Old finisher has active children; refuse to interrupt')
        os.kill(pid,signal.SIGTERM)
        met=json.loads((old/'metrics.json').read_text())
        met.update(status='interrupted',gpu_hours=0,source_forward_calls=0,
                   interruption_utc=datetime.now(timezone.utc).isoformat(),
                   reason='Idle CPU waiter superseded before calibration; independent final jobs overlap after lock',superseded_by=str(run.path))
        write_json(old/'metrics.json',met)
        write_json(root/'finish_handoff_v2.json',{'old_run':str(old),'new_run':str(run.path),'old_waiter_had_no_children':True})
        def wait_file(path):
            while not path.exists():
                if time.monotonic()-start>c['max_wait_seconds']:raise RuntimeError('M7 finalization wait deadline exceeded')
                time.sleep(10)
        def execute(label,script,config,overrides=()):
            log=Path('reports/M7_evidence')/(label+'.log')
            if log.exists():raise FileExistsError('Refuse to overwrite a completed/failed final-stage log: '+str(log))
            with mutex:live(stage=label,finish_controller=str(run.path),active_log=str(log))
            env={**os.environ,'PYTHONPATH':'.','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2',
                 'LD_LIBRARY_PATH':'/home/me/anaconda3/envs/latdefer/lib'}
            with log.open('w') as f:result=subprocess.run([sys.executable,script,config,*overrides],stdout=f,stderr=subprocess.STDOUT,env=env)
            if result.returncode:raise RuntimeError(f'M7 {label} failed; see {log}')
            with mutex:
                jobs.append({'stage':label,'log':str(log),'completed':True});write_json(root/'finish_jobs.json',jobs)
        wait_file(root/'all_trained.json');wait_file(root/'mismatch_v2_complete.json')
        for split in ['cal_A','cal_B','reader_val']:wait_file(root/'source'/split/'completed.json')
        wait_file(root/'source/reader_val/natural_commitment_completed.json')
        if not (root/'final_lock.json').exists():execute('calibration_full','scripts/m7_evaluate.py',c['evaluation_config'])
        # No model/threshold selection is performed by either child. The test
        # generator independently verifies every final-lock hash before access.
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending=[]
            if not (root/'evaluated/reader_val.json').exists():pending.append(pool.submit(execute,'evaluate_reader_val','scripts/m7_evaluate.py',c['evaluation_config'],['phase=evaluate','split=reader_val']))
            if not (root/'source/test/natural_commitment_completed.json').exists():pending.append(pool.submit(execute,'source_test_final','scripts/m7_generate.py',c['test_source_config']))
            for future in pending:future.result()
        if not (root/'evaluated/test.json').exists():execute('evaluate_test_final','scripts/m7_evaluate.py',c['evaluation_config'],['phase=evaluate','split=test'])
        execute('report_final','scripts/m7_report.py',c['report_config'])
        execute('freeze_final','scripts/m7_freeze.py',c['freeze_config'])
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=True,jobs=jobs,cost_basis='CPU controller only; child runs account GPU costs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

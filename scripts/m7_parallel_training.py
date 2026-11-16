"""Reuse the completed source GPU for disjoint reader jobs; existing training continues."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,signal,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);pid=c['existing_controller_pid']
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/m7_train_pipeline.py' not in cmd:raise ValueError('Refuse to suspend an unrelated process')
        source_completed=all((root/'source'/s/'completed.json').exists() for s in ['reader_train','reader_val','cal_A','cal_B'])
        if not source_completed:raise ValueError('Source GPU not yet released by completed source task')
        os.kill(pid,signal.SIGSTOP)
        write_json(root/'parallel_handoff.json',{'controller_pid':pid,'suspended_supervisor_only':True,'existing_CLR_11_training_continues':True,'run':str(run.path)})
        base=load_config(c['training_config']);deadline=time.monotonic()+c['max_wait_seconds']
        def worker(device,jobs,wait_first):
            if wait_first:
                while not (root/'trained/CLR_11.json').exists():
                    if time.monotonic()>deadline:raise RuntimeError('Waiting for existing CLR11 exceeded deadline')
                    time.sleep(10)
            results=[]
            for kind,seed in jobs:
                pointer=root/'trained'/f'{kind}_{seed}.json'
                if pointer.exists():raise ValueError('Unexpected duplicate training pointer '+str(pointer))
                config={**base,'kind':kind,'seed':seed,'device':device};path=root/'training_configs'/f'{kind}_{seed}.yaml';path.write_text(yaml.safe_dump(config))
                log=Path('reports/M7_evidence')/f'train_{kind}_{seed}.log'
                live(stage='parallel_reader_training',parallel_training_run=str(run.path),**{('gpu_'+device.split(':')[1]+'_job'):f'{kind}/{seed}'})
                env={**os.environ,'LD_LIBRARY_PATH':'/home/me/anaconda3/envs/latdefer/lib','PYTHONPATH':'.','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2'}
                with log.open('w') as f:proc=subprocess.run([sys.executable,'scripts/m7_train.py',str(path)],stdout=f,stderr=subprocess.STDOUT,env=env)
                if proc.returncode:raise RuntimeError(f'{kind}/{seed} failed on {device}: {log}')
                results.append({'kind':kind,'seed':seed,'device':device,'pointer':str(pointer)})
                write_json(root/('jobs_gpu'+device.split(':')[1]+'.json'),results)
            return results
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                a=pool.submit(worker,c['reused_source_device'],c['source_gpu_jobs'],False)
                b=pool.submit(worker,c['existing_reader_device'],c['reader_gpu_jobs'],True)
                jobs=a.result()+b.result()
            os.kill(pid,signal.SIGCONT)
            write_json(root/'parallel_training_completed.json',{'jobs':jobs,'original_controller_resumed':True})
            run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,jobs=jobs,cost_basis='controller only; child runs account GPU time')
        except Exception:
            live(stage='parallel_training_failed_original_supervisor_suspended',parallel_training_run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Balance remaining independent M7 jobs; preserve every running training child."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,queue,signal,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live


def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);pid=c['superseded_supervisor_pid']
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'scripts/m7_parallel_training.py' not in cmd:raise ValueError('Unexpected supervisor identity')
        expected={job['pid'] for job in c['existing_jobs']}
        os.kill(pid,signal.SIGSTOP)
        children=set(map(int,subprocess.check_output(['ps','--ppid',str(pid),'-o','pid='],text=True).split()))
        if children!=expected:
            os.kill(pid,signal.SIGCONT)
            raise ValueError('Children changed before handoff; original supervisor resumed')
        for job in c['existing_jobs']:
            child_cmd=Path(f"/proc/{job['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
            if 'scripts/m7_train.py' not in child_cmd:raise ValueError('Unexpected child identity')
        write_json(root/'balanced_handoff.json',{'run':str(run.path),'suspended_supervisor':pid,
                   'training_children_preserved':c['existing_jobs'],'remaining_jobs':c['remaining_jobs']})
        jobs=queue.Queue()
        for item in c['remaining_jobs']:jobs.put(item)
        deadline=time.monotonic()+c['max_wait_seconds'];base=load_config(c['training_config'])
        folder=root/'balanced_training_configs';folder.mkdir(exist_ok=False)
        def worker(existing):
            device=existing['device'];pointer=root/'trained'/f"{existing['kind']}_{existing['seed']}.json"
            while alive(existing['pid']):
                if time.monotonic()>deadline:raise RuntimeError('Existing child wait limit exceeded')
                time.sleep(5)
            if not pointer.exists():raise RuntimeError('Preserved child failed; inspect its original run/log')
            results=[]
            while True:
                try:kind,seed=jobs.get_nowait()
                except queue.Empty:return results
                pointer=root/'trained'/f'{kind}_{seed}.json'
                if pointer.exists():raise ValueError('Unexpected duplicate trained pointer '+str(pointer))
                cfg={**base,'kind':kind,'seed':seed,'device':device}
                path=folder/f'{kind}_{seed}.yaml';path.write_text(yaml.safe_dump(cfg))
                log=Path('reports/M7_evidence')/f'train_{kind}_{seed}.log'
                if log.exists():raise FileExistsError('Refuse to overwrite existing job log: '+str(log))
                live(stage='balanced_reader_training',balanced_training_run=str(run.path),
                     **{'gpu_'+device.split(':')[1]+'_job':f'{kind}/{seed}'})
                env={**os.environ,'LD_LIBRARY_PATH':'/home/me/anaconda3/envs/latdefer/lib',
                     'PYTHONPATH':'.','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2'}
                with log.open('w') as f:
                    result=subprocess.run([sys.executable,'scripts/m7_train.py',str(path)],stdout=f,stderr=subprocess.STDOUT,env=env)
                if result.returncode:raise RuntimeError(f'{kind}/{seed} failed on {device}; see {log}')
                results.append({'kind':kind,'seed':seed,'device':device,'pointer':str(pointer)})
                write_json(root/('balanced_jobs_gpu'+device.split(':')[1]+'.json'),results)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,e) for e in c['existing_jobs']]
            results=[item for f in futures for item in f.result()]
        # All of the old supervisor's children have exited. Retire only that
        # suspended CPU scheduler; the original aggregation supervisor can now
        # skip the completed pointers and emit all_trained.json.
        os.kill(pid,signal.SIGKILL)
        old=Path(c['superseded_run']);metrics=json.loads((old/'metrics.json').read_text())
        metrics.update(status='interrupted',gpu_hours=0,source_forward_calls=0,
                       interruption_utc=datetime.now(timezone.utc).isoformat(),
                       reason='CPU scheduler superseded by balanced queue; all training children completed normally',
                       superseded_by=str(run.path))
        write_json(old/'metrics.json',metrics)
        os.kill(c['aggregation_supervisor_pid'],signal.SIGCONT)
        write_json(root/'balanced_training_completed.json',{'jobs':results,'original_aggregation_resumed':True})
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,jobs=results,
                          cost_basis='CPU scheduler only; GPU costs recorded by each training child')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

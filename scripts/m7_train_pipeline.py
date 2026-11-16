"""Durable sequential reader jobs; no test access, no dependency on M6."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,subprocess,time
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live

def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);start=time.monotonic()
        for split in ['reader_train','reader_val']:
            while not (root/'source'/split/'completed.json').exists():
                if time.monotonic()-start>c['max_wait_seconds']:raise RuntimeError('M7 source wait limit exceeded')
                live(stage='waiting_source_for_training',training_pipeline=str(run.path),waiting_split=split)
                time.sleep(10)
        jobs=[]
        # Complete each seed's CLR/B3 pair first, then remaining controls.
        order=[(kind,seed) for seed in c['seeds'] for kind in ['CLR','B3']]
        order += [(kind,seed) for kind in c['kinds'] if kind not in ['CLR','B3'] for seed in c['seeds']]
        base=load_config(c['training_config']);folder=root/'training_configs';folder.mkdir(exist_ok=True)
        for kind,seed in order:
            pointer=root/'trained'/f'{kind}_{seed}.json'
            if pointer.exists():
                status=json.loads(Path(json.loads(pointer.read_text())['run'],'metrics.json').read_text())
                if status['status']!='completed':raise ValueError('Invalid existing trained pointer')
                jobs.append({'kind':kind,'seed':seed,'status':'completed','pointer':str(pointer)});continue
            config={**base,'kind':kind,'seed':seed,'device':c['reader_device']}
            path=folder/f'{kind}_{seed}.yaml';path.write_text(yaml.safe_dump(config))
            log=Path('reports/M7_evidence')/f'train_{kind}_{seed}.log'
            live(stage='reader_training',training_pipeline=str(run.path),active_kind=kind,active_seed=seed,training_log=str(log))
            with log.open('w') as f:
                proc=subprocess.run([sys.executable,'scripts/m7_train.py',str(path)],stdout=f,stderr=subprocess.STDOUT,
                                    env={**os.environ,'PYTHONPATH':'.','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'2',
                                         'LD_LIBRARY_PATH':'/home/me/anaconda3/envs/latdefer/lib'+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')})
            if proc.returncode:raise RuntimeError(f'M7 {kind}/{seed} failed; see {log}')
            jobs.append({'kind':kind,'seed':seed,'status':'completed','pointer':str(pointer)})
            write_json(root/'training_jobs.json',jobs)
        write_json(root/'all_trained.json',{'run':str(run.path),'jobs':jobs,'test_opened':False})
        live(stage='all_reader_training_completed',training_pipeline=str(run.path))
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,
                          cost_basis='controller only; child runs account GPU time',jobs=jobs)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

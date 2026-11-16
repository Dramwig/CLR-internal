"""Continue the live seed 11 worker and run the other seeds concurrently.

This changes scheduling only. The scientific protocol, minimum epochs, seeds,
data, precision, and cumulative 10 GPU hour guard remain unchanged.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from clr.utils.m9_runtime import hours, live, snapshot


def main(c):
    with recorded(c) as run:
        snapshot(run, c)
        root = Path(c['output'])
        supplement = root / 'execution_supplement.json'
        if supplement.exists():
            raise FileExistsError(supplement)
        # The original controller is in subprocess.run waiting for seed 11.
        # Stop only that controller; its already-running child is preserved.
        cmdline = Path(f'/proc/{c["previous_controller_pid"]}/cmdline').read_bytes()
        if b'scripts.m9_pipeline' not in cmdline:
            raise RuntimeError('The previous controller PID no longer matches')
        os.kill(c['previous_controller_pid'], signal.SIGTERM)
        write_json(supplement, {
            'run': str(run.path), 'reason': 'Concurrent scheduling to shorten wall time; no scientific changes',
            'previous_controller_run': c['previous_controller_run'],
            'preserved_worker_pid': c['preserved_worker_pid'],
            'preserved_worker_run': c['preserved_worker_run'],
            'training_configs': c['training_configs'], 'budget_gpu_hours': c['budget_gpu_hours'],
            'test_opened': False,
        })
        children = []
        for cfg in c['training_configs'][1:]:
            t = load_config(cfg)
            log = Path('reports/M9_evidence') / f'train_{t["seed"]}_parallel.log'
            stream = log.open('x')
            child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.m9_train', cfg],
                                     stdout=stream, stderr=subprocess.STDOUT)
            children.append((child, stream, t['seed']))
        write_json(run.path / 'children.json', [{'pid': p.pid, 'seed': s} for p, _, s in children])
        while True:
            for child, stream, seed in children:
                if child.poll() is not None and child.returncode:
                    raise RuntimeError(f'Training seed {seed} failed; see preserved worker log')
            refs = [root / 'trained' / f'multi_{s}.json' for s in (11, 22, 33)]
            if all(p.exists() for p in refs):
                break
            if not Path(f'/proc/{c["preserved_worker_pid"]}').exists() and not refs[0].exists():
                raise RuntimeError('Preserved seed 11 worker ended without a completed reference')
            live(stage='A_training_parallel', test_opened=False,
                 trained_seeds=[s for s, p in zip((11, 22, 33), refs) if p.exists()],
                 gpu_hours=sum(x['gpu_hours'] for x in hours()))
            time.sleep(20)
        for child, stream, _ in children:
            child.wait()
            stream.close()
        models = {}
        for cfg, ref_path in zip(c['training_configs'], refs):
            t = load_config(cfg)
            ref = json.loads(ref_path.read_text())
            metrics = json.loads((Path(ref['run']) / 'metrics.json').read_text())
            if metrics['status'] != 'completed':
                raise RuntimeError('All training must complete before final inference')
            h = hashlib.sha256(Path(ref['checkpoint']).read_bytes()).hexdigest()
            if h != ref['sha256']:
                raise ValueError('Training checkpoint hash mismatch')
            models[f'multi_lite_{t["seed"]}'] = {
                'config': t, 'checkpoint': ref['checkpoint'], 'sha256': h,
            }
        if sum(x['gpu_hours'] for x in hours()) >= c['budget_gpu_hours'] - .3:
            raise RuntimeError('Final inference reserve exhausted')
        lock = root / 'training_lock.json'
        if lock.exists():
            raise FileExistsError(lock)
        write_json(lock, {'all_training_locked': True, 'models': models,
                         'checkpoint_sha256': {k: v['sha256'] for k, v in models.items()},
                         'test_opened': False, 'execution_supplement': str(supplement)})
        live(stage='A_final_evaluation', test_opened=True)
        index = {}
        for key, m in models.items():
            index[key] = {}
            for source in ('gpt2', 'llama'):
                e = {**m['config'], 'stage': 'evaluate', 'mode': 'full', 'model_key': key,
                     'source': source, 'checkpoint': m['checkpoint'], 'eval_lock': str(lock),
                     'eval_splits': ['reader_val', 'test']}
                cfg = Path('configs/M9') / f'evaluate_{key}_{source}.yaml'
                cfg.write_text(yaml.safe_dump(e, sort_keys=False))
                log = Path('reports/M9_evidence') / f'evaluate_{key}_{source}.log'
                with log.open('x') as stream:
                    subprocess.run([sys.executable, '-u', '-m', 'scripts.m9_evaluate', str(cfg)],
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
                ref = json.loads((root / 'evaluated' / f'{key}_{source}.json').read_text())
                index[key][source] = ref['run']
                write_json(root / 'evaluation_index.json', index)
        live(stage='A_evaluation_completed', test_opened=True, primary_experiment_complete=True)
        write_json(root / 'experiment_complete.json', {
            'run': str(run.path), 'models': list(models),
            'gpu_hours': sum(x['gpu_hours'] for x in hours()),
        })
        run.write_metrics(status='completed', gpu_hours=0, source_forward_calls=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    main(load_config(p.parse_args().config))

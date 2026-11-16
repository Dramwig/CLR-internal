"""Post-C1 training/calibration orchestration, with a test-lock barrier."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

from clr.audit.features import sha256
from clr.audit.m71_runtime import live
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def job(script, values, label):
    config = Path('configs/M71') / (label + '.yaml')
    with config.open('x') as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
    log = Path('reports/M71_evidence') / (label + '.log')
    stream = log.open('x')
    process = subprocess.Popen([sys.executable, script, str(config)],
                               stdout=stream, stderr=subprocess.STDOUT)
    return process, stream, log


def finish(jobs):
    failures = []
    while jobs:
        for process, stream, log in list(jobs):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            jobs.remove((process, stream, log))
            if code:
                failures.append((str(log), code))
        if jobs:
            time.sleep(10)
    if failures:
        raise RuntimeError('M7.1 children failed: ' + repr(failures))


def source_parts(c, source, splits, tag):
    jobs = []
    for i, device in enumerate(c['devices']):
        values = {**source, 'mode': 'full', 'source_splits': splits,
                  'partitions': len(c['devices']), 'partition': i,
                  'device': device, 'min_free_GiB': c['source_min_free_GiB'][i],
                  'max_seconds': 14400, 'resume': False, 'include_controls': True}
        jobs.append(job('scripts/m71_source.py', values, tag + '_source_' + str(i)))
    finish(jobs)
    merge = {**source, 'source_splits': splits, 'partitions': len(c['devices'])}
    finish([job('scripts/m71_merge.py', merge, tag + '_merge')])


def main(c):
    root = Path(c['output'])
    while not (root / 'admitted_representation.json').exists():
        if (root / 'completion.json').exists():
            return  # The registered C1 stop branch owns reporting/freezing.
        time.sleep(20)
    admitted = json.loads((root / 'admitted_representation.json').read_text())
    rep = admitted['representation']
    selection = json.loads((root / admitted['selection_name']).read_text())
    base = yaml.safe_load(Path(c['train_config']).read_text())
    base.update(representation=rep, device=c['devices'][0])
    smoke = yaml.safe_load(Path(c['smoke_config']).read_text())
    smoke.update(representation=rep, device=c['devices'][0])
    finish([job('scripts/m71_train.py', smoke, rep + '_reader_smoke')])
    overfit = yaml.safe_load(Path(c['overfit_config']).read_text())
    overfit.update(representation=rep, device=c['devices'][0])
    finish([job('scripts/m71_train.py', overfit, rep + '_overfit')])
    c2 = json.loads((root / rep / 'C2.json').read_text())
    if not c2['passed']:
        live(stage='C2_failed', C2_passed=False, test_opened=False)
        raise RuntimeError('C2 overfit gate failed; main comparison forbidden')
    live(C2_passed=True)
    # B3 must finish before CLR so that actual exposure is matched per seed.
    for kind in ['B3', 'linear', 'B0', 'CLR', 'supervised']:
        jobs = []
        for seed, device in zip(c['seeds'], c['devices']):
            values = {**base, 'kind': kind, 'seed': seed, 'device': device}
            jobs.append(job('scripts/m71_train.py', values,
                            rep + '_train_' + kind + '_' + str(seed)))
        finish(jobs)
    source = yaml.safe_load(Path(c['source_config']).read_text())
    source.update(representation=rep, layers=selection['layers'],
                  critical_tokens=admitted['critical_tokens'],
                  selection_name=admitted['selection_name'])
    source_parts(c, source, ['cal_A', 'cal_B'], rep + '_cal')
    receiver = {**source, 'mode': 'full', 'diagnostic': 'receiver',
                'source_splits': ['cal_A', 'cal_B'],
                'subset_counts': {'cal_A': 300, 'cal_B': 300},
                'device': c['devices'][0],
                'min_free_GiB': c['source_min_free_GiB'][0],
                'max_seconds': 14400, 'resume': False}
    finish([job('scripts/m71_diagnostics.py', receiver, rep + '_cal_receiver')])
    inference = {**base, 'mode': 'calibrate', 'max_seconds': 14400}
    for kind in ['B3', 'linear', 'B0', 'CLR', 'supervised']:
        finish([job('scripts/m71_evaluate.py',
                    {**inference, 'kind': kind, 'seed': seed, 'device': device},
                    rep + '_calibrate_' + kind + '_' + str(seed))
                for seed, device in zip(c['seeds'], c['devices'])])
    for kind in ['norm', 'text']:
        finish([job('scripts/m71_baselines.py', {
            'milestone': 'M7.1', 'output': c['output'], 'representation': rep,
            'mode': 'calibrate', 'kind': kind}, rep + '_calibrate_' + kind)])
    # Predeclare every remaining scientific config before opening test.
    final_configs = []
    for kind in ['B3', 'linear', 'B0', 'CLR', 'supervised']:
        for seed, device in zip(c['seeds'], c['devices']):
            final_configs.append({**inference, 'mode': 'final', 'kind': kind,
                                  'seed': seed, 'device': device})
    planned = root / 'final_plan.json'
    write_json(planned, {'representation': rep, 'evaluation_configs': final_configs,
                        'source_config': source, 'receiver_config': receiver,
                        'seeds': c['seeds'], 'test_groups': 2000,
                        'receiver_test_groups': 300,
                        'receiver_cal_groups': {'cal_A': 300, 'cal_B': 300},
                        'natural_groups': {'reader_val': 100, 'test': 300},
                        'primary_k': 16, 'alphas': [.01, .05],
                        'test_threshold_tuning': False})
    files = [planned, root / 'projection.pt', root / admitted['selection_name'],
             root / 'admitted_representation.json', root / rep / 'C1.json',
             root / rep / 'C2.json']
    for pointer in (root / rep / 'training').glob('*.json'):
        files += [pointer, Path(json.loads(pointer.read_text())['checkpoint'])]
    files += list((root / rep / 'calibration').glob('*.json'))
    files += list((root / rep / 'baselines').glob('*.json'))
    files += list(Path('clr').rglob('*m71*.py')) + [Path('clr/audit/features.py')]
    files += list(Path('scripts').glob('m71*.py'))
    files += [Path('clr/sources/kv_relay.py'), Path('clr/data/m7.py')]
    files += list(Path('configs/M71').glob('*.yaml'))
    write_json(root / 'final_lock.json', {'sha256': {str(p): sha256(p) for p in sorted(set(files))},
                                         'test_opened': False})
    live(stage='final_models_and_thresholds_locked', test_opened=False)
    source_parts(c, source, ['test'], rep + '_test')
    receiver.update(source_splits=['test'], subset_counts={'test': 300})
    finish([job('scripts/m71_diagnostics.py', receiver, rep + '_test_receiver')])
    for kind in ['B3', 'linear', 'B0', 'CLR', 'supervised']:
        configs = [v for v in final_configs if v['kind'] == kind]
        finish([job('scripts/m71_evaluate.py', values,
                    rep + '_final_' + kind + '_' + str(values['seed']))
                for values in configs])
    for kind in ['norm', 'text']:
        finish([job('scripts/m71_baselines.py', {
            'milestone': 'M7.1', 'output': c['output'], 'representation': rep,
            'mode': 'final', 'kind': kind,
            'source_splits': ['reader_val', 'test'] if kind == 'norm' else ['test']},
            rep + '_final_' + kind)])
    natural = {**receiver, 'diagnostic': 'natural_commitment',
               'source_splits': ['reader_val', 'test'],
               'subset_counts': {'reader_val': 100, 'test': 300}}
    finish([job('scripts/m71_diagnostics.py', natural, rep + '_natural_source')])
    finish([job('scripts/m71_natural_eval.py',
                {**inference, 'seed': seed, 'device': device},
                rep + '_natural_CLR_' + str(seed))
            for seed, device in zip(c['seeds'], c['devices'])])
    live(stage='main_comparison_complete_pending_statistics', test_opened=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    main(load_config(a.config, a.overrides))

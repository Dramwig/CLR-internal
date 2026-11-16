"""Continue the registered C1 branch without opening any final-test data.

The sole alternative keeps the selected layers and retains the generator's
critical-record tokens. Failed candidate arrays are disposable caches, while
their manifests, hashes, probes, checkpoints and metrics remain evidence.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from clr.audit.features import sha256
from clr.audit.m71_runtime import live
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def launch(script, config, label):
    path = Path('reports/M71_evidence') / (label + '.log')
    with path.open('x') as handle:
        subprocess.run([sys.executable, script, str(config)], stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def config_file(label, values):
    path = Path('configs/M71') / (label + '.yaml')
    with path.open('x') as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
    return path


def release_failed_candidate(root, representation):
    folder = root / representation
    c1 = json.loads((folder / 'C1.json').read_text())
    if c1['passed']:
        raise ValueError('Cannot discard the admitted candidate')
    # All C1 processes have exited before C1.json is written by the pipeline.
    manifests = sorted((folder / 'source').glob('*/part_*/manifest.jsonl'))
    retained = []
    for manifest in manifests:
        for entry in map(json.loads, manifest.open()):
            path = Path(entry['file'])
            if sha256(path) != entry['sha256']:
                raise ValueError('Failed candidate cache SHA mismatch: ' + str(path))
            retained.append({'path': str(path), 'sha256': entry['sha256'],
                             'bytes': path.stat().st_size})
    record = folder / 'discarded_array_cache.json'
    if record.exists():
        raise FileExistsError(str(record))
    write_json(record, {'reason': 'C1 failed; preregistered alternative needs disk',
                        'recomputable': True, 'files': retained,
                        'bytes': sum(item['bytes'] for item in retained),
                        'retained_evidence': 'manifests, probes, checkpoints, predictions, metrics, source/config snapshots'})
    for item in retained:
        Path(item['path']).unlink()
    live(discarded_candidate=representation,
         discarded_array_cache=str(record))


def main(c):
    root = Path(c['output'])
    initial = root / c['initial'] / 'C1.json'
    while not initial.exists():
        time.sleep(20)
    result = json.loads(initial.read_text())
    selected = c['initial']
    if not result['passed']:
        live(stage='registered_alternative_preparing', C1_passed=False,
             alternative='critical_record_tokens', test_opened=False)
        release_failed_candidate(root, c['initial'])
        # Preserve both the original point check and its uncertainty before
        # the alternative writes the same common smoke/interval pointers.
        for name in ['source_smoke.json', 'projection_intervals.json']:
            path = root / name
            if path.exists():
                target = root / (c['initial'] + '_' + name)
                if target.exists():
                    raise FileExistsError(str(target))
                shutil.copyfile(path, target)
        source = yaml.safe_load(Path(c['source_config']).read_text())
        selection = json.loads((root / c['selection_name']).read_text())
        source.update(mode='smoke', representation=c['alternative'],
                      layers=selection['layers'], critical_tokens=True,
                      device=c['source_devices'][0],
                      min_free_GiB=c['source_min_free_GiB'][0],
                      include_controls=False, selection_name=c['alternative_selection'])
        cfg = config_file(c['tag'] + '_smoke', source)
        launch('scripts/m71_source.py', cfg, c['tag'] + '_smoke')
        scan = yaml.safe_load(Path(c['scan_config']).read_text())
        scan.update(critical_tokens=True,
                    fixed_selection_file=str(root / c['selection_name']),
                    selection_name=c['alternative_selection'],
                    device=c['source_devices'][0],
                    min_free_GiB=c['source_min_free_GiB'][0])
        cfg = config_file(c['tag'] + '_scan', scan)
        launch('scripts/m71_scan.py', cfg, c['tag'] + '_scan')
        alt = json.loads((root / c['alternative_selection']).read_text())
        if not alt['projection_loss_below_2pp']:
            live(stage='alternative_projection_failed', test_opened=False)
            rejected = root / c['alternative']
            rejected.mkdir(exist_ok=True)
            write_json(rejected / 'C1.json', {
                'passed': False, 'results': {}, 'test_opened': False,
                'failed_gate': 'projection', 'C1_executed': False,
                'selection': str(root / c['alternative_selection'])})
        else:
            audit = yaml.safe_load(Path(c['projection_audit_config']).read_text())
            audit.update(selection_name=c['alternative_selection'])
            cfg = config_file(c['tag'] + '_projection_audit', audit)
            launch('scripts/m71_projection_audit.py', cfg, c['tag'] + '_projection_audit')
            pipeline = {key: c[key] for key in ['output', 'source_config', 'gate_config',
                         'source_devices', 'source_min_free_GiB', 'gate_devices']}
            pipeline.update(milestone='M7.1', representation=c['alternative'],
                            tag=c['tag'], selection_name=c['alternative_selection'],
                            critical_tokens=True, resume=False)
            cfg = config_file(c['tag'] + '_pipeline', pipeline)
            launch('scripts/m71_pipeline.py', cfg, c['tag'] + '_pipeline')
        selected = c['alternative']
        result = json.loads((root / selected / 'C1.json').read_text())
    if result['passed']:
        live(stage='C1_admitted_pending_C2', representation=selected,
             C1_passed=True, test_opened=False)
        write_json(root / 'admitted_representation.json', {
            'representation': selected, 'C1': str(root / selected / 'C1.json'),
            'selection_name': c['alternative_selection'] if selected == c['alternative'] else c['selection_name'],
            'critical_tokens': selected == c['alternative'], 'test_opened': False})
    else:
        stop = {'milestone': 'M7.1', 'output': c['output'],
                'attempts': [c['initial'], c['alternative']],
                'selection_name': c['selection_name']}
        cfg = config_file(c['tag'] + '_stop', stop)
        launch('scripts/m71_stop_report.py', cfg, c['tag'] + '_stop')
        cfg = config_file(c['tag'] + '_freeze', {
            'milestone': 'M7.1', 'output': c['output']})
        launch('scripts/m71_freeze.py', cfg, c['tag'] + '_freeze')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    main(load_config(a.config, a.overrides))

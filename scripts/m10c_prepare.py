"""Freeze read-only measurement inputs and preserve superseded paper bytes."""
import argparse
import json
import shutil
from pathlib import Path
import yaml
from clr.paper.m10c_cost import sha


def main(config):
    c = yaml.safe_load(Path(config).read_text()); root = Path(c['output']); root.mkdir(exist_ok=True, parents=True)
    archive = root / 'm10b_paper_snapshot'; archive.mkdir(exist_ok=True)
    if not (archive / 'mapping.json').exists():
        mapping = {}
        for p in [p for p in Path('paper').rglob('*') if p.is_file()] + [Path('reports/FINAL_SUMMARY.md')]:
            dest = archive / p; dest.parent.mkdir(exist_ok=True, parents=True); shutil.copyfile(p, dest)
            mapping[str(p.resolve())] = dict(archive=str(dest), sha256=sha(p))
        (archive / 'mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    locked = {}; manifest_sha = {}
    for mp in c['manifests']:
        d = json.loads(Path(mp).read_text()); manifest_sha[mp] = sha(mp)
        for p, h in d.get('sha256', d.get('files', {})).items():
            if isinstance(h, str): locked[str(Path(p).resolve())] = h
    m1 = json.loads(Path('reports/M1_completed_v3_manifest.json').read_text())
    data = json.loads(Path(m1['data_manifest']).read_text()); manifest_sha[m1['data_manifest']] = sha(m1['data_manifest'])
    for p, h in data.items(): locked[str((Path(c['states']) / p).resolve())] = h
    old = json.loads(Path('data/M10b_v1/freeze.json').read_text())['sha256']
    drift = [dict(path=p, expected=h, actual=sha(p)) for p, h in old.items() if sha(p) != h]
    if any(x['path'] not in ['data/M10b_v1/controller_monitored.log', 'docs/M10b_LIVE.json'] for x in drift):
        raise ValueError('Unexpected M10b drift: ' + str(drift))
    (root / 'preflight_integrity.json').write_text(json.dumps(dict(M10b_changed=drift,
        note='Existing completion stdout appended after freeze; LIVE status was corrected after freeze in prior turn. No model/data/metric drift; original freeze unchanged.',
        manifest_sha256=manifest_sha), indent=2) + '\n')
    measure = {k: v for k, v in c.items() if k != 'manifests'}
    refs = {k: json.loads(Path(c[k + '_ref']).read_text()) for k in ['b5c', 'b6']}
    paths = [c['clr_checkpoint'], c['b5c_ref'], c['b6_ref'], c['normalization'], c['qa'] + '/reader_val.parquet',
             c['states'] + '/reader_val/manifest.parquet', c['templates'], refs['b5c']['checkpoint']]
    paths += [v['path'] for k, v in refs['b6']['checkpoints'].items() if k in ['Q3', 'Q4']]
    paths += [str(Path(c['b3_run']) / f'B3_{q}_11.pt') for q in ['Q3', 'Q4']]
    measure['locked_inputs'] = {}
    for p in paths:
        h = sha(p)
        if locked.get(str(Path(p).resolve())) != h: raise ValueError('Not frozen: ' + p)
        measure['locked_inputs'][p] = h
    for mode, size, batches, loops in [('smoke', 4, [1], 1), ('full', 64, [1, 64], 3)]:
        out = Path('configs/M10c') / ('smoke.yaml' if mode == 'smoke' else 'measure.yaml')
        if out.exists(): raise ValueError('Refusing to overwrite measurement config')
        out.write_text(yaml.safe_dump({**measure, 'mode': mode, 'problems': size, 'batches': batches, 'loops': loops}, sort_keys=False))
    print('Selected input hashes verified:', len(paths), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('config'); main(p.parse_args().config)

"""YAML-driven source shards, merge and full-size C1 orchestration."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.audit.m71_runtime import live


def launch(script,config,label):
    path=Path('reports/M71_evidence')/(label+'.log')
    stream=path.open('x')
    process=subprocess.Popen([sys.executable,script,str(config)],stdout=stream,stderr=subprocess.STDOUT)
    return process,stream,path


def finish(jobs):
    pending=list(jobs)
    while pending:
        for job in list(pending):
            process,stream,path=job
            code=process.poll()
            if code is None:continue
            stream.close();pending.remove(job)
            if code:
                # Other shards may complete and remain valid/resumable.
                for other,handle,log in pending:
                    other.wait();handle.close()
                raise RuntimeError(f'{path}: exit {code}')
        if pending:time.sleep(20)


def main(c):
    root=Path(c['output']);representation=c['representation'];tag=c['tag']
    selection_path=root/c['selection_name']
    while not selection_path.exists():
        progress=Path('docs/M71_LIVE.json')
        status=json.loads(progress.read_text()) if progress.exists() else {}
        if status.get('stage')=='layer_scan_failed':raise RuntimeError('Layer scan failed')
        time.sleep(20)
    selection=json.loads(selection_path.read_text())
    if not selection['projection_loss_below_2pp']:raise RuntimeError('Projection loss gate failed')
    base=yaml.safe_load(Path(c['source_config']).read_text())
    base.update(mode='full',representation=representation,layers=selection['layers'],
                source_splits=['reader_train','reader_val'],partitions=len(c['source_devices']),
                max_seconds=14400,critical_tokens=c['critical_tokens'],selection_name=c['selection_name'],
                resume=c.get('resume',False),include_controls=False)
    jobs=[]
    for part,device in enumerate(c['source_devices']):
        shard={**base,'partition':part,'device':device,'min_free_GiB':c['source_min_free_GiB'][part]}
        cfg=Path('configs/M71')/f'{tag}_source_{part}.yaml'
        cfg.write_text(yaml.safe_dump(shard,sort_keys=False))
        jobs.append(launch('scripts/m71_source.py',cfg,f'{tag}_source_{part}'))
    live(stage='C1_source_shards_running',representation=representation)
    finish(jobs)
    merge=Path('configs/M71')/f'{tag}_merge.yaml';merge.write_text(yaml.safe_dump(base,sort_keys=False))
    finish([launch('scripts/m71_merge.py',merge,f'{tag}_merge')])
    gates=[]
    for kind,device in zip(['linear','B3'],c['gate_devices']):
        gate=yaml.safe_load(Path(c['gate_config']).read_text());gate.update(representation=representation,kind=kind,device=device)
        cfg=Path('configs/M71')/f'{tag}_gate_{kind}.yaml';cfg.write_text(yaml.safe_dump(gate,sort_keys=False))
        gates.append(launch('scripts/m71_gate.py',cfg,f'{tag}_gate_{kind}'))
    finish(gates)
    results={kind:json.loads((root/representation/'gates'/(kind+'.json')).read_text()) for kind in ['linear','B3']}
    passed=all(row['passed'] for row in results.values())
    write_json(root/representation/'C1.json',{'passed':passed,'results':results,'test_opened':False})
    live(stage='C1_passed' if passed else 'C1_failed',representation=representation,C1_passed=passed)
    print(json.dumps({'passed':passed,'results':results}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

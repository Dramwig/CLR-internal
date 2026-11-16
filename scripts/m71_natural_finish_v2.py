"""Merge exact fixed diagnostic shards, evaluate A3, then close final report."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time
from pathlib import Path
from clr.audit.m71_runtime import assert_final_lock,live
from clr.data.m71 import examples,fixed_hash_subset
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def execute(script,config,label):
    stream=Path('reports/M71_evidence')/(label+'.log')
    with stream.open('x') as f:subprocess.run([sys.executable,script,config],stdout=f,stderr=subprocess.STDOUT,check=True)

def main(c):
    root=Path(c['output']);assert_final_lock(c)
    pointers=[root/'natural_partition_v2'/f'part_{i}.json' for i in range(c['partitions'])]
    val=root/'pool/source/reader_val/natural_commitment_completed.json'
    while not (val.exists() and all(p.exists() for p in pointers)):time.sleep(10)
    rows=[]
    for pointer in pointers:
        info=json.loads(pointer.read_text());assert info['gate']['passed']
        while json.loads((Path(info['run'])/'metrics.json').read_text())['status']=='running':time.sleep(1)
        assert json.loads((Path(info['run'])/'metrics.json').read_text())['status']=='completed'
        rows.extend(json.loads(x) for x in Path(info['file']).open())
    selected=set(fixed_hash_subset([e.meta['group_id'] for e in examples(c,'test')],300))
    assert len(rows)==300 and len({r['id'] for r in rows})==300 and {r['group_id'] for r in rows}==selected
    folder=root/'pool/source/test';entries=[json.loads(x) for x in (folder/'manifest.jsonl').open()]
    order={r['id']:i for i,r in enumerate(entries)};rows.sort(key=lambda r:order[r['id']])
    reserved=folder/'natural_commitment.jsonl'
    assert reserved.read_text()==''
    with reserved.open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    write_json(folder/'natural_commitment_completed.json',{'groups':300,'diagnostic_only':True,'partition_pointers':[str(p) for p in pointers]})
    # Original process closes through its FileExistsError ownership guard, so all
    # completed val cost/forward counts are recorded without interrupting a call.
    old=Path(c['original_run'])
    while json.loads((old/'metrics.json').read_text())['status']=='running':time.sleep(1)
    jobs=[]
    for seed,config in c['natural_eval_configs'].items():
        f=(Path('reports/M71_evidence')/f'pool_natural_CLR_{seed}_v2.log').open('x')
        jobs.append((subprocess.Popen([sys.executable,'scripts/m71_natural_eval.py',config],stdout=f,stderr=subprocess.STDOUT),f))
    for process,f in jobs:
        result=process.wait();f.close()
        if result:raise RuntimeError('Natural A3 evaluation failed')
    live(stage='main_comparison_complete_pending_statistics',test_opened=True)
    import shutil
    startup=root/'upstream_verification_startup.json'
    if not startup.exists():shutil.copyfile(root/'upstream_verification.json',startup)
    execute('scripts/m71_verify.py',c['verify_config'],'upstream_verify_final_v2')
    while not (root/'report_wait_natural_v2_completed.json').exists():time.sleep(10)
    execute('scripts/m71_freeze.py',c['freeze_config'],'freeze_final_v2')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

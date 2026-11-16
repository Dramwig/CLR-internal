"""Read-only upstream integrity and independent non-test split audit."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.audit.features import sha256
from clr.audit.m8c_runtime import snapshot,Cost
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);results=[]
        for file in c['freeze_manifests']:
            doc=json.loads(Path(file).read_text());hashes=doc.get('sha256',doc.get('files'))
            with ThreadPoolExecutor(max_workers=2) as pool:current=list(pool.map(sha256,hashes))
            bad=[p for (p,h),now in zip(hashes.items(),current) if h!=now]
            results.append({'manifest':file,'checked':len(hashes),'mismatches':bad,'passed':not bad})
        allsets={}
        for split,count in [('reader_train',4000),('reader_val',200),('cal_A',300),('cal_B',1000)]:
            entries=list(map(json.loads,(Path(c['output'])/'source'/split/'manifest.jsonl').read_text().splitlines()))
            ids={e['group_id'] for e in entries};assert len(entries)==count and len(ids)==count
            for other in allsets.values():assert not ids&other
            allsets[split]=ids
        write_json(Path(c['output'])/c['verification_name'],{'run':str(run.path),'results':results,'passed':all(r['passed'] for r in results),'split_counts':{k:len(v) for k,v in allsets.items()}})
        cost.update(status='completed');assert all(r['passed'] for r in results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

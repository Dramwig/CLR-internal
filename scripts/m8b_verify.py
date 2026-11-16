"""Read-only integrity of frozen upstream sources and results."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from clr.audit.features import sha256
from clr.audit.m8b_runtime import snapshot,Cost
from clr.utils.records import recorded
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);results=[]
        for file in c['freeze_manifests']:
            doc=json.loads(Path(file).read_text());hashes=doc.get('sha256',doc.get('files'));bad=[]
            for path,h in hashes.items():
                if not Path(path).exists() or sha256(path)!=h:bad.append(path)
            results.append({'manifest':file,'checked':len(hashes),'mismatches':bad,'passed':not bad})
        write_json(run.path/'verification.json',results);write_json(Path(c['output'])/c['verification_name'],{'run':str(run.path),'results':results,'passed':all(r['passed'] for r in results)})
        cost.update(status='completed');assert all(r['passed'] for r in results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

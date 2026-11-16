"""Read-only upstream freeze verification, isolated from M7.1 model results."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.audit.features import sha256
from clr.audit.m71_runtime import snapshot
from clr.sources.kv_relay import verify_idea_a
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def main(c):
    with recorded(c) as run:
        snapshot(run);results={}
        for name in c['freezes']:
            spec=json.loads(Path(name).read_text());files=spec.get('sha256',spec.get('files'))
            def check(item):
                path,expected=item
                try:actual=sha256(path)
                except OSError as error:return {'path':path,'error':repr(error)}
                return {'path':path,'expected':expected,'actual':actual} if actual!=expected else None
            with ThreadPoolExecutor(max_workers=2) as executor:
                mismatches=[value for value in executor.map(check,files.items()) if value]
            results[name]={'freeze_sha256':sha256(name),'files':len(files),'mismatches':mismatches}
            print(name,len(files),len(mismatches),flush=True)
        result={'run':str(run.path),'upstream':results,'generator':verify_idea_a(c['idea_a']),
                'passed':all(not r['mismatches'] for r in results.values())}
        write_json(Path(c['output'])/'upstream_verification.json',result)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,passed=result['passed'])
        if not result['passed']:raise ValueError('Upstream frozen artifacts changed; see verification evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

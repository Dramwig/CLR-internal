"""Recorded read-only audit of all upstream frozen artifacts, before M7 final test."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from scripts.m7_freeze import digest
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot

def main(c):
    with recorded(c) as run:
        snapshot(run);results={};cache={}
        for name in ['data/M23_v1a/freeze.json','data/M4_v0/freeze.json','data/M41_v1/freeze.json','data/M5a_v13/freeze.json','data/M5b_v1/freeze.json']:
            d=json.loads(Path(name).read_text());bad=[]
            for p,expected in d['sha256'].items():
                if p not in cache:cache[p]=digest(p) if Path(p).exists() else None
                if cache[p]!=expected:bad.append(p)
            results[name]={'files':len(d['sha256']),'mismatches':bad};print(name,len(d['sha256']),len(bad),flush=True)
        write_json(Path(c['output'])/'upstream_pretest_verification.json',{'run':str(run.path),'verification':results})
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,verification=results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

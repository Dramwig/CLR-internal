"""Read-only inventory and integrity of all pre-M9 freeze manifests."""
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m9_runtime import snapshot

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main(c):
 with recorded(c) as run:
  snapshot(run,c);entries=[];expect={}
  for p in c['manifests']:
   d=json.loads(Path(p).read_text());hh=d.get('sha256',d.get('files',{}));hh=hh if isinstance(hh,dict) else {}
   entries.append({'manifest':p,'sha256':digest(p),'files':len(hh)})
   for name,h in hh.items():
    if isinstance(h,str):expect.setdefault(str(Path(name).resolve()),[]).append((p,h))
  names=sorted(expect)
  def check(p):
   try:now=digest(p);return [(m,p,h,now) for m,h in expect[p] if h!=now]
   except Exception as e:return [(m,p,h,str(e)) for m,h in expect[p]]
  with ThreadPoolExecutor(max_workers=2) as pool:bad=sum(pool.map(check,names),[])
  write_json(Path(c['output'])/'upstream_audit.json',{'run':str(run.path),'manifest_inventory':entries,'unique_files':len(names),'mismatches':bad,'passed':not bad})
  run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,checked=len(names),mismatches=len(bad));print('checked',len(names),'mismatches',len(bad),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

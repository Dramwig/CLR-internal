"""Verify upstream bytes; authorized paper supersessions resolve to exact archives."""
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main(c):
 inventory=json.loads(Path('data/M9_v1/upstream_audit.json').read_text())['manifest_inventory'];manifests=[x['manifest'] for x in inventory]+['data/M9_v1/freeze.json'];mapping=json.loads(Path(c['archive_mapping']).read_text());archives={str(Path(p).resolve()):v for p,v in mapping.items()};expect={};records=[]
 for p in manifests:
  data=json.loads(Path(p).read_text());hh=data.get('sha256',data.get('files',{}));hh=hh if isinstance(hh,dict) else {};records.append({'manifest':p,'sha256':sha(p),'files':len(hh)})
  for file,h in hh.items():
   if isinstance(h,str):expect.setdefault(str(Path(file).resolve()),[]).append((p,h))
 def check(item):
  p,versions=item
  try:current=sha(p)
  except Exception as e:current=str(e)
  bad=[];superseded=[]
  for manifest,h in versions:
   if current==h:continue
   a=archives.get(p)
   if a and a['sha256']==h and sha(a['archive'])==h:superseded.append({'manifest':manifest,'original_path':p,'archive':a['archive'],'original_sha256':h,'current_sha256':current,'authorization':'M10 user requests paper/FINAL_SUMMARY updates; original bytes preserved'})
   else:bad.append({'manifest':manifest,'path':p,'expected_sha256':h,'current_sha256':current})
  return bad,superseded
 with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(check,expect.items()))
 bad=[r for a,b in rows for r in a];sup=[r for a,b in rows for r in b];write_json(c['result'],{'passed':not bad,'unique_files':len(expect),'manifest_inventory':records,'mismatches':bad,'authorized_supersessions':sup})
 print(len(expect),'files;',len(bad),'mismatches;',len(sup),'authorized paper supersessions')
 if bad:raise ValueError('Upstream bytes changed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Independent clean plotting environment; byte-for-byte M10 paper rebuild."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(c):
 for config in [c['primary_build'],c['repeat_build']]:subprocess.run([c['python'],'-I','scripts/paper/make_all.py',config],check=True)
 records=[]
 for sub in ['tables','figures','derived']:
  left=sorted((Path(c['primary_root'])/sub).glob('*'));right=sorted((Path(c['repeat_root'])/sub).glob('*'))
  if {p.name for p in left}!={p.name for p in right}:raise ValueError('Paper artifact inventories differ')
  for p in left:
   q=Path(c['repeat_root'])/sub/p.name;records.append(dict(artifact=str(p),sha256=sha(p),repeat_sha256=sha(q),byte_identical=p.read_bytes()==q.read_bytes()))
 if not all(r['byte_identical'] for r in records):raise ValueError('Nonidentical paper rebuild')
 probe=subprocess.run([c['python'],'-I','-c','import importlib.util,json; print(json.dumps({k:importlib.util.find_spec(k) is not None for k in ["torch","transformers","numpy","pandas","matplotlib","yaml"]}))'],capture_output=True,text=True,check=True);modules=json.loads(probe.stdout)
 if modules['torch'] or modules['transformers']:raise ValueError('Plotting environment is not independent')
 write_json(Path(c['output'])/'reproducibility.json',{'passed':True,'artifacts':records,'environment_modules':modules,'method':'clean independent .venv-paper, Python -I, separate output roots; all bytes identical'})
 print(len(records),'byte-identical products')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

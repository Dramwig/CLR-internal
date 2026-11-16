"""Preserve exact M9 current-paper bytes before authorized M10 supersession."""
import argparse,hashlib,json,shutil
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 dest=Path(c['archive']);index=dest/'mapping.json'
 if index.exists():
  d=json.loads(index.read_text())
  for r in d.values():
   if hashlib.sha256(Path(r['archive']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Archived M9 bytes changed')
  return
 mapping={};paths=sorted(Path('paper').rglob('*'))+[Path('reports/FINAL_SUMMARY.md')]
 for p in paths:
  if not p.is_file():continue
  q=dest/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);mapping[str(p)]={'archive':str(q),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 write_json(index,mapping);print(len(mapping),'original files preserved')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

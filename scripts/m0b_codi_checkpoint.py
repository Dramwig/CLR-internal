"""Download and verify the pinned public CODI training checkpoint."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
for key,value in c['runtime'].items():os.environ[key]=str(value)
from huggingface_hub import snapshot_download
with recorded(c) as run:
 t=time.perf_counter()
 try:
  meta=json.loads(Path(c['metadata']).read_text());weight=next(r for r in meta if r['path']=='pytorch_model.bin')
  if shutil.disk_usage(c['cache_dir']).free<weight['size']+c['reserve_bytes']:raise RuntimeError('Insufficient disk space before download')
  write_json(run.path/'resources.json',{'repo':c['repo'],'revision':c['revision'],'metadata':meta})
  path=Path(snapshot_download(c['repo'],revision=c['revision'],allow_patterns=c['files'],cache_dir=c['cache_dir'],max_workers=2))
  file=path/'pytorch_model.bin';h=hashlib.sha256()
  with file.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  if file.stat().st_size!=weight['size'] or h.hexdigest()!=weight['lfs']['oid']:raise RuntimeError('Official checkpoint SHA256/size mismatch')
  write_json(run.path/'verified.json',{'path':str(file),'sha256':h.hexdigest(),'bytes':file.stat().st_size})
  run.write_metrics(status='completed',path=str(path),sha256=h.hexdigest(),bytes=file.stat().st_size)
  print(run.path,flush=True)
 finally:run.write_metrics(wall_seconds=time.perf_counter()-t,gpu_hours=0,source_forward_calls=0)

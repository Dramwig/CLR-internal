"""Pinned ModelScope downloads; verify every file and upstream identity metadata."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
from urllib.parse import quote
import requests
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main(c):
 with recorded(c) as run:
  started=time.perf_counter()
  try:
   listing=json.loads(Path(c['manifest']).read_text())['Data']['Files'];files={r['Path']:r for r in listing if r['Type']=='blob'}
   upstream={r['path']:r for r in json.loads(Path(c['upstream_metadata']).read_text())}
   root=Path(c['destination']);root.mkdir(parents=True,exist_ok=True)
   needed=sum(files[n]['Size'] for n in c['files'] if not (root/n).exists())
   if shutil.disk_usage(root).free<needed+c['reserve_bytes']:raise RuntimeError('Insufficient disk space for pinned download and reserve')
   write_json(run.path/'resources.json',{'repo':c['repo'],'revision':c['revision'],'files':[files[n] for n in c['files']],'source':'ModelScope only; upstream metadata fetched separately, no HF file download'})
   results=[]
   for name in c['files']:
    entry=files[name];dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_name(dest.name+'.incomplete')
    if not dest.exists():
     offset=partial.stat().st_size if partial.exists() else 0
     if offset>entry['Size']:raise RuntimeError('Oversized partial file')
     if offset<entry['Size']:
      url=f"https://modelscope.cn/api/v1/models/{c['repo']}/repo?Revision={quote(c['revision'])}&FilePath={quote(name)}"
      with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,180)) as response:
       response.raise_for_status()
       if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Server did not honor resume range; partial retained')
       with partial.open('ab' if offset else 'xb') as f:
        for chunk in response.iter_content(4*1024*1024):
         if chunk:f.write(chunk)
     if partial.stat().st_size!=entry['Size'] or sha(partial)!=entry['Sha256']:raise RuntimeError(f'SHA/size mismatch: {name}')
     partial.rename(dest)
    digest=sha(dest)
    if dest.stat().st_size!=entry['Size'] or digest!=entry['Sha256']:raise RuntimeError(f'Existing file verification failed: {name}')
    proof={'path':name,'size':entry['Size'],'sha256':digest,'modelscope_verified':True}
    if name in upstream:
     remote=upstream[name]
     if remote.get('lfs'):
      blob=f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {entry['Size']}\n".encode()
      proof['upstream_comparison']='Git LFS pointer blob'
     else:blob=dest.read_bytes();proof['upstream_comparison']='Git file blob'
     oid=hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest();proof['upstream_matches']=oid==remote['oid'];proof['upstream_blob_oid']=remote['oid'];proof['computed_blob_oid']=oid
    results.append(proof);write_json(run.path/'files_verified.json',results);run.write_metrics(files_verified=len(results),bytes_verified=sum(r['size'] for r in results));print('verified',name,proof.get('upstream_matches'),flush=True)
   write_json(root/'CLR_DOWNLOAD_MANIFEST.json',{'run':str(run.path),'repo':c['repo'],'revision':c['revision'],'files':results})
   critical=[r for r in results if r['path'] in c['identity_required']]
   if len(critical)!=len(c['identity_required']) or not all(r.get('upstream_matches') for r in critical):raise RuntimeError('Core files differ from pinned upstream metadata; inspect before use')
   run.write_metrics(status='completed',destination=str(root),upstream_core_identity_verified=True)
   print(run.path,flush=True)
  finally:run.write_metrics(wall_seconds=time.perf_counter()-started,gpu_hours=0,source_forward_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

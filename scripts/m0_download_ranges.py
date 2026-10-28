"""Same official HF revision, validated HTTP ranges and LFS SHA-256; resumable."""
import argparse,concurrent.futures,hashlib,json,os,time
from pathlib import Path
import requests
from huggingface_hub import HfApi,get_hf_file_metadata,hf_hub_url
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
 with recorded(c) as run:
    start=time.perf_counter();root=Path(c['destination']);root.mkdir(parents=True,exist_ok=True);manifest=[];jobs=[]
    info=HfApi().model_info(c['repo'],revision=c['revision'])
    for entry in info.siblings:
        name=entry.rfilename
        if not (name.endswith(('.json','.py','.safetensors')) or name=='README.md'):continue
        url=hf_hub_url(c['repo'],name,revision=c['revision']);meta=get_hf_file_metadata(url);dst=root/name
        item={'name':name,'size':meta.size,'etag':meta.etag};manifest.append(item)
        if meta.size<10_000_000:
            response=requests.get(url,timeout=60);response.raise_for_status()
            if len(response.content)!=meta.size:raise ValueError('Small file size mismatch')
            dst.write_bytes(response.content)
            continue
        for offset in range(0,meta.size,c['chunk_bytes']):
            end=min(offset+c['chunk_bytes'],meta.size)-1
            jobs.append((url,offset,end,meta.size,root/(name+f'.part{offset}')))
    def download(job):
        url,start,end,size,path=job
        if path.exists() and path.stat().st_size==end-start+1:return
        response=requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90))
        response.raise_for_status()
        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('Server did not honor byte range')
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('wb') as f:
            for chunk in response.iter_content(1024*1024):f.write(chunk)
        if temp.stat().st_size!=end-start+1:raise ValueError('Incomplete range')
        temp.rename(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for n,_ in enumerate(pool.map(download,jobs)):
            if n%8==0:print(f'completed {n+1}/{len(jobs)} ranges',flush=True)
    for item in manifest:
        if item['size']<10_000_000:continue
        name=item['name'];sha=hashlib.sha256();temp=root/(name+'.assembling')
        with temp.open('wb') as out:
            for offset in range(0,item['size'],c['chunk_bytes']):
                part=root/(name+f'.part{offset}')
                with part.open('rb') as f:
                    for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk);out.write(chunk)
        if sha.hexdigest()!=item['etag']:raise ValueError(f'SHA256 mismatch: {name}')
        temp.rename(root/name)
        for offset in range(0,item['size'],c['chunk_bytes']):(root/(name+f'.part{offset}')).unlink()
    write_json(root/'clr_resource_manifest.json',{'repo':c['repo'],'revision':c['revision'],'files':manifest})
    write_json(run.path/'resources.json',manifest)
    run.write_metrics(status='completed',path=str(root),wall_seconds=time.perf_counter()-start,gpu_hours=0,source_forward_calls=0)
    print(root)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

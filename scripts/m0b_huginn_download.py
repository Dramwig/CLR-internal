"""Resume the pinned official Huginn snapshot; validate every LFS object."""
import argparse
import hashlib
import os
from pathlib import Path
import time

# Disable the Xet transport's nested parallelism. Four HTTP object downloads
# means the configured concurrency really stays <=8, including large objects.
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi, snapshot_download
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json


def main(c):
    if not 1 <= c['max_workers'] <= 8:
        raise ValueError('M0b download concurrency must be in [1,8]')
    with recorded(c) as run:
        start=time.perf_counter()
        run.logger.log('resume_start',cache_dir=c['cache_dir'],max_workers=c['max_workers'],transport='official_hf_http')
        info=HfApi().model_info(c['repo'],revision=c['revision'],files_metadata=True)
        if info.sha != c['revision']:
            raise ValueError('Unexpected revision returned by HF')
        lfs=[{'file':s.rfilename,'bytes':s.size,'sha256':s.lfs.sha256} for s in info.siblings if s.lfs]
        write_json(run.path/'official_lfs_manifest.json',lfs)
        root=Path(snapshot_download(c['repo'],revision=c['revision'],cache_dir=c['cache_dir'],
                                   allow_patterns=c['allow_patterns'],resume_download=True,max_workers=c['max_workers']))
        verified=[]
        for entry in lfs:
            # Asset GIF/JPEG are outside the requested inference snapshot.
            if not entry['file'].endswith('.safetensors'):
                continue
            path=root/entry['file'];digest=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):
                    digest.update(block)
            if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Official LFS size/SHA256 mismatch: {entry["file"]}')
            verified.append(entry)
            run.logger.log('lfs_verified',**entry)
        if len(verified)!=4:
            raise ValueError('All four official weight shards must be verified')
        write_json(run.path/'verified_manifest.json',{'repo':c['repo'],'revision':c['revision'],'path':str(root),'files':verified})
        run.write_metrics(status='completed',verified_shards=len(verified),snapshot=str(root),wall_seconds=time.perf_counter()-start,gpu_hours=0,source_forward_calls=0)
        print(run.path,root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('overrides',nargs='*')
    args=parser.parse_args();main(load_config(args.config,args.overrides))

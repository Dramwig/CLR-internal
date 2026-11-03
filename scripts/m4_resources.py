"""Pinned official encoder resources; no model substitution or source forwards."""
import argparse,hashlib,time
from pathlib import Path
from huggingface_hub import snapshot_download,HfApi
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
 with recorded(c) as run:
    info=HfApi().model_info(c['model_id'],revision=c['revision'],files_metadata=True)
    if info.sha!=c['revision']:raise ValueError('Encoder revision mismatch')
    files=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md']
    root=Path(snapshot_download(c['model_id'],revision=c['revision'],allow_patterns=files,local_dir=c['model_path'],max_workers=4))
    evidence={}
    for name in files:
        p=root/name;h=hashlib.sha256(p.read_bytes()).hexdigest();remote=next(s for s in info.siblings if s.rfilename==name)
        if remote.lfs and remote.lfs.sha256!=h:raise ValueError('LFS SHA mismatch: '+name)
        evidence[name]={'sha256':h,'bytes':p.stat().st_size,'lfs_verified':bool(remote.lfs)}
    write_json(run.path/'resources.json',{'model':c['model_id'],'revision':info.sha,'path':str(root),'files':evidence})
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=0,resources=evidence);print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

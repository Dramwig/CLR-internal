import argparse,hashlib,json,time
from pathlib import Path
from huggingface_hub import hf_hub_download,get_hf_file_metadata,hf_hub_url
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 t=time.perf_counter();path=hf_hub_download(c['repo'],c['file'],repo_type='dataset',revision=c['revision'],cache_dir='/home/me/datasets/.cache/huggingface',resume_download=True)
 meta=get_hf_file_metadata(hf_hub_url(c['repo'],c['file'],repo_type='dataset',revision=c['revision']))
 sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 if len(meta.etag)==64 and sha!=meta.etag:raise RuntimeError('Dataset SHA256 mismatch')
 write_json(run.path/'resource.json',{'path':path,'sha256':sha,'etag':meta.etag,'revision':c['revision']})
 run.write_metrics(status='completed',path=path,sha256=sha,wall_seconds=time.perf_counter()-t,gpu_hours=0,source_forward_calls=0)
 print(path,flush=True)

import argparse,time
from huggingface_hub import snapshot_download
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 t=time.perf_counter()
 path=snapshot_download(c['repo'],revision=c['revision'],allow_patterns=c['files'],cache_dir='/home/me/models/huggingface',max_workers=4)
 write_json(run.path/'resource.json',{'repo':c['repo'],'revision':c['revision'],'path':path})
 run.write_metrics(status='completed',wall_seconds=time.perf_counter()-t,gpu_hours=0,source_forward_calls=0,path=path)
 print(path)

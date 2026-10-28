import argparse,hashlib,json
from pathlib import Path
import requests
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.config import load_config
from clr.data.splits import problem_id,freeze
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
    data={};resources={}
    for split,url in c['urls'].items():
        r=requests.get(url,timeout=60);r.raise_for_status();resources[split]={'url':url,'sha256':hashlib.sha256(r.content).hexdigest()}
        # Exact official Internalize-CoT/Coconut data format: question || steps || answer.
        rows=[]
        for line in r.text.strip().splitlines():
            parts=line.split('||')
            if len(parts)!=2 or '##' not in parts[1]: raise ValueError('Unexpected official dataset format')
            q=parts[0];answer=parts[1].split('##')[-1]
            rows.append({'id':problem_id(q),'question':q.strip(),'answer':answer.strip()})
        data[split]=rows
    # Freeze independent val/cal problems; retain official test unchanged.
    held=sorted(data.pop('valid'),key=lambda x:x['id'])
    data['val']=held[::2];data['cal']=held[1::2]
    ids={k:[r['id'] for r in v] for k,v in data.items()}
    sha=freeze('data/splits/gsm8k_M0_v1.json',ids)
    for split,rows in data.items():
        path=Path('data/gsm8k_'+split+'.json')
        content=json.dumps(rows,ensure_ascii=False,indent=2)+'\n'
        if path.exists() and path.read_text()!=content: raise ValueError('Existing data differs')
        path.write_text(content)
    write_json(run.path/'resources.json',resources)
    run.write_metrics(status='completed',split_sha256=sha,counts={k:len(v) for k,v in data.items()},gpu_hours=0,source_forward_calls=0)
    print(run.path)

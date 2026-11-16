"""Read-only environment/resource lock, with an explicit capture time (not backdated)."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,subprocess,importlib.metadata
from pathlib import Path
from datetime import datetime,timezone
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.sources.kv_relay import verify_idea_a

def main(c):
    with recorded(c) as run:
        root=Path(c['idea_a']);core=verify_idea_a(root)
        packages=['torch','transformers','tokenizers','peft','accelerate','huggingface-hub','safetensors','numpy','scipy','scikit-learn','PyYAML']
        ours={p:importlib.metadata.version(p) for p in packages}
        code='import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in '+repr(packages)+'}))'
        idea=json.loads(subprocess.check_output(['/home/me/anaconda3/envs/latdefer/bin/python','-c',code],text=True))
        if ours!=idea:raise AssertionError('M7/Idea A related dependency versions differ')
        main_version=subprocess.check_output(['.venv/bin/python','-c','import transformers;print(transformers.__version__)'],text=True).strip()
        names=['latdef/inject/cache.py','latdef/inject/adapter.py','latdef/inject/transfer.py','latdef/s2/model.py','latdef/s2/budgets.py','latdef/handoff/base.py']
        paths={str(root/p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in names}
        hidden=Path('/home/me/datasets/HiddenBench/1e3c25b1fd798c6717f4df0463edd3825c8e37f9/benchmark.json')
        obj=json.loads(hidden.read_text())
        fmt={'type':type(obj).__name__,'rows':len(obj),'first_item_keys':list(obj[0]) if isinstance(obj,list) and obj and isinstance(obj[0],dict) else [],
             'first_item_field_types':{k:type(v).__name__ for k,v in obj[0].items()} if isinstance(obj,list) and obj and isinstance(obj[0],dict) else {}}
        result={'captured_utc':datetime.now(timezone.utc).isoformat(),'capture_scope':'supplementary lock before final test, not a backdated startup snapshot',
                'generator_core_sha256':core,'idea_a_contract_sha256':paths,'comm_packages':ours,'idea_a_packages':idea,
                'main_transformers_unchanged':main_version,'venv_base':'read-only inherited latdefer site-packages; independent venv overlay',
                'hiddenbench_format':fmt,'hiddenbench_sha256':hashlib.sha256(hidden.read_bytes()).hexdigest()}
        write_json(Path(c['output'])/'dependencies_lock.json',result)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,**result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

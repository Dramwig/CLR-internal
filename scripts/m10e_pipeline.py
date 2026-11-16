"""Resume-safe M10e finish controller; scientific configuration stays in YAML."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def main(config):
    c=load_config(config);root=Path(c['output'])
    assert not (root/'freeze.json').exists(), 'M10e already frozen'
    keys=['gpt2_HSRM','llama_HSRM','gpt2_CLUE','llama_CLUE','llama_B2']
    def update(status,**kw):
        write_json('docs/M10e_LIVE.json',dict(status=status,output=str(root),task='docs/M10e_TASK.md',**kw))
    def run(module,overrides,log):
        cmd=[sys.executable,'-m',module,config,*overrides]
        with Path(log).open('w') as f:
            subprocess.run(cmd,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
    while True:
        done=[]
        for key in keys:
            p=root/f'trained_{key}.json'
            if p.exists():
                d=json.loads(p.read_text());m=json.loads(Path(d['run'],'metrics.json').read_text())
                if m['status']=='completed':done.append(key)
        update('training',completed=done,test_opened=False,budget_gpu_hours=c['gpu_budget_hours'])
        if len(done)==len(keys):break
        time.sleep(15)
    run('scripts.m10e',['stage=lock'],'reports/M10e_lock.log')
    update('locked_before_test',test_opened=False,training_lock=str(root/'training_lock.json'))
    for key in keys:
        p=root/f'final_{key}.json'
        if p.exists():
            d=json.loads(p.read_text())
            if json.loads(Path(d['run'],'metrics.json').read_text())['status']=='completed':continue
        source,method=key.split('_')
        update('final_evaluation',current=key,test_opened=True)
        run('scripts.m10e',['stage=final',f'source={source}',f'method={method}'],f'reports/M10e_final_{key}.log')
    update('statistics',test_opened=True)
    run('scripts.m10e_summarize',[],'reports/M10e_summary.log')
    run('scripts.m10e_audit',[],'reports/M10e_upstream_after.log')
    update('experiments_statistics_complete',test_opened=True,remaining='review, Markdown handoff, final freeze')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(a.config)

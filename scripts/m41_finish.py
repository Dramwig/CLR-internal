"""Validate and freeze M4.1 without altering M4 records."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m41_budget import hours_used

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main(c):
 with recorded(c) as run:
    root=Path(c['output']);report=json.loads((root/'report_complete.json').read_text());metrics=json.loads(Path(report['run'],'metrics.json').read_text())
    if metrics['status']!='completed' or hours_used()>35:raise ValueError('Completion/budget gate failed')
    text=Path('reports/M4.1_REVIEW.md').read_text()
    if len(text.encode())>8000 or any(f'## ({x})' not in text for x in 'abcdef'):raise ValueError('Review format')
    tests=sorted(str(p) for p in Path('tests').glob('test_m41*.py'))
    with (run.path/'tests.log').open('w') as log:subprocess.run(['.venv/bin/python','-m','pytest','-q',*tests],check=True,stdout=log,stderr=subprocess.STDOUT)
    old=json.loads(Path('data/M4_v0/freeze.json').read_text());mismatch=[path for path,digest in old['sha256'].items() if sha(path)!=digest]
    if mismatch:raise ValueError('M4 frozen files changed: '+str(mismatch))
    files=[]
    for pattern in ['clr/models/clr_v1.py','clr/utils/m41*.py','clr/calib/m41*.py','scripts/m41*.py','configs/M41*.yaml','tests/test_m41*.py','reports/M41*','reports/M4.1*','docs/M4.1_TASK.md']:
        for path in Path('.').glob(pattern):
            files.extend(x for x in path.rglob('*') if x.is_file()) if path.is_dir() else files.append(path)
    files += [p for p in root.rglob('*') if p.is_file() and p.name!='freeze.json']
    for p in Path('runs').glob('*/config.json'):
        if p.parent.resolve()!=run.path.resolve() and json.loads(p.read_text()).get('milestone')=='M4.1':files += [f for f in p.parent.rglob('*') if f.is_file()]
    write_json(root/'freeze.json',{'status':'completed_pending_G1_review','sha256':{str(p):sha(p) for p in sorted(set(files))},'input_M4_freeze_sha256':sha('data/M4_v0/freeze.json'),'report':report,'g1_approved':False})
    update='M4.1实验完成，待G1审查；唯一审查文件reports/M4.1_REVIEW.md，冻结data/M41_v1/freeze.json。M4原冻结哈希全部一致。'
    Path('AGENTS.md').open('a').write('\n'+update+'\n');Path('docs/STATUS.md').open('a').write('\n'+update+'\n')
    Path('docs/EXPERIMENTS.md').open('a').write('\n## M4.1 完整实验完成，待审查\n诊断→学习能力门槛→全曝光三seed主比较→val选型迁移→最终校准与Mondrian完成。\n命令：`scripts/m41_pipeline.py configs/M41_pipeline.yaml`；三seed均值±95%CI及10k按题bootstrap见[审查报告](../reports/M4.1_REVIEW.md)。\n全部结果、成本、负面结论与协议偏离见[详细报告](../reports/M4.1_details.md)；冻结data/M41_v1/freeze.json，M4不改。\n')
    Path('docs/BLOCKERS.md').open('a').write('\n## M4.1 收尾\n工程完成不等于G1通过；根因边界、初版合成控制失败、曝光/预算协议与剩余限制见[审查报告](../reports/M4.1_REVIEW.md)。\n')
    write_json(Path('docs/M41_LIVE.json'),{'stage':'completed_pending_G1_review','review':'reports/M4.1_REVIEW.md','g1_approved':False});run.write_metrics(status='completed',gpu_hours=0.,source_forward_calls=0,frozen_files=len(set(files)),g1_approved=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

"""Freeze only verified completed M4 artifacts; never infer G1 approval."""
import argparse,json,hashlib,re,subprocess
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m4_budget import hours_used

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main(c):
 with recorded(c) as run:
    root=Path(c['output']);out=root/'freeze.json'
    if out.exists():raise ValueError('M4 freeze already exists; do not overwrite completed version')
    tests=['tests/test_m4_data.py','tests/test_m4_model.py','tests/test_m4_calibration.py','tests/test_m4_evaluation.py','tests/test_m4_statistics.py']
    with (run.path/'tests.log').open('w') as f:subprocess.run(['.venv/bin/python','-m','pytest',*tests,'-q'],check=True,stdout=f,stderr=subprocess.STDOUT)
    for frozen in ['data/M23_v1a/freeze.json','reports/M1_completed_v3_manifest.json']:
        for p,expected in json.loads(Path(frozen).read_text())['sha256'].items():
            if digest(p)!=expected:raise ValueError(f'Frozen input changed: {p}')
    complete=json.loads((root/'evaluation_complete.json').read_text());training=json.loads((root/'training_complete.json').read_text())
    if not training['all_training_locked']:raise ValueError('Missing training lock')
    for p in complete['runs'].values():
        if json.loads(Path(p,'metrics.json').read_text()).get('status')!='completed':raise ValueError(f'Incomplete evaluation: {p}')
    review=Path('reports/M4_REVIEW.md').read_text()
    if len(review.encode())>8000:raise ValueError('Review exceeds8000bytes')
    if len(re.findall(r'^\|---',review,re.M))>3:raise ValueError('More than3 review tables')
    for letter in 'abcdef':
        if f'## ({letter})' not in review:raise ValueError('Missing required review section')
    if hours_used()>c['total_budget_hours']:raise ValueError('Budget exceeded; record deviation before freeze')
    files=set()
    for glob in ['clr/models/clr_v0.py','clr/data/m4/*.py','clr/baselines/b3_multi_v0.py','clr/calib/m4*.py','clr/utils/m4_budget.py','scripts/m4*.py','configs/M4*.yaml','tests/test_m4*.py','reports/M4*.md','reports/M4*.csv','reports/M4_figures/*']:
        files.update(p for p in Path('.').glob(glob) if p.is_file())
    files.update(p for p in root.rglob('*') if p.is_file() and p!=out)
    files.add(Path(c['unseen_split']))
    for p in Path('runs').glob('*/config.json'):
        cfg=json.loads(p.read_text())
        if cfg.get('milestone')=='M4' and p.parent!=run.path and json.loads((p.parent/'metrics.json').read_text()).get('status')!='running':files.update(x for x in p.parent.rglob('*') if x.is_file())
    write_json(out,{'status':'completed_pending_G1_review','version':'v0','sha256':{str(p):digest(p) for p in sorted(files)},'review':'reports/M4_REVIEW.md','training_runs':training['runs'],'evaluation_runs':complete['runs'],'input_freezes':['data/M23_v1a/freeze.json','reports/M1_completed_v3_manifest.json']})
    p=Path('AGENTS.md');text=p.read_text();text=re.sub(r'<!-- M4_LIVE -->.*?<!-- /M4_LIVE -->','<!-- M4_LIVE -->\nM4 v0实验完成，待G1审查；唯一审查文件 reports/M4_REVIEW.md。代码/数据/结果冻结于 data/M4_v0/freeze.json；未经用户G1审查不启动M5。\n<!-- /M4_LIVE -->',text,flags=re.S);text=re.sub(r'^M4当前：.*\n?','',text,flags=re.M);text=text.replace('M4 已授权并开始','M4 实验完成待 G1 审查');p.write_text(text)
    p=Path('docs/STATUS.md');p.write_text(p.read_text()+'\nM4 v0实验完成，待G1审查。唯一审查文件 `reports/M4_REVIEW.md`；完整报告 `reports/M4_clr_v0.md`；冻结 `data/M4_v0/freeze.json`。M1/M23冻结哈希复核无变化，测试通过；不自动启动M5。\n')
    p=Path('docs/EXPERIMENTS.md');p.write_text(p.read_text()+'''\n## M4 v0 — 完整实验完成，待G1审查
执行：`scripts/m4_train_suite.py configs/M4_train_v0.yaml`，`scripts/m4_pipeline.py configs/M4_pipeline.yaml`（均使用项目Python与PYTHONPATH=.）。
三seed11/22/33全部纳入；均值±95%CI、按题10000次配对bootstrap、核心结果表见 [唯一审查文件](../reports/M4_REVIEW.md)。
完整指标与负面结果见 [完整报告](../reports/M4_clr_v0.md)，所有t/模板/clean-traj结果在 `reports/M4_metrics.csv`。
预算与实际成本见报告；可选消融未执行、未见池重叠/解析失败及修复记录如实保留。完成不等于G1通过。
冻结：`data/M4_v0/freeze.json`；M1/M23输入冻结哈希复核一致。未经审查不进入M5。
''')
    p=Path('docs/BLOCKERS.md');p.write_text(p.read_text()+'\n## M4 v0 收尾\n必做实验完成，G1待审查；未见题族重叠、Q4缺失及有限训练预算的影响未被消除，详见 `reports/M4_REVIEW.md` 的(d)/(e)，不以工程完成替代研究结论。\n')
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,frozen_files=len(files),review_bytes=len(review.encode()),g1_approved=False)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

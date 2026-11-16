"""Produce M10e reports and immutable evidence manifest after final results."""
import argparse
import json
import math
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from scripts.m10d import digest
from scripts.m10e_document import main as document


def main(c):
    root=Path(c['output'])
    assert not (root/'freeze.json').exists(),'M10e already frozen; do not overwrite'
    assert json.loads((root/'upstream_after.json').read_text())['passed']
    assert json.loads((root/'aggregation_checks.json').read_text())['passed']
    document(c)
    summary=pd.read_csv(root/'summary.csv');cells=json.loads((root/'table_cells.json').read_text())['cells']
    lock=json.loads((root/'training_lock.json').read_text())
    costs=[json.loads(p.read_text()) for p in root.glob('cost_*.json')]
    total=sum(r['gpu_hours'] for r in costs)
    write_json(root/'cost_ledger.json',dict(gpu_hours=total,gpu='RTX4090',source_forward_calls=0,records=costs,
        convention='Conservative full run wall time for GPU runs, including loading/smoke/train/calibration/final inference; CLUE CPU-only wall separate; no failed GPU runs omitted.'))
    def value(source,method,q,metric,split='test',kind='clean'):
        r=summary[(summary.source==source)&(summary.model==method)&(summary.qtype==q)&(summary.metric==metric)&(summary.split==split)&(summary.kind==kind)]
        assert len(r)==1
        return float(r.iloc[0]['mean'])
    def sig3(x):
        return '0.00' if x==0 else f'{x:.{max(0,2-math.floor(math.log10(abs(x))))}f}'
    lines=['|源/方法|val Q1 clean/traj|test Q1 NLL|Q2 AUC|Q3/Q4 acc|','|---|---|---|---|---|']
    for source,method in [('gpt2','HSRM'),('llama','HSRM'),('gpt2','CLUE'),('llama','CLUE'),('llama','B2')]:
        v=[value(source,method,'Q1','nll','reader_val',k) for k in ['clean','traj']]
        q3q4='/'.join(sig3(value(source,method,q,'accuracy')) for q in ['Q3','Q4']) if method=='B2' else '—'
        lines.append(f"|{source}/{method}|{sig3(v[0])}/{sig3(v[1])}|{sig3(value(source,method,'Q1','nll'))}|{sig3(value(source,method,'Q2','auroc'))}|{q3q4}|")
    checked=json.loads((root/'upstream_after.json').read_text())['checked']
    review=f'''### (a) 状态
M10e三项实验完成待审查，再次冻结；论文修改按用户指令交接。
### (b) 关卡
通过：5项val smoke、5项单测、18权重+2质心测试前hash锁、10,000次按题bootstrap、{checked}项旧科学输入SHA一致。新增{sig3(total)}GPUh、源前向0。
未执行：论文改稿/编译/8页验收，已按用户修订移交。
### (c) 主表
{chr(10).join(lines)}
CI及reader_val Q2/Q3/Q4见summary.csv；CLUE确定性单次，其余三seed。
### (d) 新增偏离与阻塞
HSRM用BCE、CLUE用末层prefix delta，未冒称原文完整复现。CPU汇总曾因近0.5舍入差中止；v2按既有表1 float64导出并双精度重放通过，原预测/温度不改。
### (e) 待决定
无。
### (f) 证据
reports/M10e_PAPER_HANDOFF.md；reports/M10e_details.md；data/M10e_v1/{{summary.csv,training_lock.json,freeze.json}}；final_*.json索引真实runs及预测。
'''
    assert len(review.encode())<=2000,len(review.encode())
    Path('reports/M10e_REVIEW.md').write_text(review)
    details=['# M10e 详细证据','',Path('docs/M10e_TASK.md').read_text(),
        '\n## 运行协议与审计\n',
        '原文与SHA：data/M10e_v1/literature/sources.json。HSRM第3节及附录C、CLUE第3节及附录A已读；无源前向。',
        'HSRM两源原始末层观测prefix输入，2层256宽/4头/FF1024/dropout0.1；有效位置均值后LN标量。BCE监督Q1独立Bernoulli；Q2复用同一标量，Q1/Q2分别在cal_A温度缩放。未输入源问题、标签或split标识。',
        'CLUE每条reader_train完整clean/扰动轨迹终点只计一次；每类算术平均raw delta。评分=到失败质心欧氏距离−到成功质心距离；各t只用当前可观测delta=z_t−z_0。cal_A对Q1/Q2独立无正则logistic，数值标准化仅帮助优化，与原始距离差的仿射模型等价。单次确定性，不以重复运行伪造seed CI。',
        'B2延用M3输入/模板/数值选项/训练归一化/优化器/2LR/8epoch/每qtype选择；只扩2048维，Q3原始sqrt(768)缩放保留。旧GPT2 B2仅从冻结M3 statistics取clean/seen/三seed/各t等权，legacy_gpt2_B2.json保存来源hash。',
        'HSRM完整搜索2LR×8epoch×3seed，各epoch全量reader_val NLL选优；后期val NLL上升如实保存在history.json，不改变预设训练网格。全部训练结束后才统一training_lock，cal_A随后拟合并保存，test_access.json记录最终访问锁。',
        '训练启动快照由各run实际保存config/git/environment；data/M10e_v1/code_before为开发阶段保全，非每run源码快照。训练期间仅补充冻结与测试入口的配置/依赖hash完整性校验，未改训练计算；当前代码SHA在测试前锁定。',
        'smoke每项20训练/20val题；GPT2比例0.100%/1.00%，LLaMA 0.293%/2.00%，均≤5%且≤15min。具体时间/数据审计见smoke_*.json对应run和cost_*.json。',
        '统计：test/clean/seen，各t内micro、t及固定seed等权。95%CI为10,000次题目cluster bootstrap，题目重采样在t/seed间共享；条件于固定训练模型，不覆盖重新训练seed的不确定性。reader_val Q1 clean/traj及Q2–Q4分列summary.csv，所有逐t/seed指标在最终run statistics.json。',
        'CPU汇总v1 run 20261112T072248432462Z_189260e52f8b_40c2ab 因B2 reader_val Q4 online float32与导出float64 accuracy差2.38e-5触发审计中止。近0.5处float32 softmax造成额外argmax平局。原脚本、失败日志、预测、温度和在线statistics全部保留；m10e_summarize_v2分别重放原生精度并核对原statistics，再以既有scripts/m9_export.py的float64 serialized-logit口径独立核对最终汇总，没有放宽一致性阈值、重训、重校准或重推断。新增近零logit回归测试，合计5项通过。精度差逐项见aggregation_checks.json。',
        f'成本{total:.12g} GPUh，RTX4090，共享GPU。包含GPU任务完整wall（加载、smoke、训练、校准和最终预测），保守高于实际GPU kernel占用。CLUE CPU时间单列cost_ledger.json；源前向0。',
        f'旧输入前后{checked}项hash均一致；核验范围是相关冻结dump/QA和各freeze清单原字节，另核验复用的M3统计文件。不宣称消除历史冻结日志漂移。',
        '用户后续明确改为交付Markdown而非修改论文；故未改/编译manuscripts，未声称新版8页已验收。reports/M10e_PAPER_HANDOFF.md含完整三行LaTeX、两句setup、原值/CI/来源、文献及编译要求。',
        '\n## 主表及区间\n',
        '| '+' | '.join(summary.columns)+' |\n| '+' | '.join(['---']*len(summary.columns))+' |\n'+
        '\n'.join('| '+' | '.join(f'{v:.9g}' if isinstance(v,float) else str(v) for v in row)+' |' for row in summary.itertuples(index=False,name=None)),
        '\n## 训练/评估索引\n']
    for key,d in lock['models'].items():
        details.append(f"- {key}: training `{d['run']}`; final `{json.loads((root/f'final_{key}.json').read_text())['run']}`")
        if Path(d['run'],'selection.json').exists():details.append('```json\n'+Path(d['run'],'selection.json').read_text()+'\n```')
    Path('reports/M10e_details.md').write_text('\n\n'.join(details)+'\n')
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    status=f'M10e两源HSRM三seed/确定性CLUE及LLaMA B2 Q1–Q4三seed完成待审查，实验再次冻结；{total:.3g}GPUh、源前向0，旧科学输入SHA一致。论文按用户修订仅交付reports/M10e_PAPER_HANDOFF.md，未改稿/编译。唯一审查reports/M10e_REVIEW.md；冻结data/M10e_v1/freeze.json。'
    for name in ['AGENTS.md','docs/STATUS.md']:
        p=Path(name);s=p.read_text();s=re.sub(r'<!-- M10e_LIVE -->.*?<!-- /M10e_LIVE -->',f'<!-- M10e_LIVE -->\n{status}\n<!-- /M10e_LIVE -->',s,flags=re.S);p.write_text(s)
    records={
        'docs/EXPERIMENTS.md':f'\n## {now} — M10e\n两源HSRM/CLUE及LLaMA M3 B2完成，三seed或确定性按题bootstrap10000 CI见data/M10e_v1/summary.csv。\n命令：python -m scripts.m10e configs/M10e/run.yaml stage={{smoke,train,lock,final}}；后接m10e_summarize_v2/audit/finish。\n新增{total:.3g}GPUh、源前向0；HSRM后期val NLL上升及负面比较如实保留，不改test/旧冻结。\n用户改为Markdown论文交接，未改稿编译；审查reports/M10e_REVIEW.md，详情reports/M10e_details.md。\n',
        'docs/BLOCKERS.md':'\n## M10e — 方法适配与论文交接\nHSRM按任务用Bernoulli BCE替代原文组内排序；CLUE末层prefix delta替代原文全层显式思维边界，完整轨迹质心不随t重训。\n无模型/训练精度/批大小降级；CPU导出精度审计失败后按既有表1 float64口径双重重放修复，原日志/预测/温度保留。\n论文改稿/8页编译按用户后续指令交给另一个对话，不能标作已验收。证据reports/M10e_details.md、M10e_PAPER_HANDOFF.md。\n',
        'docs/CHANGELOG.md':f'\n## {now} — M10e\n只新增外部基线、LLaMA B2与证据；旧代码/数据/结果/阈值不改。完成后冻结data/M10e_v1；用户最新要求仅交付论文修改Markdown。\n'}
    for p,s in records.items():
        assert len(s.strip().splitlines())<=10
        with Path(p).open('a') as f:f.write(s)
    write_json('docs/M10e_LIVE.json',dict(status='completed_pending_review',completed_at=now,output=str(root),
        review='reports/M10e_REVIEW.md',handoff='reports/M10e_PAPER_HANDOFF.md',freeze=str(root/'freeze.json'),gpu_hours=total,source_forward_calls=0,
        manuscript_edited=False,compilation_performed=False,experiments_frozen=True))
    files=[]
    for p in root.rglob('*'):
        if p.is_file():files.append(p)
    runs=set()
    for p in list(root.glob('smoke_*.json'))+list(root.glob('trained_*.json'))+list(root.glob('final_*.json')):
        runs.add(json.loads(p.read_text())['run'])
    runs.add(lock['run']);runs.add(json.loads((root/'table_cells.json').read_text())['run'])
    if (root/'summary_recovery.json').exists():
        runs.add(json.loads((root/'summary_recovery.json').read_text())['failed_run'])
    for run in runs:files.extend(p for p in Path(run).rglob('*') if p.is_file())
    files.extend(Path(p) for p in ['docs/M10e_TASK.md','docs/M10e_LIVE.json','clr/baselines/m10e.py','configs/M10e/run.yaml','tests/test_m10e.py'])
    files.extend(Path('scripts').glob('m10e*.py'));files.extend(Path('reports').glob('M10e*'))
    hashes={str(p):digest(p) for p in sorted(set(files)) if p.is_file()}
    freeze=dict(milestone='M10e',version='v1',status='completed_pending_review',created_at=now,sha256=hashes,
        gpu_hours=total,source_forward_calls=0,scientific_results_frozen=True,manuscript_editing_delegated_by_user=True)
    write_json(root/'freeze.json',freeze)
    bad=[p for p,h in hashes.items() if digest(p)!=h];assert not bad,bad
    write_json(root/'verification.json',dict(passed=True,checked=len(hashes),mismatches=bad,freeze_sha256=digest(root/'freeze.json')))
    print(status,flush=True);print('review_bytes',len(review.encode()),'frozen_files',len(hashes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

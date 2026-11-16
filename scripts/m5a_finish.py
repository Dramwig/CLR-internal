"""Persist completed M5a state and freeze its reproducibility evidence."""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(c):
    root=Path(c['search_run']);out=Path(c['output']);analysis=read(out/'analysis/analysis.json')
    metrics=read(root/'metrics.json')
    if metrics['status']!='completed' or not analysis.get('combined_gate_passed') or analysis['smoke']:
        raise RuntimeError('Full passing search and analysis required')
    if analysis['questions']!=300 or analysis['seeds']!=[11,22,33] or analysis['cost_gpu_hours_conservative']>10:
        raise RuntimeError('Full protocol/budget incomplete')
    review=Path(c['review'])
    if len(review.read_bytes())>8000:
        raise RuntimeError('Review too long')
    if (out/'freeze.json').exists():
        raise RuntimeError('Completed evidence already frozen; use a new version')
    report_run=Path(analysis['report_run'])
    extension=Path(analysis['vote_extension_run']) if analysis.get('vote_extension_run') else None
    now=datetime.now(timezone.utc).isoformat()
    primary=next(r for r in analysis['summary'] if (r['method'],r['value'],r['N'],r['b'],r['horizon'])==('S3','oracle',16,4,6))
    def pp(x):return f'{100*x:.3g}'
    result=f"oracle同预算S1增益{pp(primary['delta'])}pp，95%CI[{pp(primary['delta_low'])},{pp(primary['delta_high'])}]"
    status=f"M5a全量val300/三seed及配对统计完成，待审查；{result}；B4/B3未见正收益。保守累计{analysis['cost_gpu_hours_conservative']:.3g}GPUh≤10，不触碰test。"
    live=read('docs/M5a_LIVE.json')
    live.update(updated_utc=now,stage='completed_pending_review',milestone_complete=True,
                active_run=str(root),report_run=str(report_run),review=c['review'],detail=c['detail'],
                freeze=str(out/'freeze.json'),analysis=str(out/'analysis'),decision_needed=None,
                completed_questions=300,seeds=[11,22,33],
                combined_gate_passed=True,vote_extension_run=str(extension) if extension else None,
                conservative_total_gpu_hours=analysis['cost_gpu_hours_conservative'],test_access=False)
    write_json(Path('docs/M5a_LIVE.json'),live)
    p=Path('AGENTS.md');s=p.read_text();s=re.sub(r'<!-- M5A_LIVE -->.*?<!-- /M5A_LIVE -->',
        '<!-- M5A_LIVE -->\n'+status+'仅S3 oracle限N16/b4全步（用户继续授权），其余网格完整。唯一审查reports/M5a_REVIEW.md；冻结data/M5a_v13/freeze.json，接续docs/M5a_LIVE.json；不代表G1通过。\n<!-- /M5A_LIVE -->',s,flags=re.S);p.write_text(s)
    p=Path('docs/STATUS.md');s=p.read_text();a=s.index('## 2026-11-05 当前 M5a 接续');b=s.index('\n## 里程碑',a)
    s=s[:a]+'## 2026-11-05 当前 M5a 接续\n\n'+status+'\n用户“允许 继续完成”后按10GPUh方案执行：仅S3 oracle限N16/b4全步，其余S1/S2/B4/B3原网格不变。两项研究问题的估计/限制见唯一审查reports/M5a_REVIEW.md。详细reports/M5a_search.md；逐题CSV/图/成本data/M5a_v13/analysis；冻结data/M5a_v13/freeze.json；接续docs/M5a_LIVE.json。专项授权不代表G1通过，完整M5/M6未启动。\n'+s[b:];p.write_text(s)
    entries={
      'docs/EXPERIMENTS.md':f'''M5a val300×seed11/22/33完成；{result}（题内平均seed后按题bootstrap10,000次）。
命令：scripts/m5a_search.py configs/M5a_full_v13.yaml；scripts/m5a_report.py configs/M5a_report_v13.yaml。
全部B4/B3/S1/S2网格，S3 oracle按继续授权限N16/b4全步；负面结果与比例不稳定情况均保留。
累计保守{analysis['cost_gpu_hours_conservative']:.6g}GPUh含smoke/失败/中断预留；无test访问。
唯一审查reports/M5a_REVIEW.md；完整表/CI/曲线/成本reports/M5a_search.md及data/M5a_v13/analysis/；冻结data/M5a_v13/freeze.json。''',
      'docs/BLOCKERS.md':'''M5a继续授权后仅S3 oracle限N16/b4全步，其他网格/题/seed/K不缩小；10GPUh内完成。
统计入口首次误用当前任务文档hash核对旧smoke，已改核对其启动时源码快照；保留失败日志，无新增GPU成本。
离散投票存在不足一条轨迹余额，允许预算曲线可欠花且不外推缺失点；oracle MC/终点gold特权不构成严格数学上界。
比例使用固定N16/b4三者实际费用最大值的共同预算；全计费时probe未花满，分母CI含零不解释比例。
本次新增及预检限制见reports/M5a_search.md；旧预检失败/成本缺口原样保留，未作为零成本。''',
      'docs/CHANGELOG.md':'''完成M5a授权v13搜索、全部配置CSV、同预算配对、按题bootstrap、actual/allowed compute曲线与唯一审查。
新增统计/收尾/接续代码；M5a输入与旧预检快照保留，M1/M23/M4冻结结果不改；未触碰test。'''}
    for path,body in entries.items():
        with Path(path).open('a') as f:f.write('\n## '+now+' — M5a v13完成待审查\n'+body+'\n')
    # Snapshot mutable documentation so later milestones can update live indices.
    snapshot=out/'completed_snapshot'
    names=['AGENTS.md','docs/M5a_TASK.md','docs/M5a_LIVE.json','docs/STATUS.md',
           'docs/EXPERIMENTS.md','docs/BLOCKERS.md','docs/CHANGELOG.md',
           'clr/search/m5a.py','clr/search/analysis_m5a.py','scripts/m5a_search.py',
           'scripts/m5a_report.py','scripts/m5a_finish.py','scripts/m5a_pipeline.py','tests/test_m5a.py',
           'scripts/m5a_extend_votes.py','configs/M5a_extend_votes_v13.yaml',
           'configs/M5a_full_v13.yaml','configs/M5a_smoke_v13.yaml',
           'configs/M5a_report_v13.yaml','configs/M5a_report_smoke_v13.yaml',
           'configs/M5a_pipeline_v13.yaml',c['config_path']]
    for name in names:
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    for name in [c['review'],c['detail'],'reports/M5a_full_v13.log','reports/M5a_smoke_v13.log',
                 'reports/M5a_tests_analysis_v13.log','reports/M5a_report_v13.log',
                 'reports/M5a_extend_votes_v13.log','reports/M5a_pipeline_v13.log']:
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
    paths=[p for p in out.rglob('*') if p.is_file()]
    paths += [p for directory in [root,report_run]+([extension] if extension else []) for p in directory.rglob('*') if p.is_file()]
    # Also reference the already frozen preflight without rewriting any file.
    paths.append(Path('data/M5a_preflight_v0/preflight_evidence_freeze.json'))
    hashes={str(p):digest(p) for p in sorted(set(paths))}
    freeze={'version':'M5a_v13','created_utc':now,'status':'completed_pending_review',
            'search_run':str(root),'report_run':str(report_run),'test_access':False,
            'sha256':hashes,'cost_gpu_hours_conservative':analysis['cost_gpu_hours_conservative']}
    write_json(out/'freeze.json',freeze)
    mismatches=[p for p,h in hashes.items() if digest(Path(p))!=h]
    write_json(out/'verification.json',{'verified_utc':now,'files':len(hashes),'mismatches':mismatches})
    if mismatches:raise RuntimeError('Freeze mismatch')
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*')
    a=p.parse_args();c=load_config(a.config,a.overrides);c['config_path']=a.config;main(c)

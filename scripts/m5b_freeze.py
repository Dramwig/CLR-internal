"""Final M5b completion audit, persistent status and immutable SHA manifest."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(c):
    root=Path(c['output'])
    if (root/'freeze.json').exists():raise ValueError('M5b already frozen; create a new version for changes')
    report=read(root/'report_complete.json');cost=read(root/'analysis/costs.json')
    if cost['gpu_hours']>25 or cost['missing']:raise ValueError('Budget ledger incomplete')
    review=Path('reports/M5b_REVIEW.md').read_text()
    headings=[line for line in review.splitlines() if line.startswith('## (')]
    tables=sum(line.startswith('|---') for line in review.splitlines())
    if len(review.encode())>8000 or headings!=[next((line for line in headings if line.startswith(f'## ({x})')),None) for x in 'abcdef'] or tables>3:raise ValueError('Review format invalid')
    if report['review_bytes']!=len(review.encode()):raise ValueError('Report completion pointer is stale')
    complete=read(root/'completion.json');lock=read(root/'A_training_lock.json')
    bad_checkpoints=[p for p,h in lock['checkpoint_sha256'].items() if not Path(p).exists() or sha(p)!=h]
    if bad_checkpoints or not lock['all_training_locked']:raise ValueError('A training lock changed')
    for p in complete['A']['runs'].values():
        m=read(Path(p)/'metrics.json')
        if m['status']!='completed' or not m['test_opened']:raise ValueError('Missing authorized final A evaluation: '+p)
    isolated=[complete['B_run'],*complete['C_runs'],complete['D_generation'],*complete['D']['training_runs'].values(),*complete['D']['C_runs'],read(root/'D_B_rescore.json')['run']]
    for p in isolated:
        m=read(Path(p)/'metrics.json')
        if m['status']!='completed' or m.get('test_opened') is not False:raise ValueError('B/C/D isolation/completion failed: '+p)
    with (root/'analysis/C_all_settings.csv').open() as f:
        panel=Counter(tuple(r[k] for k in ['method','value','N','b','horizon']) for r in csv.DictReader(f))
    bc=read(root/'analysis/BC_analysis.json')
    if len(panel)!=120 or set(panel.values())!={900} or bc['B_rows']!=91800 or bc['C_rows']!=108000:raise ValueError('Incomplete B/C grid')
    write_json(root/'final_audit.json',{'review_bytes':len(review.encode()),'review_sha256':sha('reports/M5b_REVIEW.md'),'core_tables':tables,'A_checkpoint_hashes':len(lock['checkpoint_sha256']),'A_checkpoint_mismatches':bad_checkpoints,'A_final_evaluations':len(complete['A']['runs']),'B_C_D_completed_test_false_runs':isolated,'B_rows':bc['B_rows'],'C_rows':bc['C_rows'],'C_settings':len(panel),'rows_per_setting':900,'test_isolation_milestone_gate':'failed: invalid early A development exposure retained/excluded','gpu_hours':cost['gpu_hours']})
    verify={}
    for name in ['data/M41_v1/freeze.json','data/M5a_v13/freeze.json']:
        frozen=read(name);bad=[p for p,h in frozen['sha256'].items() if not Path(p).exists() or sha(p)!=h]
        verify[name]={'files':len(frozen['sha256']),'mismatches':bad}
        if bad:raise ValueError('Frozen upstream evidence changed: '+str(bad))
    write_json(root/'frozen_inputs_verification.json',verify)
    complete.update(milestone_complete=True,stage='completed_pending_review',review='reports/M5b_REVIEW.md',report=report)
    write_json(root/'completion.json',complete)
    stamp=datetime.now(timezone.utc).isoformat();h=cost['gpu_hours']
    brief=f'M5b A–D/三seed、配对统计完成，待审查；累计{h:.3g}/25 GPUh。唯一审查reports/M5b_REVIEW.md；冻结data/M5b_v1/freeze.json，接续docs/M5b_LIVE.json。搜索结论见审查报告；A历史无效开发run提前test暴露保留并排除，B/C/D未触碰test。用户G1按Q3/Q4有条件通过；未启动M6。'
    ag=Path('AGENTS.md');text=ag.read_text();begin=text.index('<!-- M5B_LIVE -->');end=text.index('<!-- /M5B_LIVE -->')+len('<!-- /M5B_LIVE -->');ag.write_text(text[:begin]+'<!-- M5B_LIVE -->\n'+brief+'\n<!-- /M5B_LIVE -->'+text[end:])
    with Path('docs/STATUS.md').open('a') as f:f.write('\n## '+stamp+' — M5b完成待审查\n'+brief+'\n详细reports/M5b_details.md，全部CSV/曲线/成本data/M5b_v1/analysis；run索引data/M5b_v1/completion.json。M41/M5a共4154个冻结文件SHA一致。\n')
    with Path('docs/EXPERIMENTS.md').open('a') as f:f.write('\n## '+stamp+' — M5b完成待审查\nA重训前缀/单状态消融、B同父MC诊断、C含clean搜索v2与D排序微调，三seed及10k题bootstrap完成。\n命令：scripts/m5b_finish.py configs/M5b_finish.yaml；scripts/m5b_report.py configs/M5b_report.yaml。\n三seed均值/95%CI、全部负面结果与偏离见[唯一审查](../reports/M5b_REVIEW.md)及[详细报告](../reports/M5b_details.md)。\n累计'+f'{h:.6g}'+' GPUh≤25，含失败/smoke/无效run；B/C/D无test，A早期无效test暴露保留并排除。\n冻结data/M5b_v1/freeze.json；run/CSV/曲线/成本索引data/M5b_v1/completion.json与analysis/index.json。\n')
    with Path('docs/CHANGELOG.md').open('a') as f:f.write('\n## '+stamp+' — M5b最终统计与冻结\n新增逐qtype同覆盖率配对CI、兄弟诊断、全部搜索网格及同时CI、D配对比较、成本总账与唯一审查。\n正式A模型在test评估前锁定；初始无效开发test暴露未抹去。M4.1/M5a冻结文件哈希全部一致。\n')
    with Path('docs/BLOCKERS.md').open('a') as f:f.write('\n## '+stamp+' — M5b收尾边界\n实验/统计完成；搜索收益关卡未通过，组内排序改善未转化为显著搜索收益。\n同覆盖率集合比较是使用评估标签的描述性诊断；计算匹配是平均前向预算上限，非严格等FLOPs/延迟。\nA早期无效test暴露仍构成隔离偏离；其处置与全部局限见[唯一审查](../reports/M5b_REVIEW.md)，不重复历史失败。\n')
    live=read('docs/M5b_LIVE.json');live.update(stage='completed_pending_review',milestone_complete=True,completed_utc=stamp,cost_gpu_hours=h,review='reports/M5b_REVIEW.md',freeze=str(root/'freeze.json'),completion=str(root/'completion.json'),test_scope='A only; historical invalid early exposure retained/excluded; B/C/D test=false')
    write_json('docs/M5b_LIVE.json',live)
    state=read(root/'finish_pipeline.json');state.update(stage='completed_pending_review');write_json(root/'finish_pipeline.json',state)
    docs=root/'document_snapshot';docs.mkdir(exist_ok=True)
    for p in ['AGENTS.md','docs/M5b_TASK.md','docs/M5b_LIVE.json','docs/STATUS.md','docs/EXPERIMENTS.md','docs/BLOCKERS.md','docs/CHANGELOG.md']:
        target=docs/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    runs=[]
    for p in Path('runs').glob('*/config.json'):
        cfg=read(p)
        if cfg.get('milestone')=='M5b':runs.append(p.parent)
    files=set()
    for pattern in ['clr/models/m5b*.py','clr/search/m5b*.py','clr/utils/m5b*.py','scripts/m5b*.py','configs/M5b*.yaml','tests/test_m5b.py','reports/M5b*']:
        for p in Path('.').glob(pattern):
            # A redirected freezer stdout can append its completion receipt after
            # this manifest is written. Keep that receipt outside the payload.
            if p.is_file() and not (p.parent==Path('reports') and p.name.startswith('M5b_freeze') and p.suffix=='.log'):files.add(p)
    for parent in [root,*runs]:
        files.update(p for p in parent.rglob('*') if p.is_file() and p.name!='freeze.json' and '__pycache__' not in p.parts)
    write_json(root/'freeze.json',{'status':'completed_pending_review','version':'M5b_v1','created_utc':stamp,'gpu_hours':h,'review':'reports/M5b_REVIEW.md',
                                 'test_scope':'A final pre-specified ablation plus retained invalid early A exposure; B/C/D val/reader_train only',
                                 'initial_freeze_attempt_audit':str(root/'freeze_initial_attempt_audit.json') if (root/'freeze_initial_attempt_audit.json').exists() else None,
                                 'upstream_verification':verify,'sha256':{str(p):sha(p) for p in sorted(files)}})
    print(json.dumps({'files':len(files),'review_bytes':len(review.encode()),'gpu_hours':h,'status':'completed_pending_review'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

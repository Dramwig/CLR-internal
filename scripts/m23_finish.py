"""Wait for explicitly configured M23 runs, manage disk reserve, then report/freeze."""
import argparse,json,subprocess,sys,time,hashlib,re
from pathlib import Path
from clr.utils.config import load_config
from clr.data.resume import ensure_space
from clr.utils.idea_a import write_json

def capsule(text):
    for file in ['AGENTS.md','docs/STATUS.md']:
        p=Path(file);s=p.read_text();block='<!-- M23_LIVE -->\n'+text+'\n<!-- /M23_LIVE -->'
        if '<!-- M23_LIVE -->' in s:s=re.sub(r'<!-- M23_LIVE -->.*?<!-- /M23_LIVE -->',lambda m:block,s,flags=re.S)
        else:s+='\n\n'+block+'\n'
        p.write_text(s)

def main(c,config):
    output=Path(c['output']);statuspath=output/'pipeline_status.json';previous=None
    keys=['question_run','baseline_run','patchscope_run','layers_run','quant_run','official_run','svamp_run','asdiv_run','audit_run']
    while True:
        ensure_space(output)
        status={k:json.loads(Path(c[k],'metrics.json').read_text()).get('status') for k in keys}
        if status!=previous:
            capsule('M2–M3当前运行：'+json.dumps(status)+'；索引 '+str(statuspath)+'。未完成项不得宣称完成。')
            write_json(statuspath,{'status':'running','runs':{k:c[k] for k in keys},'run_status':status})
            with Path('docs/EXPERIMENTS.md').open('a') as f:f.write('\nM23自动状态更新：`'+json.dumps(status)+'`。运行索引 `'+str(statuspath)+'`。\n')
            with Path('docs/STATUS.md').open('a') as f:f.write('\nM23运行状态：`'+json.dumps(status)+'`；详见 `'+str(statuspath)+'`。\n')
            previous=status
        failed=[k for k,v in status.items() if v not in ['running','completed']]
        if failed:
            write_json(statuspath,{'status':'blocked_failed_run','failed':failed,'runs':{k:c[k] for k in keys}})
            with Path('docs/BLOCKERS.md').open('a') as f:f.write('\nM23必需run未通过：'+str(failed)+'；未生成完成报告，未自动更改模型/设置。\n')
            return 1
        if all(v=='completed' for v in status.values()):break
        time.sleep(30)
    result=subprocess.run([sys.executable,'scripts/m23_report.py',config],check=True,capture_output=True,text=True)
    print(result.stdout,flush=True)
    if result.stderr:print(result.stderr,file=sys.stderr,flush=True)
    report_run=Path(result.stdout.strip().splitlines()[-1])
    assert report_run.is_dir()
    files=[Path('reports')/f for f in ['M2_questions.md','M3_baselines.md','M3_shift.md','M3_unseen_pools.md','M3_metrics.csv']]
    files+=list(Path('clr/data/questions').glob('*.py'))+list(Path('clr/baselines').glob('*.py'))+[Path('clr/calib/m23.py'),Path(c['cal_split']),Path(config)]
    files+=list(Path('scripts').glob('m23_*.py'))+list(Path('configs').glob('M23*.yaml'))+list(Path('tests').glob('test_m23*.py'))
    files+=[p for p in Path('reports').glob('M23*.log') if p.name!='M23_finish.log']
    files+=list(Path('reports/M3_figures').glob('*.png'))+[Path('reports/M23_tests.log'),Path(c['templates'])]
    files+=list((output/'full').glob('*'))+[output/'qa_reproduction.json']
    for key in keys:
        files+=list(Path(c[key]).glob('*.json'))+list(Path(c[key]).glob('*.parquet'))+list(Path(c[key]).glob('*.pt'))+list(Path(c[key]).glob('*.npz'))
    for key in keys:files+=list(Path(c[key]).rglob('*'))
    files+=list(report_run.rglob('*'))
    for extra in c.get('supplementary_runs',[]):files+=list(Path(extra).rglob('*'))
    files+=list((output/'unseen_v2').rglob('*'))+list((output/'asdiv').rglob('*'))
    files=[p for p in files if p.is_file()]
    report=Path('reports/M3_baselines.md').read_text()
    summary=report.split('## 已见模板、干净状态主表',1)[1].split('## 保留模板',1)[0]
    cost=report.split('## 运行与成本',1)[1].split('## Mondrian conformal',1)[0]
    with Path('docs/EXPERIMENTS.md').open('a') as f:
        f.write('\n\n## M2–M3最终完整运行（2026-11-03）\n\n完成类型化问题库、B0–B5a、Mondrian、域/层诊断、未见池和M1量化补查。未训练CLR。完整命令以各run配置快照为准；报告入口 `.venv/bin/python scripts/m23_report.py configs/M23_report.yaml`。\n\n三seed主表（均值±95% CI，B5a另按题bootstrap）：\n'+summary+'\n成本与偏离：\n'+cost+'\n负面结果与三项审查问题见 reports/M3_baselines.md 末尾；不能将这些基线结果作为CLR有效性结论。\n')
    write_json(output/'freeze.json',{'status':'completed; awaiting user review','report_run':str(report_run),'runs':{k:c[k] for k in keys},'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    write_json(statuspath,{'status':'completed; awaiting user review','report_run':str(report_run),'runs':{k:c[k] for k in keys},'freeze':str(output/'freeze.json')})
    capsule('M2–M3已完成，待用户审查；未训练CLR。冻结索引 '+str(output/'freeze.json')+'。')
    p=Path('AGENTS.md');s=p.read_text().replace('M2–M3合并里程碑进行中，不训练CLR','M2–M3合并里程碑已完成，待用户审查；未训练CLR').replace('完整三seed基线和B5a运行中。','完整三seed基线和B5a已完成，结果待审查。');p.write_text(s)
    p=Path('docs/STATUS.md');s=p.read_text().replace('|M2–M3|进行中；问题库与审计完成，三seed基线和B5a正在运行；见下方M23_LIVE|','|M2–M3|全部必需实验与报告完成，待用户审查；冻结索引data/M23_v1a/freeze.json|');p.write_text(s)
    note='\n\nM2–M3问题库、B0–B5a、层/域诊断、官方500题、两候选未见池与量化抽查均完成，待用户审查。B5b推迟M6，未训练CLR。报告：reports/M2_questions.md、M3_baselines.md、M3_shift.md、M3_unseen_pools.md；冻结索引 '+str(output/'freeze.json')+'。\n'
    for file in ['AGENTS.md','docs/STATUS.md','docs/EXPERIMENTS.md','docs/BLOCKERS.md','docs/CHANGELOG.md']:
        with Path(file).open('a') as f:f.write(note)
    return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();sys.exit(main(load_config(a.config,a.overrides),a.config))

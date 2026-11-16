"""Final experiment freeze after final reports, paper rebuild and integrity audit."""
import argparse,hashlib,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import ledger,live

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main(c):
 root=Path(c['output'])
 if (root/'freeze.json').exists():raise FileExistsError('M10 already frozen')
 for name in ['exports_completed.json','reproducibility.json','upstream_audit.json']:
  if not (root/name).exists():raise ValueError('Completion gate missing '+name)
 for name in ['reproducibility.json','upstream_audit.json']:
  if not json.loads((root/name).read_text())['passed']:raise ValueError('Gate failed '+name)
 costs=ledger();hours={s:sum(r['gpu_hours'] for r in costs if r['section']==s) for s in c['limits']}
 if sum(hours.values())>40 or any(hours[s]>limit for s,limit in c['limits'].items()):raise ValueError('M10 budget exceeded')
 incomplete=[r for r in costs if r['status'] not in ['completed','failed']]
 if incomplete:raise ValueError('Nonterminal GPU run records: '+str(incomplete))
 review=Path('reports/M10_REVIEW.md').read_text()
 if len(review.encode())>8000 or not all('### ('+x+')' in review for x in 'abcdef'):raise ValueError('Review format gate failed')
 if len(Path('reports/FINAL_SUMMARY.md').read_bytes())>10000:raise ValueError('FINAL_SUMMARY >10KB')
 status='M10投稿补实验及论文产物完成待审查，实验最终冻结；B5c全量因预算NR，HiddenBench仅探索性。唯一审查reports/M10_REVIEW.md；冻结data/M10_v1/freeze.json。'
 for p in [Path('AGENTS.md'),Path('docs/STATUS.md')]:
  s=p.read_text();a=s.index('<!-- M10_LIVE -->');b=s.index('<!-- /M10_LIVE -->')+len('<!-- /M10_LIVE -->');p.write_text(s[:a]+'<!-- M10_LIVE -->\n'+status+'\n<!-- /M10_LIVE -->'+s[b:])
 with Path('docs/EXPERIMENTS.md').open('a') as f:f.write('\n## 2026-11-10 M10最终冻结\nLLaMA缺失格、B6/每类型CLR、HiddenBench和M9主体初始化三seed完成；B5c预算准入失败NR。\n累计'+str(round(sum(hours.values()),3))+'/40GPUh，含smoke/失败/校准/推断；全训练hash锁后才test。\n命令configs/M10/final_controller.yaml；论文独立重建、最终证据及CI见reports/M10_REVIEW.md和M10_details.md。\nM9原paper66文件归档，旧freeze不改；更新paper与FINAL_SUMMARY并最终冻结data/M10_v1/freeze.json。\n')
 with Path('docs/CHANGELOG.md').open('a') as f:f.write('\n## 2026-11-10 M10最终产物\n新增P1投稿控制、HiddenBench/跨用途证据和论文补表/曲线；上游冻结SHA核验，授权paper supersession逐字节可恢复。\n实验最终冻结；B5c缺失及HiddenBench小样本不能升格为正结论。\n')
 paths=set(root.rglob('*'))|set(Path('configs/M10').rglob('*'))|set(Path('scripts').glob('m10*.py'))|set(Path('paper').rglob('*'))|{Path('tests/test_m10.py'),Path('reports/M10_REVIEW.md'),Path('reports/M10_details.md'),Path('reports/FINAL_SUMMARY.md')}
 paths.update(Path(p) for p in ['clr/models/m10.py','clr/models/m10_b5c.py','clr/data/m10.py','clr/utils/m10_runtime.py','clr/audit/m10_transfer.py','clr/audit/m10_hiddenbench.py','clr/paper/render_m10.py']);paths.update(Path('scripts/paper').glob('make_*m10*.py'));paths.update(Path('scripts/paper')/p for p in ['make_t4_hiddenbench.py','make_appendix_per_type.py','make_f4_cross_use_transfer.py'])
 paths.update(Path('reports/M10_evidence').rglob('*'));paths.add(Path('docs/M10_TASK.md'))
 for r in costs:paths.update(Path(r['run']).rglob('*'))
 paths={p for p in paths if p.is_file() and p.suffix not in ['.pyc'] and 'runtime_locks' not in str(p) and p.name not in ['status_observation.json']}
 hashes={str(p):sha(p) for p in sorted(paths)};write_json(root/'freeze.json',{'version':'M10_v1','status':status,'GPUh_by_section':hours,'gpu_hours':sum(hours.values()),'source_forward_calls':sum(r['source_forward_calls'] or 0 for r in costs),'sha256':hashes,'upstream_audit':'data/M10_v1/upstream_audit.json','paper_version_relationship':'M9 original bytes archived; current paper explicitly superseded under M10 authorization'})
 live(stage='complete',milestone_complete=True,status=status,test_opened=True);print(status)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

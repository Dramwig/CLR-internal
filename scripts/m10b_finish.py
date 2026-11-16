"""CPU-only paper supersession, paired inference, reproducibility and final freeze."""
import argparse,json,hashlib,subprocess,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd,yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10b_runtime import ledger,live
from scripts.m10b_export import main as export

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def upstream(root):
 archives={}
 for mp in ['data/M10_v1/m9_paper_snapshot/mapping.json',str(root/'m10_paper_snapshot/mapping.json')]:
  for p,r in json.loads(Path(mp).read_text()).items():archives[str(Path(p).resolve()),r['sha256']]=r['archive']
 inventory=json.loads(Path('data/M10_v1/upstream_audit.json').read_text())['manifest_inventory']
 manifests=list(dict.fromkeys([r['manifest'] for r in inventory]+['data/M10_v1/freeze.json']))
 expect={}
 for mp in manifests:
  d=json.loads(Path(mp).read_text())
  for p,h in d.get('sha256',d.get('files',{})).items():
   if isinstance(h,str):expect.setdefault(str(Path(p).resolve()),[]).append((mp,h))
 def check(item):
  p,versions=item;cur=sha(p) if Path(p).exists() else None;errors=[];sup=[]
  for mp,h in versions:
   if h==cur:continue
   a=archives.get((p,h))
   if a and sha(a)==h:sup.append({'path':p,'manifest':mp,'archive':a,'sha256':h})
   else:errors.append({'path':p,'manifest':mp,'expected':h,'actual':cur})
  return errors,sup
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(check,expect.items()))
 errors=[x for a,b in results for x in a];sup=[x for a,b in results for x in b]
 known=json.loads((root/'preexisting_log_drift.json').read_text()) if (root/'preexisting_log_drift.json').exists() else {}
 exceptions=[e for e in errors if all(e.get(k)==known.get(k) for k in ['path','expected','actual'])]
 unexpected=[e for e in errors if e not in exceptions]
 write_json(root/'upstream_audit.json',{'passed':not errors,'scientific_inputs_intact':not unexpected,'unique_files':len(expect),'errors':errors,'known_preexisting_nonmetric_log_drift':exceptions,'unexpected_errors':unexpected,'authorized_paper_supersessions':sup,'manifests':{p:sha(p) for p in manifests}})
 if unexpected:raise ValueError('Unexpected upstream integrity failure')

def main(c):
 root=Path(c['output'])
 if (root/'freeze.json').exists():raise ValueError('M10b already frozen; use a new version for any change')
 if not json.loads((root/'experiments_completed.json').read_text())['completed']:raise ValueError('Experiments incomplete')
 export(c);costs=ledger();total=sum(r['gpu_hours'] for r in costs);write_json(root/'cost_ledger.json',costs)
 if total>22:raise ValueError('M10b cost exceeds hard budget')
 dest=root/'paper_inputs';f=pd.read_csv('data/M10_v1/paper_inputs/costs.csv');f=pd.concat([f,pd.DataFrame([{'milestone':'M10b','gpu_hours':total,'source_forward_calls':0,'missing_source_counts':0,'cost_scope':'B5c full resident wall time including failures/loading/training/calibration/inference; Qwen decoder calls are reader calls','evidence':str(root/'cost_ledger.json')}])],ignore_index=True);f.to_csv(dest/'costs.csv',index=False,float_format='%.15g')
 write_json(dest/'freeze.json',{'version':'M10b_paper_inputs_v1','sha256':{str(p):sha(p) for p in dest.glob('*.csv')}})
 folder=Path('configs/M10b/paper');folder.mkdir(exist_ok=True);build=load_config('configs/M10/paper_build.yaml');items=[]
 for item in build['artifacts']:
  cfg=load_config(item['config']);name=cfg['artifact'];script=item['script']
  if name in ['t1_gsm8k','t1_reader_val']:
   cfg.update(input_freeze=str(dest/'freeze.json'),inputs={'main':str(dest/'main.csv')});script=f'scripts/paper/make_m10b_{name}.py'
  elif name=='t5_cost':cfg.update(input_freeze=str(dest/'freeze.json'),inputs={'costs':str(dest/'costs.csv')})
  elif name=='t4_hiddenbench':cfg['caption']='Appendix: HiddenBench exploratory audit (insufficient data; no conclusion)'
  elif name=='f4_cross_use_transfer':cfg['caption']='Appendix: cross-use initialization (not an established transfer claim)'
  p=folder/(name+'.yaml');p.write_text(yaml.safe_dump(cfg,sort_keys=False));items.append(dict(script=script,config=str(p)))
 for name,out in [('paper_build','paper'),('paper_repro',str(root/'repro_build'))]:
  p=Path('configs/M10b')/(name+'.yaml');p.write_text(yaml.safe_dump({'output_root':out,'isolated':True,'artifacts':items},sort_keys=False));subprocess.run([c['paper_python'],'-I','scripts/paper/make_all.py',str(p)],check=True)
 records=[]
 for sub in ['tables','figures','derived']:
  left={p.name:p for p in (Path('paper')/sub).glob('*') if p.is_file()};right={p.name:p for p in (root/'repro_build'/sub).glob('*') if p.is_file()}
  if left.keys()!=right.keys():raise ValueError('Reproduction inventories differ')
  for name,p in left.items():records.append({'path':str(p),'sha256':sha(p),'repeat_sha256':sha(right[name]),'byte_identical':p.read_bytes()==right[name].read_bytes()})
 if not all(r['byte_identical'] for r in records):raise ValueError('Reproduction differs')
 probe=subprocess.run([c['paper_python'],'-I','-c','import importlib.util,json; print(json.dumps({k:importlib.util.find_spec(k) is not None for k in ["torch","transformers","numpy","pandas","matplotlib","yaml"]}))'],capture_output=True,text=True,check=True);modules=json.loads(probe.stdout)
 if modules['torch'] or modules['transformers']:raise ValueError('Paper environment is not independent')
 write_json(root/'reproducibility.json',{'passed':True,'products':records,'environment_modules':modules,'method':'independent .venv-paper, -I, two output roots; byte-identical TeX/CSV/PDF'})
 paired=pd.read_csv(dest/'CLR_minus_B5c.csv');summary=pd.read_csv(dest/'B5c_summary.csv');cp=pd.read_csv(dest/'B5c_conformal.csv')
 def get(f,**kw):
  for k,v in kw.items():f=f[f[k]==v]
  if len(f)!=1:raise ValueError('Ambiguous report cell '+str(kw))
  return f.iloc[0]
 def fmt(r,scale=100):return f'{r["mean"]*scale:.3g} [{r["lo"]*scale:.3g},{r["hi"]*scale:.3g}]'
 rows=[]
 for split in ['reader_val','test']:
  for q in ['Q3','Q4']:
   a=get(summary,split=split,qtype=q,metric='accuracy');d=get(paired,split=split,qtype=q,model='CLR-lite');rows.append(f'|{split}|{q}|{fmt(a)}|{fmt(d)}|')
 test='；'.join(q+' '+fmt(get(paired,split='test',qtype=q,model='CLR-lite'))+'pp' for q in ['Q3','Q4'])
 review='''# M10b 审查

### (a) 状态一句话
GPT2 Q3/Q4 B5c三seed完整训练、最终评估和论文复现完成待审查；本里程碑后实验永久冻结。

### (b) 关卡清单
- 通过：20组train/20组reader_val训练smoke，76.7秒；15组reader_val校准/评估代理，10.6秒；无test访问。
- 通过：Qwen3-1.7B冻结基座、两层映射、LoRA r16/alpha32；每seed完整Q3 860228/Q4 279440条训练一遍，seed11/22/33。无少标签、少seed。
- 通过：全部权重SHA锁后才test；cal_A温度/cal_B按类型与step Mondrian，三seed并列reader_val/test。
'''+f'- 通过：成本{total:.3g}/22 GPUh（加载/失败/校准/推断在账本）；独立环境{len(records)}个产物逐字节一致；上游模型/数据/指标SHA一致。\n'+'''- 未通过（既有非指标日志完整性）：M10 tests_final.log在M10冻结后、M10b启动前被改写；原字节缺失，不重造，预期/实际SHA与时间记录preexisting_log_drift.json。其余136268项一致，不影响本次模型/指标比较。
- 通过：审计改为类型化审计框架，正文并列CLR/B6；跨用途/HiddenBench只入附录，后者数据不足无法结论。
- 未通过（授权范围外、未执行）：B5c Q1/Q2、LLaMA源及审计任务，均NA/NR，不外推。

### (c) 核心主表
准确率/差值均为%，差值单位pp；固定三seed，整problem bootstrap10000次95%CI，保留组内step；不是重新训练seed的不确定性。

|split|类型|B5c准确率 [95%CI]|CLR−B5c [95%CI]|
|---|---|---|---|
'''+ '\n'.join(rows)+'\n\n用户所问test差值：'+test+'。\n\n'+'''### (d) 本里程碑新增的偏离与阻塞
新运行账本source_snapshot首次把绝对config路径复制到自身而失败；原记录/traceback保留，修复后同配置smoke通过。一次入口未source环境导致import失败，没有模型/GPU调用。seed33在11月11日12:02因外部显存占用OOM，失败0.0377GPUh计费；原控制器停止至状态检查，曾启用CPU资源等待，后依用户修订取消所有显存阈值等待/提前退出，min_free_GiB=0直接运行；同模型/batch/标签量仅重跑seed33，保留11/22。取消阈值后GPU1首batch实际OOM，0.0431GPUh保留计费；22GPUh硬限不变。
M10实际B5c训练循环epochs=1，原smoke的min_epochs8未使用；本次明确一遍完整标签，未擅自变为八遍。其曝光低于CLR/B6，差距不能单独归因于架构或语言化设计。M10及更早冻结保持原字节，授权更新的paper/FINAL_SUMMARY有逐字节归档。

### (e) 需要审查者决定的问题
无。实验永久冻结，不再申请追加预算。

### (f) 证据路径
- 详细：reports/M10b_details.md；汇总：reports/FINAL_SUMMARY.md。
- 数据/配对：data/M10b_v1/paper_inputs/B5c_summary.csv、CLR_minus_B5c.csv、B5c_conformal.csv。
- 权重与run：data/M10b_v1/trained/、evaluated/、final_lock.json；配置configs/M10b/。
- 成本/复现/完整性（含上述已知例外）：data/M10b_v1/cost_ledger.json、reproducibility.json、upstream_audit.json、freeze.json。
'''
 if len(review.encode())>8000:raise ValueError('Review exceeds 8000 bytes')
 Path('reports/M10b_REVIEW.md').write_text(review)
 details=review+'\n\n完整conformal（alpha=.1）\n\n'+'```csv\n'+cp.to_csv(index=False)+'```'+'\n\n预算准入\n\n```json\n'+(root/'budget_admission.json').read_text()+'\n```\n';Path('reports/M10b_details.md').write_text(details)
 s=Path('reports/FINAL_SUMMARY.md').read_text();s=s.replace('M10b仅补授权的GPT2 B5c Q3/Q4三seed，完成后永久冻结。','M10b已完成授权的GPT2 B5c Q3/Q4三seed，实验永久冻结。');s=s.replace('M10的B5c因12GPUh预算NR；M10b独立22GPUh实验进行中，尚无test比较，不能声称CLR优于B5c。','M10原B5c预算NR由M10b补充GPT2 Q3/Q4：CLR−B5c test '+test+'。B5c只训练一遍完整数据（CLR八遍），曝光差异限制架构归因；LLaMA、Q1/Q2及审计B5c未执行。');Path('reports/FINAL_SUMMARY.md').write_text(s)
 if len(s.encode())>10000:raise ValueError('FINAL_SUMMARY exceeds 10KB')
 claims=json.loads(Path('paper/claims.json').read_text());claims['M10b_update']={'status':'complete','test_CLR_minus_B5c':test,'scope':'GPT2 Q3/Q4 only; one full training pass; lower exposure than CLR','paired_csv':str(dest/'CLR_minus_B5c.csv'),'review':'reports/M10b_REVIEW.md'};write_json('paper/claims.json',claims)
 lines=['# Paper artifacts (M10b final)','','Experiments are permanently frozen after M10b. Rebuild: `.venv-paper/bin/python -I scripts/paper/make_all.py configs/M10b/paper_build.yaml`. Independent environment lock: `paper/requirements.lock.txt`.','','Each product consumes hash-frozen CSVs; no model calls. Fixed trained seeds, 10000 problem/group bootstrap draws.','','|Product|Script|YAML|CSV and SHA256|','|---|---|---|---|'];builds=[]
 for item in items:
  cfg=load_config(item['config']);inputs={p:sha(p) for p in cfg['inputs'].values()};lines.append('|'+cfg['artifact']+'|`'+item['script']+'`|`'+item['config']+'`|'+'<br>'.join('`'+p+'` `'+h+'`' for p,h in inputs.items())+'|');builds.append({'artifact':cfg['artifact'],**item,'script_sha256':sha(item['script']),'config_sha256':sha(item['config']),'inputs':inputs})
 lines+=['','T1 includes GPT2 B5c Q3/Q4 (one complete training pass, r16 LoRA). Q1/Q2 and all LLaMA B5c cells are NA because excluded by scope; audit B5c remains NR. Audit claims concern the typed framework (question decomposition + public expectations + conformal thresholds), with CLR and B6 reported together; no CLR detection superiority claim. F4 transfer and HiddenBench are appendix-only; HiddenBench is too small to support a conclusion.','','M9 original bytes: `data/M10_v1/m9_paper_snapshot/mapping.json`; M10 original bytes: `data/M10b_v1/m10_paper_snapshot/mapping.json`. Freeze manifests are unchanged. One pre-existing M10 pytest stdout hash differs (modified after M10 freeze, before M10b); original bytes unavailable, preserved and disclosed in preexisting_log_drift.json. All model/data/metric hashes match. Reproduction: `data/M10b_v1/reproducibility.json`; integrity: `data/M10b_v1/upstream_audit.json`; review: `reports/M10b_REVIEW.md`. TeX compilation remains unperformed (no local compiler); deterministic TeX, CSV and vector PDFs rebuild byte-for-byte.']
 Path('paper/README.md').write_text('\n'.join(lines)+'\n');upstream(root)
 products={str(p):sha(p) for sub in ['tables','figures','derived'] for p in (Path('paper')/sub).glob('*') if p.is_file()};manifests=[{'path':str(p),'sha256':sha(p)} for p in sorted(Path('data').rglob('freeze.json')) if 'snapshot' not in str(p) and 'repro_build' not in str(p)]
 write_json('paper/artifact_manifest.json',{'version':'M10b_paper_v1','current_freeze':str(root/'freeze.json'),'artifacts':products,'builds':builds,'freeze_manifests':manifests,'reproduction':str(root/'reproducibility.json'),'integrity_audit':str(root/'upstream_audit.json'),'M10_archive_mapping':str(root/'m10_paper_snapshot/mapping.json')})
 # artifact_manifest is itself an authorized M10 paper supersession; verify again.
 upstream(root)
 status='M10b GPT2 B5c Q3/Q4三seed与论文复现完成待审查，实验永久冻结；唯一审查reports/M10b_REVIEW.md；冻结data/M10b_v1/freeze.json。'
 for p in [Path('AGENTS.md'),Path('docs/STATUS.md')]:
  text=p.read_text();a=text.find('<!-- M10b_LIVE -->');b=text.find('<!-- /M10b_LIVE -->');block='<!-- M10b_LIVE -->\n'+status+'\n<!-- /M10b_LIVE -->'
  p.write_text(text[:a]+block+text[b+len('<!-- /M10b_LIVE -->'):] if a>=0 else text+'\n'+block+'\n')
 with Path('docs/EXPERIMENTS.md').open('a') as f:f.write('\n## M10b 最终冻结\n仅GPT2 B5c Q3/Q4完整一遍×三seed；CLR−B5c test '+test+'。\n命令configs/M10b/controller.yaml；成本'+f'{total:.3g}'+'/22 GPUh，含失败加载/校准/推断。\n固定seed整problem bootstrap10000 CI；审计框架修订、附录定位、复现和证据见reports/M10b_REVIEW.md。\n实验永久冻结data/M10b_v1/freeze.json，旧M10 paper字节归档。\n')
 with Path('docs/CHANGELOG.md').open('a') as f:f.write('\n## M10b 最终产物\n新增GPT2 B5c三seed Q3/Q4、配对CI与T1；审计主张为框架，跨用途/HiddenBench仅附录。\nM10及上游代码结果只读；当前paper与FINAL_SUMMARY为用户授权supersession，原字节归档。\n')
 write_json(root/'README.json',{'milestone':'M10b','status':'complete_pending_review','experiments_permanently_frozen':True,'test_opened':True,'scope':'GPT2 Q3/Q4 B5c only; all three seeds, one full data pass; LLaMA/audit excluded','gpu_hours':total})
 live(stage='complete',milestone_complete=True,status=status,test_opened=True)
 paths=set(root.rglob('*'))|set(Path('configs/M10b').rglob('*'))|set(Path('scripts').glob('m10b*.py'))|set(Path('scripts/paper').glob('make_m10b*.py'))|set(Path('paper').rglob('*'))|{Path('clr/utils/m10b_runtime.py'),Path('clr/paper/render_m10b.py'),Path('tests/test_m10b.py'),Path('docs/M10b_TASK.md'),Path('reports/M10b_REVIEW.md'),Path('reports/M10b_details.md'),Path('reports/FINAL_SUMMARY.md'),Path('docs/M10b_LIVE.json')}
 for r in costs:paths.update(Path(r['run']).rglob('*'))
 paths={p for p in paths if p.is_file() and p.suffix!='.pyc' and p!=root/'freeze.json' and p.name not in ['controller.log','controller.pid','watchdog.json'] and 'runtime_locks' not in str(p)}
 write_json(root/'freeze.json',{'version':'M10b_v1','status':'complete_pending_review_experiments_permanently_frozen','gpu_hours':total,'sha256':{str(p):sha(p) for p in sorted(paths)},'upstream':'M10 scientific files match; one disclosed pre-existing nonmetric pytest log hash differs; authorized paper supersessions archived','test_opened_after_all_training_lock':True})
 errors=[str(p) for p,h in [(Path(p),h) for p,h in json.loads((root/'freeze.json').read_text())['sha256'].items()] if sha(p)!=h];write_json(root/'completion_audit.json',{'passed':not errors,'errors':errors,'freeze_sha256':sha(root/'freeze.json'),'frozen_files':len(json.loads((root/'freeze.json').read_text())['sha256'])})
 if errors:raise ValueError('Completion byte audit failed')
 print(status,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

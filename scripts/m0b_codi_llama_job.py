"""Versioned sequential smoke -> official reproduction -> CLR comparison job."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from clr.utils.config import load_config
p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();c=load_config(a.config)
status={'started':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'status':'running','runs':{}};dest=Path(c['status'])
if dest.exists():raise RuntimeError('Job status exists; do not overwrite an earlier job')
def save():dest.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
def step(name,script,config,overrides=()):
 status['stage']=name;save();log=Path(c['log_dir'])/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,PYTHONPATH='.',PYTHONUTF8='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2')
 with log.open('x',encoding='utf-8') as f:subprocess.run([sys.executable,script,config,*overrides],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
 line=log.read_text(encoding='utf-8').strip().splitlines()[-1];run=line.split()[0]
 metrics=json.loads(Path(run,'metrics.json').read_text())
 if metrics['status']!='completed':raise RuntimeError('Stage did not complete')
 status['runs'][name]=run;save();return run
save()
try:
 proof=json.loads(Path(c['download_run'],'metrics.json').read_text())
 if proof['status']!='completed':raise RuntimeError('Verified checkpoint download required')
 s=step('source_smoke','scripts/m0b_codi_llama_eval_v2.py',c['source_smoke'])
 o=step('official_smoke','scripts/m0b_codi_reference_v2.py',c['official_smoke'])
 d=step('determinism','scripts/m0b_codi_llama_determinism_v2.py',c['determinism'])
 r=step('official_full','scripts/m0b_codi_reference_v2.py',c['official_full'],['smoke_run='+o])
 f=step('source_full','scripts/m0b_codi_llama_eval_v2.py',c['source_full'],['smoke_run='+s,'reference='+str(Path(r)/'predictions.json')])
 status['status']='completed';save()
 # Reproduction only; noise and milestone review remain separate.
 a=json.loads(Path(r,'metrics.json').read_text());b=json.loads(Path(f,'metrics.json').read_text())
 text='# M0b CODI-LLaMA 复现 v2\n\n作者checkpoint与ModelScope底模均已校验。官方脚本未修改；strict加载只用于拒绝遗漏权重。\n\n'
 text+='运行清单：\n```json\n'+json.dumps(status['runs'],indent=2)+'\n```\n\n'
 text+='官方原始1319题准确率及按题10k bootstrap、清洁1312子集：\n```json\n'+json.dumps(a,indent=2)+'\n```\n\n'
 text+='CLR batch16准确率及与官方逐题一致率：\n```json\n'+json.dumps(b,indent=2)+'\n```\n\n'
 diff=100*a['accuracy']['mean']-c['paper_reference_percent']
 text+=f"与既有M0报告暂列论文参考{c['paper_reference_percent']}%差值为{diff:.3f}pp。"
 text+='若绝对差>2pp，须继续核验checkpoint/论文配置对应关系、prompt及精度协议，不能通过改划分/挑seed追平。该数值不是已完成原因分析。\n\n'
 text+='确定性逐题潜状态差见determinism run；本轮没有训练/噪声曲线结果，不宣称M0b整体验收。\n'
 Path(c['report']).write_text(text,encoding='utf-8')
 with open('docs/EXPERIMENTS.md','a',encoding='utf-8') as out:out.write('\n## CODI-LLaMA v2复现流水线结束\n详见 `'+c['report']+'`；噪声与最终审查待完成。\n')
except BaseException:
 status['status']='failed';status['error']=traceback.format_exc();save()
 with open('docs/BLOCKERS.md','a',encoding='utf-8') as out:out.write('\n## CODI-LLaMA v2流水线阻塞\n'+status['error']+'\n未降级模型、batch或精度。阶段日志见 '+c['log_dir']+'。\n')
 raise

"""M10c CPU paper build, evidence mapping, audit and immutable completion."""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import yaml
from clr.paper.m10c_cost import sha
from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def audit(root):
    archives={}
    for mp in ['data/M10_v1/m9_paper_snapshot/mapping.json','data/M10b_v1/m10_paper_snapshot/mapping.json',str(root/'m10b_paper_snapshot/mapping.json')]:
        for p,r in json.loads(Path(mp).read_text()).items(): archives[(str(Path(p).resolve()),r['sha256'])]=r['archive']
    prior=json.loads(Path('data/M10b_v1/upstream_audit.json').read_text())
    manifests=list(dict.fromkeys(list(prior['manifests'])+['data/M10b_v1/freeze.json']))
    expected={}
    for mp in manifests:
        d=json.loads(Path(mp).read_text())
        for p,h in d.get('sha256',d.get('files',{})).items():
            if isinstance(h,str):expected.setdefault(str(Path(p).resolve()),[]).append((mp,h))
    known=json.loads(Path('data/M10b_v1/preexisting_log_drift.json').read_text())
    exceptions=[known]+json.loads((root/'preflight_integrity.json').read_text())['M10b_changed']
    allowed={(str(Path(r['path']).resolve()),r['expected'],r['actual']) for r in exceptions}
    def check(item):
        p,versions=item;current=sha(p) if Path(p).exists() else None; errors=[];sup=[]
        for mp,h in versions:
            if current==h:continue
            archive=archives.get((p,h))
            if archive and sha(archive)==h:sup.append(dict(path=p,manifest=mp,archive=archive,sha256=h))
            else:errors.append(dict(path=p,manifest=mp,expected=h,actual=current,known_nonmetric=(p,h,current) in allowed))
        return errors,sup
    with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(check,expected.items()))
    errors=[r for a,b in results for r in a];unexpected=[r for r in errors if not r['known_nonmetric']]
    write_json(root/'upstream_audit.json',dict(passed=not errors,scientific_inputs_intact=not unexpected,
               unique_files=len(expected),errors=errors,unexpected_errors=unexpected,
               authorized_supersessions=[r for a,b in results for r in b],manifests={p:sha(p) for p in manifests}))
    if unexpected:raise ValueError('Unexpected upstream hash drift: '+str(unexpected[:3]))


def main(c):
    root=Path(c['output'])
    if (root/'freeze.json').exists():raise ValueError('Already frozen')
    f=pd.read_csv(root/'paper_inputs/cost_precision.csv');cp=pd.read_csv(root/'paper_inputs/conformal_paired.csv')
    def cell(model,q,b=1):return f[(f.model==model)&(f.qtype==q)&(f.batch==b)].iloc[0]
    def ci(split,q,m,scale=1):
        r=cp[(cp.split==split)&(cp.qtype==q)&(cp.metric==m)].iloc[0]
        return f'{r["mean"]*scale:.3g} [{r.lo*scale:.3g},{r.hi*scale:.3g}]'
    ratios={}
    for q in ['Q3','Q4']:
        for b in [1,64]:
            a=cell('CLR-lite',q,b);d=cell('B5c',q,b)
            ratios[f'{q}_b{b}']=dict(flops=float(d.flops_per_query/a.flops_per_query),latency=float(d.latency_per_batch_ms/a.latency_per_batch_ms))
    write_json(root/'cost_ratios.json',ratios)
    items=load_config('configs/M10b/paper_build.yaml')['artifacts']
    for name,inputs,caption in [('t6_cost_precision',{'cost':str(root/'paper_inputs/cost_precision.csv')},'T6: CODI-GPT2 precision and query cost'),('t6_conformal',{'conformal':str(root/'paper_inputs/conformal_paired.csv')},'T6 companion: frozen conformal coverage and paired set sizes')]:
        p=Path('configs/M10c/paper')/(name+'.yaml');p.parent.mkdir(exist_ok=True)
        p.write_text(yaml.safe_dump(dict(artifact=name,caption=caption,input_freeze=str(root/'paper_inputs/freeze.json'),inputs=inputs),sort_keys=False))
        items.append(dict(script='scripts/paper/make_'+name+'.py',config=str(p)))
    for name,out in [('paper_build','paper'),('paper_repro',str(root/'repro_build'))]:
        p=Path('configs/M10c')/(name+'.yaml');p.write_text(yaml.safe_dump(dict(output_root=out,isolated=True,artifacts=items),sort_keys=False))
        with (root/(name+'.log')).open('w') as log:subprocess.run([c['paper_python'],'-I','scripts/paper/make_all.py',str(p)],stdout=log,stderr=subprocess.STDOUT,check=True)
    products=[]
    for sub in ['tables','figures','derived']:
        a={p.name:p for p in (Path('paper')/sub).glob('*') if p.is_file()};b={p.name:p for p in (root/'repro_build'/sub).glob('*') if p.is_file()}
        if a.keys()!=b.keys():raise ValueError('Repro inventory differs')
        for n,p in a.items():products.append(dict(path=str(p),sha256=sha(p),repeat_sha256=sha(b[n]),byte_identical=sha(p)==sha(b[n])))
    if not all(r['byte_identical'] for r in products):raise ValueError('Repro bytes differ')
    modules=json.loads(subprocess.run([c['paper_python'],'-I','-c','import importlib.util,json;print(json.dumps({k:importlib.util.find_spec(k) is not None for k in ["torch","transformers","numpy","pandas","matplotlib","yaml"]}))'],capture_output=True,text=True,check=True).stdout)
    if modules['torch'] or modules['transformers']:raise ValueError('Paper environment not independent')
    write_json(root/'reproducibility.json',dict(passed=True,products=products,environment_modules=modules,method='independent locked .venv-paper, -I, two output roots; byte-identical TeX/CSV/vector PDF'))
    # Current paper and prose supersede archived bytes; no historical scientific file changes.
    prose=Path(c['summary']).read_text()
    prose=prose.replace('M10及上游实验保持冻结；M10b已完成授权的GPT2 B5c Q3/Q4三seed，实验永久冻结。','M10及上游实验保持冻结；M10b已完成GPT2 B5c Q3/Q4三seed，实验永久冻结。M10c仅只读测量开销并汇总已有CSV，未训练或重校准。')
    title='## C1 状态与选项交互读取的架构比较（新增B6后按类型限定）'
    cost_sentence=(f'在已测两个CODI源的Q3/Q4上优于MLP与注意力探针；在CODI-GPT2上与训练式语言化解码器精度相当（描述性），但单查询FLOPs低约{ratios["Q3_b1"]["flops"]:.3g}倍（Q3）/{ratios["Q4_b1"]["flops"]:.3g}倍（Q4）。')
    prose=prose.replace(title,'## C1 优于MLP与注意力探针；与训练式语言化解码器精度相当但计算更省\n\n'+cost_sentence)
    note=(f'\n\nM10c T6：GPT2 CLR−B5c test Q3 −0.659 [−1.51,0.190]pp，Q4 −0.0864 [−0.984,0.830]pp；CI跨零并不证明统计等价。FLOPs倍数含B5c冻结Qwen3完整前向；batch64为{ratios["Q3_b64"]["flops"]:.3g}/{ratios["Q4_b64"]["flops"]:.3g}倍。共享RTX4090上batch1延迟低{ratios["Q3_b1"]["latency"]:.3g}/{ratios["Q4_b1"]["latency"]:.3g}倍，batch64低{ratios["Q3_b64"]["latency"]:.3g}/{ratios["Q4_b64"]["latency"]:.3g}倍；并发负载/预处理排除限制外推。训练成本口径与任务曝光不同；B3独立耗时缺失。沿用cal_B阈值α=.1，test B5c−CLR size配对差Q3 {ci("test","Q3","size_difference")}，Q4 {ci("test","Q4","size_difference")}，覆盖率不同，不是同覆盖率优势。\n')
    prose=prose.replace('\n\n## C2 ',note+'\n\n## C2 ',1)
    prose=prose.replace('T1–T5、F1–F3及附录','T1–T6、F1–F3及附录')
    prose+='\n\nM10c复现说明：reports/M10_evidence/tests_final.log在M10冻结后、M10b开始前被改写；证据data/M10b_v1/preexisting_log_drift.json。原日志不可恢复，保留完整性例外，不能宣称上游全字节通过；模型、数据、预测和指标哈希一致，不影响科学结论。M10c另记录启动前已有M10b控制器追加stdout及LIVE索引漂移；新审计data/M10c_v1/upstream_audit.json。旧paper/总结归档data/M10c_v1/m10b_paper_snapshot/mapping.json；新表图双目录逐字节复现见data/M10c_v1/reproducibility.json。\n'
    if len(prose.encode())>10000:raise ValueError('FINAL_SUMMARY >10KB')
    Path(c['summary']).write_text(prose)
    claims=json.loads(Path('paper/claims.json').read_text());claims['claims'][0]['claim']=cost_sentence
    claims['claims'][0]['evidence']+=' M10c T6：'+note.strip()
    claims['M10_update']['limitations']=[s.replace('B5c full comparison unavailable under budget','M10b completed GPT2 Q3/Q4 B5c; LLaMA and audit remain unavailable') for s in claims['M10_update']['limitations']]
    claims['M10c_update']=dict(review=c['review'],cost_ratios=ratios,conformal='paper/tables/t6_conformal.tex',no_training=True,no_recalibration=True)
    write_json('paper/claims.json',claims)
    readme=Path('paper/README.md').read_text().replace('# Paper artifacts (M10b final)','# Paper artifacts (M10c read-only cost update)').replace('configs/M10b/paper_build.yaml','configs/M10c/paper_build.yaml',1)
    readme+='\nM10c adds no training or scientific predictions. T6 raw measurement rebuild: `source scripts/m7_env.sh && .venv/bin/python scripts/m10c_measure.py configs/M10c/measure.yaml` (refuses overwrite; use a new version for remeasurement). Aggregation: `.venv/bin/python scripts/m10c_summarize.py configs/M10c/summarize.yaml`; no calibration. Cost scope and concurrency limitations are in T6 and reports/M10c_details.md.\n\n|Product|Script|YAML|CSV and SHA256|\n|---|---|---|---|\n'
    builds=[]
    for item in items:
        cfg=load_config(item['config']);inputs={p:sha(p) for p in cfg['inputs'].values()}
        builds.append(dict(artifact=cfg['artifact'],**item,script_sha256=sha(item['script']),config_sha256=sha(item['config']),inputs=inputs))
        if cfg['artifact'].startswith('t6_'):readme+='|'+cfg['artifact']+'|`'+item['script']+'`|`'+item['config']+'`|'+'<br>'.join('`'+p+'` `'+h+'`' for p,h in inputs.items())+'|\n'
    readme+='\nM10c original M10b paper bytes: `data/M10c_v1/m10b_paper_snapshot/mapping.json`. Current reproduction: `data/M10c_v1/reproducibility.json`; current integrity exceptions: `data/M10c_v1/upstream_audit.json` (M10 pytest stdout and two pre-existing M10b administrative files; scientific hashes intact).\n'
    Path('paper/README.md').write_text(readme)
    manifest=json.loads(Path('paper/artifact_manifest.json').read_text());manifest.update(version='M10c_paper_v1',current_freeze=str(root/'freeze.json'),artifacts={r['path']:r['sha256'] for r in products},builds=builds,reproduction=str(root/'reproducibility.json'),integrity_audit=str(root/'upstream_audit.json'),M10b_archive_mapping=str(root/'m10b_paper_snapshot/mapping.json'))
    manifest['freeze_manifests']=[dict(path=str(p),sha256=sha(p)) for p in sorted(Path('data').rglob('freeze.json')) if 'snapshot' not in str(p) and 'repro_build' not in str(p)]
    write_json('paper/artifact_manifest.json',manifest)
    audit(root)
    write_json(root/'paper_completed.json',dict(completed=True,artifacts=len(items),no_training=True,no_new_test_inference=True))
    print('Paper and integrity complete; write review/status then freeze using m10c_freeze.py',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');main(load_config(p.parse_args().config))

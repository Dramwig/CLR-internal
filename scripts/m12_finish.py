"""Evidence checks, compact review, records and immutable M12 artifact manifest."""
import argparse,json,subprocess
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m12_runtime import digest,cost
from clr.calib.m12_audit import audit_summary,audit_permutation

ROOT=Path('data/M12_ablation')
def fmt(x):return f'{x:.3g}'
def main(c):
    if (ROOT/'freeze.json').exists():raise ValueError('M12 is frozen; use a new version for later changes')
    subprocess.run(['.venv/bin/python','-m','pytest','tests/test_m12.py','-q'],check=True,stdout=Path('reports/M12_evidence/unit_final.log').open('w'),stderr=subprocess.STDOUT)
    lock=json.loads((ROOT/'training_complete.json').read_text());registry=json.loads((ROOT/'source_registry.json').read_text());summary=pd.read_csv(ROOT/'summary.csv');perm=pd.read_csv(ROOT/'permutation.csv')
    if len(summary)!=8*(5+len(lock['retained'])+len(lock['dropped'])):raise ValueError('Missing summary cells')
    completion=audit_summary(summary,lock['retained'],lock['dropped'])
    completion['permutation']=audit_permutation(perm,pd.read_parquet(ROOT/'permutation_observations.parquet'),lock['retained'])
    paper=json.loads((ROOT/'paper_readonly_audit.json').read_text())
    aliases={r['historical_path']:r['archive_path'] for r in paper['archive_resolution']}
    paper_mismatches=[p for p,h in paper['expected_sha256'].items() if not Path(aliases.get(p,p)).is_file() or digest(aliases.get(p,p))!=h]
    write_json(ROOT/'paper_final_audit.json',{'files':len(paper['expected_sha256']),'mismatches':paper_mismatches,'passed':not paper_mismatches})
    if paper_mismatches:raise ValueError('Paper content differs from preceding frozen versions')
    for p,h in lock['code_sha256'].items():
        if digest(p)!=h:raise ValueError('Scientific code changed after training lock: '+p)
    access=json.loads((ROOT/'test_access.json').read_text())
    if datetime.fromisoformat(lock['created']).timestamp()>access['timestamp']:raise ValueError('Test opened before training lock')
    if set(lock['runs'])!={f'{a}_{s}' for a in lock['retained'] for s in [11,22,33]}:raise ValueError('Training seed registry incomplete')
    splits=json.loads((ROOT/'split_metadata_audit.json').read_text())
    if not splits['passed'] or any(digest(p)!=h for p,h in splits['sha256'].items()):raise ValueError('Frozen split metadata differs from leakage audit')
    original_runs=json.loads(Path('data/M41_v1/training_suite.json').read_text())
    protected=['root','qa_root','protocol','templates','cal_split','model_path','batch_size','eval_batch','encoder_lr','head_lr','weight_decay','epochs','min_epochs','patience','variant','max_length','initial_checkpoint','fewshot_n','qtypes']
    for key,path in lock['runs'].items():
        if digest(Path(path)/'best.pt')!=lock['checkpoint_sha256'][key]:raise ValueError('Changed checkpoint')
        m=json.loads(Path(path,'metrics.json').read_text());manifest=json.loads(Path(path,'training_manifest.json').read_text())
        if m['status']!='completed' or m['epochs']<8 or manifest['test_opened']:raise ValueError('Training protocol failure')
        expected={'Q1':740228,'Q2':140000,'Q3':860228,'Q4':279440}
        if manifest['train_counts']!=expected:raise ValueError('Changed training population')
        if any(m['examples'][q]!=m['epochs']*n for q,n in expected.items()):raise ValueError('Incomplete exposure')
        a,seed=key.split('_');cfg=json.loads(Path(path,'config.json').read_text());ref=json.loads(Path(original_runs[f'main_lite_{seed}'],'config.json').read_text())
        if cfg['case']!=a or cfg['ablation']!=a or cfg['seed']!=int(seed) or cfg['milestone']!='M12':raise ValueError('Training identity mismatch')
        numeric={'encoder_lr','head_lr','weight_decay'}
        differences=[k for k in protected if (float(cfg[k])!=float(ref[k]) if k in numeric else cfg.get(k)!=ref.get(k))]
        if differences:raise ValueError('Training differs from frozen M41 settings: '+str(differences))
        if cfg.get('augmentation') or cfg.get('mode')!='full':raise ValueError('Unexpected training data mode')
        if cfg['total_budget_hours']!=30 or cfg['deadline']!=c['deadline']:raise ValueError('Current M12 budget/deadline differs')
        total=sum(expected.values());weights={q:(1. if a=='A5' else total/(4*n)) for q,n in expected.items()}
        if manifest['objective_weights']!=weights or not manifest['all_instances_per_epoch']:raise ValueError('Type objective/sampling changed')
        if m['optimizer_steps']!=sum((n+cfg['batch_size']-1)//cfg['batch_size'] for n in expected.values())*m['epochs']:raise ValueError('Optimizer exposure mismatch')
        start_hash=json.loads(Path(path,'code_sha256.json').read_text())
        for p in ['clr/models/m12.py','clr/data/m12.py','scripts/m12_train.py']:
            if start_hash[p]!=lock['code_sha256'][p]:raise ValueError('Scientific training code changed between runs')
        history=json.loads(Path(path,'history.json').read_text());selected=json.loads(Path(path,'best.json').read_text())
        eligible=[r for r in history if r['epoch']>=cfg['min_epochs']]
        chosen=min(eligible,key=lambda r:r['selection_nll'])
        if selected['epoch']!=chosen['epoch'] or selected['selection_nll']!=chosen['selection_nll']:raise ValueError('Early stopping selection differs')
        if len(history)!=m['epochs'] or [r['epoch'] for r in history]!=list(range(1,m['epochs']+1)):raise ValueError('Epoch history incomplete')
        best=float('inf');bad=0;stopped=False
        for row in history:
            if stopped:raise ValueError('Training continued after patience threshold')
            if row['epoch']>=cfg['min_epochs']:
                if row['selection_nll']<best:best=row['selection_nll'];bad=0
                else:bad+=1
                stopped=bad>=cfg['patience']
        if not stopped and m['epochs']!=cfg['epochs']:raise ValueError('Training stopped before original early-stop rule')
    files={}
    for name in ['M23_v1a','M4_v0','M41_v1','M5b_v1','M9_v1','M10_v1']:
        source=Path('data',name,'freeze.json');data=json.loads(source.read_text());files[str(source)]=digest(source)
        for p,h in data['sha256'].items():
            # Paper historical versions intentionally superseded by later tasks;
            # only check scientific inputs actually consumed in this experiment.
            if p.startswith(('paper/','manuscripts/')):continue
            files[str(Path(p).resolve())]=h
    used=json.loads((ROOT/'statistics_provenance.json').read_text())['source_sha256'];used.update({p:lock['code_sha256'][p] for p in lock['code_sha256'] if str(Path(p).resolve()) in files})
    new_runs={Path(v).resolve() for v in lock['runs'].values()}
    new_runs.update(Path(v).resolve() for v in json.loads((ROOT/'evaluation_complete.json').read_text()).values())
    mismatches=[];missing=[]
    for p,h in used.items():
        expected=files.get(str(Path(p).resolve()))
        if expected is None:
            # New M12 outputs are frozen below, not claimed as historical.
            if not Path(p).resolve().is_relative_to(ROOT.resolve()) and not any(Path(p).resolve().is_relative_to(v) for v in new_runs):missing.append(p)
            continue
        actual=digest(p)
        if actual!=h or actual!=expected:mismatches.append(p)
    # Check every recorded historical checkpoint used for the new permutation diagnostic.
    train=json.loads(Path('data/M41_v1/training_suite.json').read_text())
    for s in [11,22,33]:
        p=str(Path(train[f'main_lite_{s}'])/'best.pt');rel=str(Path(p).relative_to(Path.cwd())) if Path(p).is_absolute() else p
        h=files.get(str(Path(p).resolve()))
        if h is None or digest(p)!=h:mismatches.append(p)
    write_json(ROOT/'upstream_audit.json',{'checked_consumed_files':len(used),'mismatches':mismatches,'missing_frozen_source_entries':missing,'historical_manifest_sha256':{k:v for k,v in files.items() if k.endswith('/freeze.json')}})
    if mismatches or missing:raise ValueError('Upstream input freeze audit failed')
    total=cost()
    if total>30:raise ValueError('Budget exceeded')
    costrows=[]
    for p in sorted((ROOT/'jobs').glob('*.json')):
        j=json.loads(p.read_text());r=Path(j['run']);m=json.loads((r/'metrics.json').read_text());cfg=json.loads((r/'config.json').read_text());costrows.append({'run':str(r),'case':cfg['case'],'stage':cfg.get('stage'),'mode':cfg['mode'],'seed':cfg['seed'],'device':cfg.get('reader_device'),'status':m['status'],'gpu_hours':max(m.get('gpu_hours',0),json.loads((r/'budget.json').read_text()).get('own_gpu_hours',0) if (r/'budget.json').exists() else 0),'source_forward_calls':m.get('source_forward_calls',0)})
    pd.DataFrame(costrows).to_csv(ROOT/'costs.csv',index=False);write_json(ROOT/'costs.json',{'gpu_hours':total,'limit':30,'source_forward_calls':0,'gpu':'NVIDIA GeForce RTX 4090','scope':'sum entire GPU job walltime incl CPU data preparation, smoke/failure/calibration/permutation; CPU-only controllers/statistics excluded'})
    completion.update(training_lock_before_test=True,gpu_hours=total,budget_gpu_hours=30,source_forward_calls=0,upstream_audit='data/M12_ablation/upstream_audit.json')
    write_json(ROOT/'completion_audit.json',completion)
    lines=['# M12 CLR-lite组件消融','', '(a) 完成待审查；仅GPT-2，不改论文。', '', '(b) 通过：val smoke、三seed/完整曝光、先锁hash再test、按题10⁴ bootstrap。', '', '(c) test均值（CI与配对差见CSV）', '', '|模型/来源|Q1 NLL↓|Q2 AUC↑|Q3↑|Q4↑|C3|C4|','|---|---:|---:|---:|---:|---:|---:|']
    order=['CLR-lite',*lock['retained'],'CLR-lite-single','CLR-per-type','CLR-v1','CLR-lite-multi'];labels={'CLR-lite':'主/M41','CLR-lite-single':'single/M5b','CLR-per-type':'per-type/M10','CLR-v1':'v1/M41','CLR-lite-multi':'双源/M9'}
    for a in order:
        g=summary[(summary.variant==a)&(summary.split=='test')].set_index('qtype')
        vals=[g.loc[q,'mean'] for q in ['Q1','Q2','Q3','Q4']]+[g.loc[q,'conformal_size'] for q in ['Q3','Q4']]
        lines.append('|'+labels.get(a,a)+'|'+'|'.join(fmt(x) for x in vals)+'|')
    lines+=['','A1全注意力；A2点积头；A3无数值；A4晚融合；A5无类型权重。','C3/C4：cal_B α=.1集合大小；复用行使用原校准/预测。']
    pp=perm[(perm.split=='test')&perm.canonical.astype(bool)]
    perm_run=json.loads((ROOT/'permutation_complete.json').read_text())['run']
    canonical_details=[r for r in json.loads(Path(perm_run,'details.json').read_text()) if r['canonical']]
    tie_note='原接口变化均为最高分并列，对齐logits差0。' if canonical_details and all(r['max_abs_logit_difference']==0 and r['changes']==r['changes_with_original_top_ties'] for r in canonical_details) else ''
    lines+=['Q3置换变化率：'+ '；'.join(f"{r.variant}={fmt(r['mean'])}" for _,r in pp.iterrows())+'。原接口先排序；raw结果亦见CSV。', '', '(d) '+('放弃：'+','.join(lock['dropped'])+'（成本/截止准入）。' if lock['dropped'] else '无未完变体。')+'A1保留原共享位置；复用设计并非单组件控制。'+('A4/11共享显存OOM后同设置换卡重跑，失败计费。' if (ROOT/'recovery.json').exists() else '')+('置换导入失败后显式加载现有C++库恢复。' if (ROOT/'postprocess_environment_recovery.json').exists() else ''), '', '(e) 无。', '', '(f) `data/M12_ablation/summary.csv`、`permutation.csv`；详情`reports/M12_details.md`。', '', f'成本 {fmt(total)}/30 GPUh，4090；含smoke/失败/校准/置换，源前向0。']
    if tie_note:lines.insert(lines.index('(e) 无。')-1,tie_note)
    review='\n'.join(lines)+'\n'
    if len(review.encode())>2000:raise ValueError('Review exceeds 2KB: '+str(len(review.encode())))
    Path('reports/M12_REVIEW.md').write_text(review)
    details=f'''# M12 详细证据

完成时间：{datetime.now().astimezone().isoformat()}；截止：{c['deadline']}。配置与组件定义：docs/M12_TASK.md。M41主模型真实配置复制到configs/M12/base.yaml；完整训练每epoch 2019896条QA，batch1024，无放回，宏NLL最早8/最多10/patience1。所有保留变体三seed，无test选择。

训练锁：data/M12_ablation/training_complete.json；逐run history/best/training_manifest可审计完整曝光。预算准入：admission.json；每个run config/git/env/seed/code_sha256/startup_source.tar.gz/日志/metrics/budget齐全。code快照包含未跟踪文件，git提交号不冒充覆盖它们。

统计：summary.csv为长表，每split/qtype/variant一行，seed_11/22/33、mean、95%按题cluster bootstrap10000次、variant-minus-CLR-lite配对差CI；固定三seed在每次抽样内平均，清洁状态/seen模板/t等权，每t内部按QA实例加权，Q2每t/seed AUROC后等权。CI条件于原校准集，不含重新抽校准集不确定性。reader_val_q1_clean_traj.csv另报Q1两类状态。主输出选择沿用M41，未用test调参。

Conformal：cal_A温度，cal_B按qtype/t Mondrian；每题/每t固定选一个实例，alpha.1；集合大小和经验覆盖率/CI见summary.csv及conformal_observations.parquet。覆盖保证依赖分组内校准/测试可交换性；clean与训练分布偏移下不额外承诺保证。复用行未重校准，逐stratum集合大小与原conformal.json对照误差≤1e-7。

置换：permutation_observations.parquet与permutation.csv；主/A1，三seed，在完整clean/seen Q3上逐行固定非恒等随机排列，按选项身份恢复后比较argmax，报告canonical接口及raw接口。A1只改变attention mask；原共享逻辑位置与规范化排序仍保留，因此零变化不表示选项隔离。raw浮点差异及最高分并列计数详情见permutation_complete.json所指run/details.json。

运行恢复：A4 seed11在共享GPU3首epoch一个优化步后OOM；失败run和0.0225GPUh保留，同模型/batch/bf16/曝光在GPU0从头重跑；A4 seed22/33不重启。接续控制器复用A1–A4完成回执，seed设备0/2/5。证据recovery.json、recovered_controller.json及原training_exception.json。

后处理环境恢复：全部最终评估完成后，置换入口在recorded/GPU初始化之前因Pillow/libLerc所需GLIBCXX_3.4.29缺失而退出。显式LD_PRELOAD现有latdefer/lib/libstdc++.so.6，依赖导入预检通过后仅接续置换/统计/交付；未修改锁定科学代码/配置，未重跑训练/评估。证据postprocess_environment_recovery.json、reports/M12_evidence/permutation.log及permutation_retry_environment.log。该失败未初始化GPU，不计作GPU作业，原CPU监督失败记录保留。

CPU统计提速：原高级索引bootstrap在2000题合成面板耗时41.1秒；改为相同RNG(11)整数抽样的计数矩阵乘法后0.398秒。固定10000次抽样、按题聚类/原步与seed聚合均不变；含缺失步骤、不等样本权重及三seed的合成对照中点估计/CI误差≤1e-14，抽样计数逐项相同。旧CPU统计进程被显式中止、日志和启动快照保留，新进程重新读取同一科学预测完成最终统计，无新GPU/推断/校准。证据statistics_runtime_recovery.json及statistics_counted_parity_tests.log；未改冻结旧实现。

论文只读核验：paper_readonly_audit.json；paper/及已存在的重命名归档manuscripts/CVPR27-CLR-achieve-2026-11-12共96文件与旧冻结SHA一致，未改稿/编译。

复用：source_registry.json逐seed指向原始结果，M41 lite/v1、M5b single、M10 per-type、M9联合两源；已有行不重训/重推理/重校准。仅新增任务所需主模型置换诊断。复用设计存在训练曝光/数据来源区别，不能解释为新增A1–A5的严格单组件对照。

数据：problem_metrics.parquet、auroc_observations.parquet、conformal_observations.parquet；原始预测evaluation_complete.json；源文件sha统计statistics_provenance.json；冻结输入核验upstream_audit.json。

成本：{total:.9g} GPUh≤30，4090；完整GPU作业墙钟总和，含smoke、初始化、数据装载、失败与推断；CPU控制器/统计不计GPUh；源前向0，旧里程碑成本不重复计入。成本逐项costs.csv。

代码数值验证：TensorSerializer与冻结Serializer四种类型真实val batch逐tensor相同；控制类与原Lite训练梯度逐bit一致，mask/padding/排列等变、A3数值消融、A4无模板中转、split隔离、Q1标签和AUROC配对bootstrap均有最小测试。启动前/收尾日志reports/M12_evidence/unit*.log。训练smoke20训练题、协议proxy50个reader_val题，cal_A/B/eval为15/15/20互斥；规模≤5%、每阶段≤900秒。smoke.json保存实际耗时。
'''
    Path('reports/M12_details.md').write_text(details)
    message=f'M12组件消融完成待审查；{len(lock["retained"])}变体三seed，{total:.3g}/30GPUh；唯一审查reports/M12_REVIEW.md，冻结data/M12_ablation/freeze.json；未改论文。'
    for filename in ['AGENTS.md','docs/STATUS.md']:
        p=Path(filename);s=p.read_text()
        if filename=='AGENTS.md':
            a=s.index('<!-- M12_LIVE -->');b=s.index('<!-- /M12_LIVE -->',a)+len('<!-- /M12_LIVE -->');s=s[:a]+'<!-- M12_LIVE -->\n'+message+'\n<!-- /M12_LIVE -->'+s[b:]
        else:s+='\n## M12完成\n'+message+'\n'
        p.write_text(s)
    with Path('docs/EXPERIMENTS.md').open('a') as f:f.write(f'\n## M12 — CLR-lite组件消融\n{message}\n入口：`.venv/bin/python -m scripts.m12_pipeline configs/M12/base.yaml`。\nQ1/Q2/Q3/Q4、按题CI和配对差：data/M12_ablation/summary.csv；置换permutation.csv。\n全部负面结果保留；详细reports/M12_details.md；复用来源逐行记录，无模型/曝光缩量。\n')
    with Path('docs/BLOCKERS.md').open('a') as f:f.write('\n## M12 — 解释范围\nA1保留原逻辑位置及规范化排序；零置换敏感性不能证明选项隔离，raw诊断另列。\n复用设计有不同训练数据/曝光；其差异不作严格单组件因果结论；bootstrap条件于原校准集。\n'+('成本/截止未准入：'+','.join(lock['dropped'])+'。\n' if lock['dropped'] else '新增未解决运行阻塞：无。\n')+'详细reports/M12_details.md；旧科学输入SHA核验见data/M12_ablation/upstream_audit.json。\n')
    with Path('docs/CHANGELOG.md').open('a') as f:f.write('\n## M12\n新增五项单组件消融与复用汇总、置换诊断、三seed/按题CI及配对差；旧冻结和论文不改。\n任务授权及细节docs/M12_TASK.md；证据reports/M12_details.md。\n')
    write_json('docs/M12_LIVE.json',{'stage':'completed_pending_review','message':message,'review':'reports/M12_REVIEW.md','gpu_hours':total,'test_opened':True,'retained':lock['retained'],'dropped':lock['dropped']})
    paths=sorted(set([*ROOT.rglob('*'),*Path('configs/M12').glob('*.yaml'),*Path('scripts').glob('m12*.py'),Path('clr/models/m12.py'),Path('clr/calib/m12_audit.py'),Path('clr/calib/m12_statistics.py'),Path('clr/data/m12.py'),Path('clr/utils/m12_runtime.py'),Path('tests/test_m12.py'),Path('reports/M12_REVIEW.md'),Path('reports/M12_details.md'),*Path('reports/M12_evidence').glob('*'),Path('docs/M12_TASK.md')]))
    run_paths={j['run'] for p in (ROOT/'jobs').glob('*.json') for j in [json.loads(p.read_text())]}
    recovery=ROOT/'statistics_runtime_recovery.json'
    if recovery.exists():run_paths.update(json.loads(recovery.read_text())['old_cpu_runs'])
    receipt=ROOT/'statistics_run.json'
    if receipt.exists():run_paths.add(json.loads(receipt.read_text())['run'])
    for r in run_paths:paths.extend(Path(r).glob('*'))
    receipt=ROOT/'recovered_controller.json'
    if receipt.exists():
        controller=Path(json.loads(receipt.read_text())['controller_run'])
        # This CPU supervisor is still waiting on us; freeze its startup evidence,
        # while its final metrics/events necessarily close after this child exits.
        paths.extend(controller/name for name in ['config.json','config.yaml','git.json','git_commit.txt','working-tree.patch','dependencies.json','environment.json','startup_code_sha256.json','startup_source.tar.gz'])
    active_logs={'controller.log','finish.log','controller_recovered.log','recovery_supervisor.log'}
    hashes={str(p):digest(p) for p in paths if p.is_file() and p!=ROOT/'freeze.json' and p.name not in active_logs}
    write_json(ROOT/'freeze.json',{'status':'completed_pending_review','created':datetime.now().astimezone().isoformat(),'review_bytes':len(review.encode()),'gpu_hours':total,'source_forward_calls':0,'sha256':hashes})
    write_json(ROOT/'verification.json',{'files':len(hashes),'mismatches':[p for p,h in hashes.items() if digest(p)!=h]})
    print(message,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

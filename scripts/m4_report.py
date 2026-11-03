"""Evidence-only M4 final reports; refuses missing required three-seed experiments."""
import argparse,json,hashlib,time
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m4_budget import hours_used
from clr.calib.m4 import apply

METRICS={'Q1':'nll','Q2':'auroc','Q3':'accuracy','Q4':'accuracy'}
def interval(values):
    v=np.array(values,float);m=float(v.mean());h=float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else float('nan')
    return m,h
def fmt(values):
    m,h=interval(values);return f'{m:.3g} ± {h:.3g}'
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])

def main(c):
 with recorded(c) as run:
    root=Path(c['output']);suite=json.loads((root/'evaluation_complete.json').read_text())['runs'];stats=[];cp=[]
    cases=['main','augmented','b3_multi','B0','B3','B4']+[f'{case}_{n}' for case in ['transfer','scratch_b3'] for n in [100,1000,10000]]
    for case in cases:
        for s in c['seeds']:
            p=Path(suite[f'{case}_{s}']);m=json.loads((p/'metrics.json').read_text())
            if m.get('status')!='completed':raise ValueError(f'Final evaluation incomplete: {p}')
            stats+=json.loads((p/'statistics.json').read_text());cp+=json.loads((p/'conformal.json').read_text())
    df=pd.DataFrame(stats);cf=pd.DataFrame(cp);df.to_csv('reports/M4_metrics.csv',index=False);cf.to_csv('reports/M4_conformal.csv',index=False)
    primary=df[(df.kind=='clean')&(~df.heldout)&(df.subset=='all')]
    primarycp=cf[(~cf.heldout)&(cf.subset=='all')]
    def vals(case,split,q,metric,cal='temperature'):
        a=primary[(primary.case==case)&(primary.split==split)&(primary.qtype==q)&(primary.calibration==cal)]
        return a.groupby('seed')[metric].mean().reindex(c['seeds']).to_numpy()
    def cpvals(case,split,q,metric,cal='temperature',alpha=.1):
        a=primarycp[(primarycp.case==case)&(primarycp.split==split)&(primarycp.qtype==q)&(primarycp.calibration==cal)&(primarycp.alpha==alpha)]
        return a.groupby('seed')[metric].mean().reindex(c['seeds']).to_numpy()
    # Matched (problem,t) paired bootstrap: average duplicate label/template instances
    # within a problem and step, then average steps equally as in primary tables.
    rng=np.random.default_rng(20261103);paired=[]
    def per_problem(case,s,split,q,metric):
        p=Path(suite[f'{case}_{s}']);r=pd.read_parquet(p/f'{q}_{split}_False.parquet');r=r[r.kind=='clean'].reset_index(drop=True)
        cal=json.loads((p/'calibration.json').read_text())[q]['temperature'];prob=softmax(apply(np.stack(r.logits),cal),axis=1)
        if metric=='accuracy':
            pred=prob.argmax(1);r['value']=(pred==r.label).astype(float) if q=='Q3' else np.where(pred==1,r.label_soft,1-r.label_soft)
        elif metric=='nll':r['value']=-(r.label_soft*np.log(np.maximum(prob[:,1],1e-12))+(1-r.label_soft)*np.log(np.maximum(prob[:,0],1e-12)))
        else:raise ValueError(metric)
        return r.groupby(['problem_id','t']).value.mean()
    comparisons=[('main','B3','Q3','accuracy'),('main','B3','Q4','accuracy'),('main','B3','Q1','nll'),('main','b3_multi','Q3','accuracy'),('main','b3_multi','Q4','accuracy')]
    comparisons += [(f'transfer_{n}',f'scratch_b3_{n}','Q4','accuracy') for n in [100,1000,10000]]
    for split in ['reader_val','test']:
        for a,b,q,metric in comparisons:
            diffs=[]
            for s in c['seeds']:
                x=per_problem(a,s,split,q,metric);y=per_problem(b,s,split,q,metric)
                if not x.index.equals(y.index):raise ValueError('Paired evaluation keys differ')
                diffs.append(x-y)
            delta=pd.concat(diffs,axis=1).mean(axis=1).unstack('t');matrix=delta.to_numpy();n=len(matrix);boot=[]
            for start in range(0,10000,100):
                ii=rng.integers(n,size=(min(100,10000-start),n));sample=matrix[ii]
                boot.extend(np.nanmean(np.nanmean(sample,axis=1),axis=1).tolist())
            lo,hi=np.quantile(boot,[.025,.975]);paired.append({'split':split,'a':a,'b':b,'qtype':q,'metric':metric,'delta':float(np.nanmean(np.nanmean(matrix,axis=0))),'ci_low':float(lo),'ci_high':float(hi),'bootstrap':10000,'unit':'problem cluster; all steps together; mean of three seeds'})
    pd.DataFrame(paired).to_csv('reports/M4_paired_bootstrap.csv',index=False)
    def difftext(a,b,q,split='test'):
        d=next(x for x in paired if (x['a'],x['b'],x['qtype'],x['split'])==(a,b,q,split));return f"{d['delta']:.3g} [{d['ci_low']:.3g}, {d['ci_high']:.3g}]"
    # Figures: retained standalone publication-style artifacts, no dashboard dependency.
    figs=Path('reports/M4_figures');figs.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,4))
    for case,label in [('transfer','CLR transfer'),('scratch_b3','B3 scratch')]:
        points=[interval(vals(f'{case}_{n}','test','Q4','accuracy')) for n in [100,1000,10000]]
        ax.errorbar([100,1000,10000],[p[0] for p in points],yerr=[p[1] for p in points],marker='o',label=label)
    ax.set(xscale='log',xlabel='Q4 training labels',ylabel='Test Q4 accuracy');ax.legend();fig.tight_layout();fig.savefig(figs/'q4_transfer.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,3))
    for ax,q in zip(axes,['Q1','Q2','Q3','Q4']):
        for case in ['main','B3','b3_multi']:
            coverage=[np.mean(cpvals(case,'test',q,'coverage',alpha=a)) for a in [.2,.1,.05]];sizes=[np.mean(cpvals(case,'test',q,'size',alpha=a)) for a in [.2,.1,.05]]
            ax.plot(coverage,sizes,'o-',label=case)
        ax.set(title=q,xlabel='Empirical coverage',ylabel='Mean set size')
    axes[0].legend();fig.tight_layout();fig.savefig(figs/'conformal_efficiency.png',dpi=180);plt.close(fig)
    # Aggregate reports retain all per-t / heldout / state-kind rows in CSV.
    rows=[]
    for case in ['B0','B4','B3','b3_multi','main']:
        for split in ['reader_val','test']:
            metrics=[fmt(vals(case,split,q,METRICS[q])) if not(case=='B4' and q in ['Q3','Q4']) else '—' for q in METRICS]
            gain='—' if case=='B4' else fmt(vals(case,split,'Q3','accuracy')-vals('B0',split,'Q3','accuracy'))
            rows.append([case,split,*metrics,gain])
    tab1=table(['model','split','Q1 NLL','Q2 AUROC','Q3 acc','Q4 acc','Q3−B0'],rows)
    rows=[]
    for q in METRICS:
        rows.append([q,*[f"{np.mean(cpvals(case,split,q,'coverage')):.3g} / {np.mean(cpvals(case,split,q,'size')):.3g}" for split in ['reader_val','test'] for case in ['B3','main']]])
    tab2=table(['qtype','val B3 cov/size','val CLR cov/size','test B3 cov/size','test CLR cov/size'],rows)
    tab3=table(['N','val CLR','val B3','test CLR','test B3'],[[n,*[fmt(vals(f'{case}_{n}',split,'Q4','accuracy')) for split in ['reader_val','test'] for case in ['transfer','scratch_b3']]] for n in [100,1000,10000]])
    affine=[]
    for case in ['main','B3']:
        for split in ['reader_val','test']:
            for q in METRICS:
                affine.append({'case':case,'split':split,'qtype':q,**{m:float(np.mean(vals(case,split,q,m)-vals(case,split,q,m,'affine'))) for m in ['nll','ece15']}})
    pd.DataFrame(affine).to_csv('reports/M4_affine_gains.csv',index=False)
    gaps=[]
    for case in ['main','augmented']:
        for q,m in METRICS.items():gaps.append({'case':case,'qtype':q,'metric':m,'reader_val':float(np.mean(vals(case,'reader_val',q,m))),'test':float(np.mean(vals(case,'test',q,m))),'gap_val_minus_test':float(np.mean(vals(case,'reader_val',q,m)-vals(case,'test',q,m)))})
    pd.DataFrame(gaps).to_csv('reports/M4_shift_gaps.csv',index=False)
    cost=hours_used();dmetrics=json.loads(Path(c['source_run'],'metrics.json').read_text());qsummary=json.loads((root/'qa/summary.json').read_text())
    details=['# M4 CLR v0：完整结果','仅CODI-GPT2。主表clean状态、seen模板、逐t等权平均；reader_val另有traj结果、held-out模板、逐t指标和仿射对照，全部保存在M4_metrics.csv。训练三seed11/22/33；±为三seed t区间（df2）；配对差异为按题10000次bootstrap，合并三seed均值。没有预设非劣界，不把不显著当作已证明不劣。',
    '## 主结果',tab1,'## Mondrian α=0.1',tab2,'所有α=.05/.1/.2及两种校准见M4_conformal.csv。Q1预测未来Bernoulli正误而非V本身；校准每题固定抽一个MC结果，评估覆盖率对K16结果平均。每(qtype,t)每题仅一实例。依赖层内可交换性；cohort padding、Q4跨题匹配、选项干扰池造成依赖，报告经验覆盖率而非无条件理论保证。未见池有分布偏移，不主张GSM阈值保覆盖。',
    '## 小样本Q4迁移',tab3,'![Q4 transfer](M4_figures/q4_transfer.png)','![CP](M4_figures/conformal_efficiency.png)',
    '## 架构与协议','官方ModernBERT-base固定revision8949b909ec900327062f0ebf497f51aef5e6f0c8，LFS SHA-256核验。22层双向encoder，冻结词嵌入、训练其余主体，2层768维adapter+步位置+source token；隔k层cross-attention读观测前缀。模板和选项只见祖先与自身，父块不读子块，兄弟共享逻辑位置；8维数值特征与B3相同。公开接口先规范排序再恢复选项顺序，另有不依赖排序的mask隔离测试。Q4从值选项标记读yes-no，Q1/Q2从CLS读；不输入原题、不读feedback噪声向量。',
    '来源：[ModernBERT官方](https://huggingface.co/answerdotai/ModernBERT-base)、[NDM原文](https://arxiv.org/html/2610.25553)。采用ModernBERT是因本地Transformers支持自定义双向SDPA和逻辑RoPE位置，适合上述块掩码；没有更换到小模型。训练设置、吞吐与预算见M4_budget.md；k选择和完整检查点锁见data/M4_v0/hyperparameters_lock.json及training_complete.json。',
    '## 校准','按qtype在cal_A拟合温度，以及温度+二分类log-odds偏置。Q3共享选项偏置被softmax抵消，仿射等价温度；不引入位置偏置。B3完全相同校准流程。改善量见M4_affine_gains.csv；纯温度为主方案，未据test选择校准。',
    '## D数据与限制',json.dumps(qsummary,ensure_ascii=False),
    'SVAMP993按ID496/497、ASDiv1000按ID500/500冻结。复用M3干净/早退，train K8轨迹、eval每t K16π；原生产fp32/batch1/cohort上下文保持。原始方程式259题解析失败，仅跳过这些题Q4，其他类型保留；有解析成功题的池保留Q4。两个新半池之间172题13-gram重叠（精确0），保留用户指定划分并另报825无重叠eval子集；不把这172题称独立题族泛化证据。与GSM训练/测试/Aug去重复用M3证据。',
    '源成本：'+json.dumps(dmetrics,ensure_ascii=False),'总M4 GPU作业墙钟小时（含加载/失败/smoke）='+str(cost)+'；GPU RTX4090，训练评估新source forward=0，D调用单列。CPU报告不记GPU。预算内优先四项必做；可选无cross、单步、无块、随机初始化实验未做，不以未执行消融支撑结论。',
    '## 证据索引','runs见data/M4_v0/evaluation_suite.json、training_suite.json；逐实例预测在各评估run；M4_paired_bootstrap.csv、M4_shift_gaps.csv；冻结清单data/M4_v0/freeze.json。']
    Path('reports/M4_clr_v0.md').write_text('\n\n'.join(details))
    # G1 statements are estimates with uncertainty; no automatic claim of noninferiority.
    g1='(1) CLR−B3：Q3 '+difftext('main','B3','Q3')+'；Q4 '+difftext('main','B3','Q4')+'。Q3扣相同B0后差值不变。\n'
    g1+='(2) 实际覆盖率与集合大小见表2；仅在覆盖率相当且不低于目标时可称效率优势，完整α曲线见详细报告。\n'
    g1+='(3) CLR−B3 Q1 NLL '+difftext('main','B3','Q1')+'；CLR−B3-multi Q3 '+difftext('main','b3_multi','Q3')+'、Q4 '+difftext('main','b3_multi','Q4')+'；未预设非劣界，不能据不显著声称不劣。\n'
    g1+='(4) 迁移−scratch Q4：N100 '+difftext('transfer_100','scratch_b3_100','Q4')+'；N1k '+difftext('transfer_1000','scratch_b3_1000','Q4')+'；N10k '+difftext('transfer_10000','scratch_b3_10000','Q4')+'。'
    afftext='；'.join(f"{case} test平均NLL/ECE改善 {np.mean([x['nll'] for x in affine if x['case']==case and x['split']=='test']):.3g}/{np.mean([x['ece15'] for x in affine if x['case']==case and x['split']=='test']):.3g}" for case in ['main','B3'])
    gaptext='；'.join(f"{q} val−test落差 {next(x['gap_val_minus_test'] for x in gaps if x['case']=='main' and x['qtype']==q):.3g}→{next(x['gap_val_minus_test'] for x in gaps if x['case']=='augmented' and x['qtype']==q):.3g}" for q in METRICS)
    unseen=[]
    for case in ['main','B3','augmented']:
        for q in METRICS:unseen.append(f"{case}/{q} {np.mean(cpvals(case,'unseen_eval',q,'coverage')):.3g}")
    review=['## (a) 状态','M4实验完成，待G1审查；不能把完成等同优势关卡通过。','## (b) 关卡',
    f'通过：val-only smoke、块隔离/置换与输入约束测试；三seed必做模型/迁移/增强训练已完成并在最终评估前锁定；温度/仿射及Mondrian均已执行。总成本{cost:.3g} GPUh，D {dmetrics.get("gpu_hours",float("nan")):.3g} GPUh。G1优势结论按下列差值区间审查，尚未标为通过。',g1,
    '## (c) 核心主表','均值±95% seed-t CI；clean/seen，逐t等权。表2为α=.1的三seed均值；详细CI与其他α见CSV。',tab1,tab2,tab3,
    '仿射−温度的改善（正为更好）：'+afftext+'。','加入未见训练后：'+gaptext+'。','GSM cal_B阈值下unseen_eval覆盖率（α=.1）：'+'；'.join(unseen)+'。',
    '## (d) 新增偏离与阻塞','预算固定4epoch、每qtype每epoch10000采样，仅第2/4epoch验证；可能欠训练。可选消融未做。Q3共享仿射偏置抵消。未见池259题方程解析失败跳过Q4；172条eval与新train存在13-gram重叠、精确0，另报825无重叠子集。调试出现控制器状态误判及无限阈值JSON错误，均在val修复，失败记录保留。',
    '## (e) 需要审查者决定','根据G1差值及覆盖率判断是否通过进入M5；是否另建按题族分组的未见池版本（当前冻结划分不改）。',
    '## (f) 证据路径','详细：reports/M4_clr_v0.md、reports/M4_budget.md；CSV：reports/M4_metrics.csv、M4_conformal.csv、M4_paired_bootstrap.csv、M4_affine_gains.csv、M4_shift_gaps.csv；run索引：data/M4_v0/training_suite.json、evaluation_suite.json；数据与代码冻结：data/M4_v0/freeze.json。']
    text='\n\n'.join(review)+'\n'
    if len(text.encode())>8000:raise ValueError(f'Review exceeds8KB: {len(text.encode())}; shorten without dropping required evidence')
    Path('reports/M4_REVIEW.md').write_text(text)
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,review_bytes=len(text.encode()),total_m4_gpu_hours=cost)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

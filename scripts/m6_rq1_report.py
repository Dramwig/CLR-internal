"""RQ1 closure from the fixed small-sigma run and previously frozen analyses."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch  # Initialize the tested C++ runtime before Pillow/Matplotlib.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from scripts.m6_shift import bootstrap_matrix
from clr.utils.config import load_config
from clr.utils.m6_reports import markdown
from clr.utils.records import recorded
from clr.utils.m6_runtime import snapshot
from clr.utils.idea_a import write_json

def main(c):
 with recorded(c) as run:
    snapshot(run,{**c,'section':c.get('section','CD')})
    print(run.path,flush=True)
    start=time.monotonic();source=Path(c['search_run']);m=json.loads((source/'metrics.json').read_text())
    if m.get('status')!='completed':raise ValueError('Complete three-seed RQ1 run required')
    out=Path(c['output'])/'rq1';out.mkdir(parents=True,exist_ok=True)
    df=pd.concat([pq.read_table(p).to_pandas() for p in sorted(source.glob('pred_*.parquet'))]);df.to_csv(out/'problem_results.csv',index=False)
    if set(df.seed)!={11,22,33} or df.problem_id.nunique()!=300:raise ValueError('Incomplete RQ1 val300/three seeds')
    matrices=[];keys=[];result=[]
    for key,group in df.groupby(['sigma','value']):
        d=group.assign(delta=group.correct.astype(float)-group.greedy_correct.astype(float)).groupby('problem_id')[['correct','greedy_correct','pass_at_8','delta']].mean()
        keys.append(key);matrices.append(d['delta'].values);ci=bootstrap_matrix(d.values)
        result.append({'sigma':key[0],'value':key[1],**{col:float(d[col].mean()) for col in d},'delta_ci_low':float(ci[0,3]),'delta_ci_high':float(ci[1,3]),'pass8_ci_low':float(ci[0,2]),'pass8_ci_high':float(ci[1,2])})
    # Paired max-standardized bootstrap intervals control this final four-case family.
    x=np.column_stack(matrices);means=x.mean(0);rng=np.random.default_rng(11);boot=[]
    for first in range(0,10000,500):boot.append(x[rng.integers(len(x),size=(500,len(x)))].mean(1))
    boot=np.concatenate(boot);se=boot.std(0).clip(1e-12);critical=np.quantile(np.max(abs((boot-means)/se),axis=1),.95)
    for r,mu,s in zip(result,means,se):r['simultaneous_low']=float(mu-critical*s);r['simultaneous_high']=float(mu+critical*s)
    pd.DataFrame(result).to_csv(out/'summary.csv',index=False)
    old=pd.read_csv('data/M5b_v1/analysis/B_summary.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for name in ['B3','CLR-lite']:
        a=old[(old.value==name)&(old.metric=='between_auroc')&(old.t.astype(str)!='all')].copy();b=old[(old.value==name)&(old.metric=='spearman')&(old.t.astype(str)!='all')].copy()
        a['t']=a.t.astype(int);b['t']=b.t.astype(int);v=a.merge(b,on='t',suffixes=('_auc','_rho')).sort_values('t')
        axes[0].plot(v['t'],v.mean_auc,marker='o',label=name);axes[1].errorbar(v['t'],v.mean_rho,yerr=[v.mean_rho-v.ci_low_rho,v.ci_high_rho-v.mean_rho],marker='o',label=name)
    axes[0].set_ylabel('Between-problem AUROC');axes[1].set_ylabel('Within-parent Spearman rho')
    for ax in axes:ax.set_xlabel('Observed step t');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'between_within_by_t.pdf');fig.savefig(out/'between_within_by_t.png',dpi=160);plt.close(fig)
    historical=pd.read_csv('data/M5b_v1/analysis/C_summary.csv');p=historical[(historical.method=='S2')&(historical.value=='pass@N')&(historical.horizon==6)].sort_values('N')
    fig,ax=plt.subplots(figsize=(5,3.6));ax.errorbar(p.N,p.accuracy,yerr=[p.accuracy-p.accuracy_ci_low,p.accuracy_ci_high-p.accuracy],marker='o');ax.set(xlabel='N (one clean candidate included)',ylabel='pass@N, val300',xticks=p.N);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'pass_at_N.pdf');fig.savefig(out/'pass_at_N.png',dpi=160);plt.close(fig)
    # Existing early-exit artifact is copied byte-for-byte; no new source/model calls.
    shutil.copyfile('reports/M3_figures/unseen_early_exit.png',out/'historical_early_exit.png')
    positive=any(r['simultaneous_low']>0 for r in result)
    def f(x):return f'{x:.3g}'
    lines=['# M6 RQ1 收尾','val300、seed11/22/33；sigma0候选纳入S3每个父节点的8个子节点，b4，全6步；冻结sigma1.5下训练的CLR-lite/B3。相同噪声种子按父索引配对。pass@8为独立候选池（1clean+7pi轨迹）的特权上限，不等于beam终点候选数。',
      '|sigma|value|accuracy|Δgreedy pp (95% simultaneous CI)|pass@8|','|---|---|---|---|---|']
    for r in result:lines.append('|'+ '|'.join([f(r['sigma']),r['value'],f(r['correct']),f(100*r['delta'])+' ['+f(100*r['simultaneous_low'])+', '+f(100*r['simultaneous_high'])+']',f(r['pass_at_8'])])+'|')
    lines+=['较小sigma'+('在预设四项比较中出现相对greedy的正区间下界；因此须修订RQ1负面结论，范围限本设置。' if positive else '未在预设四项比较中建立相对greedy的正收益；RQ1保留为负面/分析性结论。无显著正收益不等同证明所有小sigma搜索必定无效。'),
       'CI按题cluster bootstrap10000次，平均三个固定训练seed，四设置用max标准化同时区间；不把rollout/步骤当独立题。每seed值和普通配对区间见CSV。所有新增计算只用val；完整成本在M6成本台账。',
       '## 既有分析图（无新增模型计算）','![between vs within](../data/M6_v1/rq1/between_within_by_t.png)','来源：data/M5b_v1/analysis/B_summary.csv。跨题AUROC与同题同父候选的rho衡量不同排序任务，不能以高跨题AUROC推断搜索有效。',
       '![pass@N](../data/M6_v1/rq1/pass_at_N.png)','来源：冻结C_summary.csv；仅使用既有N4/8/16完整6步池，不外推更大N。',
       '![early exit](../data/M6_v1/rq1/historical_early_exit.png)','早退图逐字节复用reports/M3_figures/unseen_early_exit.png；其原池/旧协议限制沿用reports/M3_unseen_pools.md，不将之冒充M6新增LLaMA结果。',
       '证据：'+str(source)+'；data/M6_v1/rq1/problem_results.csv、summary.csv；configs/M6_rq1_full.yaml。']
    Path('reports/M6_rq1_closure.md').write_text(markdown(lines));run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start,rq1_positive=positive)
    write_json(out/'completion.json',{'run':str(run.path),'search_run':str(source),'positive_simultaneous_lower_bound':positive})
    print(run.path,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

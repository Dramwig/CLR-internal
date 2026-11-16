"""M6 final statistics, evidence index and the only reviewer-facing artifact.

Inference/settings must already be locked. Bootstrap units are problems after
averaging the three fixed seeds; no trajectory or step pseudo-replication.
"""
import argparse,hashlib,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import softmax
import torch  # Initialize the tested C++ runtime before Pillow/Matplotlib.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.m6_shift import bootstrap_matrix
from clr.calib.m4_evaluation import target_array
from clr.calib.m23 import metrics
from clr.utils.config import load_config
from clr.utils.m6_reports import markdown
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import hours,snapshot,live


def f(x):
    return 'NA' if x is None or not np.isfinite(x) else f'{x:.3g}'


def estimate(values):
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    ci=bootstrap_matrix(values)
    return values.mean(0),ci


def cluster_estimate(frame,columns):
    """Frozen M4.1 estimand: micro within step, macro across steps/seeds.

    Resample problem clusters, keeping their step-specific QA counts. This
    avoids changing accuracy when different problems have different row counts.
    """
    g=frame.groupby(['problem_id','t'])[[*columns,'instances']].mean()
    ids=sorted(g.index.get_level_values(0).unique());steps=sorted(g.index.get_level_values(1).unique())
    index=pd.MultiIndex.from_product([ids,steps],names=['problem_id','t']);g=g.reindex(index)
    x=g[columns].to_numpy().reshape(len(ids),len(steps),len(columns));w=g.instances.fillna(0).to_numpy().reshape(len(ids),len(steps),1)*np.isfinite(x)
    numerator=np.nan_to_num(x)*w
    def finish(n,d):
        v=np.divide(n,d,out=np.full_like(n,np.nan),where=d>0)
        return np.divide(np.nansum(v,axis=-2),np.isfinite(v).sum(axis=-2),out=np.full(v.shape[:-2]+(v.shape[-1],),np.nan),where=np.isfinite(v).sum(axis=-2)>0)
    point=finish(numerator.sum(0),w.sum(0));rng=np.random.default_rng(11);boot=[]
    for first in range(0,10000,500):
        ii=rng.integers(len(ids),size=(500,len(ids)));boot.append(finish(numerator[ii].sum(1),w[ii].sum(1)))
    return point,np.quantile(np.concatenate(boot),[.025,.975],axis=0)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def optional_unseen_cost(c):
    """Validate the cost-only D decision made after C3 and before C2/C4."""
    path=Path(c['output'])/'llama_unseen_budget.json'
    decision=json.loads(path.read_text())
    if decision['status']=='completed':
        data=decision['data']
        if json.loads(Path(data['audit_run'],'metrics.json').read_text()).get('status')!='completed':
            raise ValueError('Optional LLaMA unseen audit is incomplete')
        for split,count in decision['problem_counts'].items():
            completed=json.loads(Path(data['root'],split,'completed.json').read_text())
            if completed['problem_count']!=count:raise ValueError('Optional unseen count differs from admission')
        return decision
    if decision['status']!='not_admitted_insufficient_CD_budget':
        raise ValueError('Optional LLaMA unseen decision remains unfinished')
    blockers=Path('docs/BLOCKERS.md');marker='## M6 — LLaMA unseen 成本准入未通过'
    if marker not in blockers.read_text():
        with blockers.open('a') as stream:
            stream.write('\n'+marker+'\nC3后、C2/C4前：完整'+str(decision['problem_count'])+'题生成及六个C1 reader推断估计'+f(decision['estimated_additional_gpu_hours'])+'GPUh，CD剩余'+f(decision['remaining_CD_gpu_hours'])+'GPUh，已承诺核心推断预留'+f(decision['existing_inference_reserved_gpu_hours'])+'GPUh，另留'+f(decision['guard_gpu_hours'])+'GPUh；未执行。\n仅用登记计时/冻结规模，非结果选择；精确分解data/M6_v1/llama_unseen_budget.json。\n')
    return decision


def point_rows(pred,q):
    p=softmax(np.asarray([r['calibrated_logits'] for r in pred]),axis=1);y=target_array(pred,q)
    good=y[np.arange(len(p)),p.argmax(1)]
    brier=(1-2*(p*y).sum(1)+(p*p).sum(1))/(1 if q=='Q3' else 2)
    nll=-(y*np.log(p.clip(1e-12))).sum(1)
    return pd.DataFrame({'problem_id':[r['problem_id'] for r in pred],'t':[r['state_ref']['t'] for r in pred],
       'accuracy':good,'brier':brier,'nll':nll,
       'mae':np.abs(p[:,1]-y[:,1]) if q!='Q3' else np.nan})


def collect(c,lock,index,out):
    points=[];seed_stats=[]
    # Four qtypes are read one at a time; source text never enters a model.
    for key,by_source in index.items():
        cfg=lock['models'][key]['config'];sd=cfg['seed']
        for source,run in by_source.items():
            for split in ['reader_val','test']:
                for q in ['Q1','Q2','Q3','Q4']:
                    path=Path(run)/f'{source}_{q}_{split}.parquet'
                    pred=pq.read_table(path).to_pylist()
                    for kind in (['clean','traj','all'] if q=='Q1' and split=='reader_val' else ['clean']):
                        rs=pred if kind=='all' else [r for r in pred if r['state_ref']['kind']==kind]
                        if not rs:continue
                        per_step=[]
                        for t in sorted({r['state_ref']['t'] for r in rs}):
                            tt=[r for r in rs if r['state_ref']['t']==t];xx=np.asarray([r['calibrated_logits'] for r in tt]);per_step.append(metrics(xx,target_array(tt,q),q=='Q3'))
                        names=set().union(*(v.keys() for v in per_step));m={name:float(np.mean([v[name] for v in per_step if v.get(name) is not None])) if any(v.get(name) is not None for v in per_step) else None for name in names}
                        common={'model_key':key,'source':source,'split':split,'qtype':q,'kind':kind,
                           'seed':sd,'family':cfg['kind'],'donor':cfg.get('donor',''),
                           'n_labels':str(cfg.get('n_labels') or 'full')}
                        seed_stats.append({**common,**m,'instances':len(rs)})
                        values=point_rows(rs,q);frame=values.groupby(['problem_id','t'],sort=True)[['accuracy','brier','nll','mae']].mean();frame['instances']=values.groupby(['problem_id','t']).size()
                        for (pid,t),v in frame.iterrows():points.append({**common,'problem_id':pid,'t':t,**v.to_dict()})
            print('statistics',key,source,flush=True)
    frame=pd.DataFrame(points);pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),out/'problem_metrics.parquet',compression='zstd')
    pd.DataFrame(seed_stats).to_csv(out/'seed_metrics.csv',index=False)
    aggregate=[]
    # Strip the seed suffix only; target/donor/N remain explicit.
    frame['case']=frame.model_key.str.replace(r'_(11|22|33)$','',regex=True)
    for key,g in frame.groupby(['case','source','split','qtype','kind','family','donor','n_labels'],dropna=False,sort=False):
        if set(g.seed)!={11,22,33}:raise ValueError('Missing training seed in final case '+str(key))
        columns=['accuracy','brier','nll','mae'];mu,ci=cluster_estimate(g,columns)
        common=dict(zip(['case','source','split','qtype','kind','family','donor','n_labels'],key));common['problems']=g.problem_id.nunique()
        for j,metric in enumerate(columns):aggregate.append({**common,'metric':metric,'mean':mu[j],'ci_low':ci[0,j],'ci_high':ci[1,j]})
    agg=pd.DataFrame(aggregate);agg.to_csv(out/'metrics.csv',index=False)
    paired=[]
    for source in ['gpt2','llama']:
      for split in ['reader_val','test']:
       for q in ['Q1','Q2','Q3','Q4']:
        a=frame[(frame.case==f'single_{source}_lite')&(frame.source==source)&(frame.split==split)&(frame.qtype==q)&(frame.kind=='clean')]
        b=frame[(frame.case==f'single_{source}_b3')&(frame.source==source)&(frame.split==split)&(frame.qtype==q)&(frame.kind=='clean')]
        d=a.merge(b,on=['seed','problem_id','t'],suffixes=('_lite','_b3'),validate='one_to_one')
        for metric in ['accuracy','brier','nll']:
            if not d.instances_lite.equals(d.instances_b3):raise ValueError('Paired QA count mismatch')
            dd=d[['problem_id','t']].copy();dd['instances']=d.instances_lite;dd['delta']=d[metric+'_lite']-d[metric+'_b3'];mu,ci=cluster_estimate(dd,['delta'])
            paired.append({'source':source,'split':split,'qtype':q,'metric':metric,'delta_lite_minus_b3':mu[0],'ci_low':ci[0,0],'ci_high':ci[1,0],'problems':dd.problem_id.nunique()})
    pd.DataFrame(paired).to_csv(out/'single_source_paired.csv',index=False)
    joint=[]
    for source in ['gpt2','llama']:
      for split in ['reader_val','test']:
       for q in ['Q1','Q2','Q3','Q4']:
        a=frame[(frame.case=='multi_lite')&(frame.source==source)&(frame.split==split)&(frame.qtype==q)&(frame.kind=='clean')]
        if a.empty:continue
        b=frame[(frame.case==f'single_{source}_lite')&(frame.source==source)&(frame.split==split)&(frame.qtype==q)&(frame.kind=='clean')]
        d=a.merge(b,on=['seed','problem_id','t'],suffixes=('_joint','_single'),validate='one_to_one')
        for metric in ['accuracy','brier','nll']:
            v=d[['problem_id','t']].copy();v['instances']=d.instances_joint;v['delta']=d[metric+'_joint']-d[metric+'_single'];mu,ci=cluster_estimate(v,['delta'])
            joint.append({'source':source,'split':split,'qtype':q,'metric':metric,'delta_joint_minus_single':mu[0],'ci_low':ci[0,0],'ci_high':ci[1,0],'problems':v.problem_id.nunique()})
    pd.DataFrame(joint).to_csv(out/'multisource_paired.csv',index=False)
    # Transfer vs each scratch competitor uses identical target label panels.
    transfer=[]
    for a,b in [('gpt2','llama'),('llama','gpt2')]:
      for n in [100,300,1000,3000,'full']:
       tc=f'transfer_{a}_to_{b}_transfer_{n}'
       for baseline in ['b3','lite','b5b']:
        bc=(f'single_{b}_{baseline}' if n=='full' else f'transfer_{a}_to_{b}_{baseline}_{n}') if baseline!='b5b' else f'b5b_{b}_{n}'
        for split in ['reader_val','test']:
         for q in ['Q1','Q2','Q3','Q4']:
          aa=frame[(frame.case==tc)&(frame.split==split)&(frame.qtype==q)&(frame.kind=='clean')]
          bb=frame[(frame.case==bc)&(frame.split==split)&(frame.qtype==q)&(frame.kind=='clean')]
          if aa.empty or bb.empty:continue
          d=aa.merge(bb,on=['seed','problem_id','t'],suffixes=('_transfer','_scratch'),validate='one_to_one')
          metric='brier' if q=='Q1' else 'accuracy';v=d[['problem_id','t']].copy();v['instances']=d.instances_transfer;v['delta']=d[metric+'_transfer']-d[metric+'_scratch'];mu,ci=cluster_estimate(v,['delta'])
          transfer.append({'direction':a+'->'+b,'n_labels':n,'baseline':baseline,'split':split,'qtype':q,'metric':metric,'delta_transfer_minus_scratch':mu[0],'ci_low':ci[0,0],'ci_high':ci[1,0],'problems':v.problem_id.nunique()})
    transfer=pd.DataFrame(transfer,columns=['direction','n_labels','baseline','split','qtype','metric','delta_transfer_minus_scratch','ci_low','ci_high','problems'])
    transfer.to_csv(out/'transfer_paired.csv',index=False)
    return frame,agg,pd.DataFrame(paired),transfer


def same_problem(c,index,out):
    allrows=[]
    for split in ['reader_val','test']:
        correct={}
        for s,spec in c['sources'].items():
            correct[s]={r['problem_id']:bool(r['correct']) for r in pq.read_table(Path(spec['root'])/split/'manifest.parquet',columns=['problem_id','correct'],filters=[('kind','=','clean'),('t','=',6)]).to_pylist()}
        for sd in [11,22,33]:
            preds={}
            for s in ['gpt2','llama']:
                run=index[f'single_{s}_lite_{sd}'][s];rs=pq.read_table(Path(run)/f'{s}_Q3_{split}.parquet').to_pylist()
                preds[s]={}
                for r in rs:
                    if r['state_ref']['traj_id']!=-1:continue
                    ix=int(np.argmax(r['calibrated_logits']));key=(r['problem_id'],r['state_ref']['t'])
                    preds[s][key]=(r['options'][ix],int(ix==r['label']))
            for key in sorted(preds['gpt2'].keys()&preds['llama'].keys()):
                pid,t=key;num=int(correct['gpt2'][pid])+int(correct['llama'][pid]);group=['both_wrong','one_correct','both_correct'][num]
                a,b=preds['gpt2'][key],preds['llama'][key]
                allrows.append({'problem_id':pid,'t':t,'seed':sd,'split':split,'group':group,'prediction_agrees':float(a[0]==b[0]),'gpt2_accuracy':float(a[1]),'llama_accuracy':float(b[1])})
    frame=pd.DataFrame(allrows);pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),out/'same_problem.parquet',compression='zstd');summary=[]
    for (split,group),g in frame.groupby(['split','group']):
        means=g.groupby('problem_id')[['prediction_agrees','gpt2_accuracy','llama_accuracy']].mean();mu,ci=estimate(means.values)
        for j,name in enumerate(means):summary.append({'split':split,'group':group,'metric':name,'mean':mu[j],'ci_low':ci[0,j],'ci_high':ci[1,j],'problems':len(means)})
    pd.DataFrame(summary).to_csv(out/'same_problem_summary.csv',index=False)


def curves(agg,out,c):
    full_counts={source:pq.read_table(Path(spec['qa'])/'reader_train.parquet',columns=['is_heldout_template'],filters=[('is_heldout_template','=',False)]).num_rows for source,spec in c['sources'].items()}
    for split in ['reader_val','test']:
      fig,axes=plt.subplots(2,2,figsize=(11,7))
      for row,(a,b) in enumerate([('gpt2','llama'),('llama','gpt2')]):
       for col,q in enumerate(['Q3','Q4']):
        ax=axes[row,col]
        for family,label in [('transfer','frozen-body CLR'),('lite','scratch CLR'),('b3','scratch B3'),('b5b','frozen Qwen mapping')]:
            data=agg[(agg.source==b)&(agg.split==split)&(agg.qtype==q)&(agg.kind=='clean')&(agg.metric=='accuracy')&(agg.family==family)]
            if family=='transfer':data=data[data.donor==a]
            elif family in ['lite','b3']:data=data[(data.donor==a)|(data['case']==f'single_{b}_{family}')]
            if data.empty:continue
            data=data.copy();data['labels']=data.n_labels.map(lambda n:full_counts[b] if n=='full' else int(n));data=data.sort_values('labels')
            ax.errorbar(data['labels'],data['mean'],yerr=[data['mean']-data.ci_low,data.ci_high-data['mean']],marker='o',label=label)
        ax.set(xscale='log',xlabel='QA label instances N',ylabel=q+' accuracy',title=a+' → '+b+' / '+split)
        ax.set_xticks([100,300,1000,3000,full_counts[b]],['100','300','1k','3k','full']);ax.grid(alpha=.2);ax.legend(fontsize=7)
      fig.tight_layout();fig.savefig(out/f'learning_curves_{split}.pdf');fig.savefig(out/f'learning_curves_{split}.png',dpi=160)
      if split=='test':fig.savefig(out/'learning_curves.pdf');fig.savefig(out/'learning_curves.png',dpi=160)
      plt.close(fig)


def write_reports(c,lock,index,out,agg,paired,transfer,costs):
    shift=pd.read_csv(Path(c['output'])/'shift'/'summary.csv');rq=pd.read_csv(Path(c['output'])/'rq1'/'summary.csv')
    shift_points=pq.read_table(Path(c['output'])/'shift'/'problem_results.parquet',columns=['model','qtype','shift','method','alpha','problem_id','coverage'],filters=[('alpha','=',.1)]).to_pandas()
    costs_by={s:sum(x['gpu_hours'] for x in costs if x['section']==s) for s in ['A','B','CD']};total=sum(costs_by.values())
    sigma=json.loads((Path(c['output'])/'plan.json').read_text());selection=sigma.get('noise_selection',{}).get('selected')
    unseen_budget=json.loads(Path(c['output'],'llama_unseen_budget.json').read_text())
    unseen_note=('LLaMA unseen已补：完整'+str(unseen_budget['problem_count'])+'题；源数据和审计见data/M6_v1/llama_unseen/completion.json。' if unseen_budget.get('status')=='completed' else 'LLaMA unseen未补：C3后CD剩余'+f(unseen_budget['remaining_CD_gpu_hours'])+'GPUh，完整补充估计'+f(unseen_budget['estimated_additional_gpu_hours'])+'GPUh，核心推断另预留'+f(unseen_budget.get('existing_inference_reserved_gpu_hours',0))+'GPUh；证据data/M6_v1/llama_unseen_budget.json。')
    audit_pipeline=json.loads(Path(c['pipeline']).read_text())
    audit=json.loads(Path(audit_pipeline['steps']['data_audit']['run'],'audit.json').read_text())
    q4_marginal_note='Q4数值边际TVD（seen模板）：'+ '、'.join(split+'='+f(audit[split]['q4_numeric_marginals']['total_variation']) for split in ['reader_train','reader_val','cal','test'])+'；原配对完整复用，ID前缀非配对闭集，未重配。'
    omitted=lock.get('skipped',[]);yes=[]
    q1_degraded=bool(selection and selection.get('degraded'))
    q1_policy_note='Q1按源定义V：GPT2 πσ=1.5，LLaMA πσ='+f(sigma['sigma'])+'；跨源Q1并非同一目标。'+('LLaMA Q1：标签退化程度高，单列报告。' if q1_degraded else '')
    for q in ['Q3','Q4']:
        rr=paired[(paired.source=='llama')&(paired.split=='test')&(paired.qtype==q)&(paired.metric=='accuracy')].iloc[0]
        yes.append(q+('优势复现' if rr.ci_low>0 else '未建立优势')+': Δacc='+f(100*rr.delta_lite_minus_b3)+'pp ['+f(100*rr.ci_low)+', '+f(100*rr.ci_high)+']')
    q1_answer='CODI-LLaMA：'+ '；'.join(yes)+'。配对区间完全大于0才支持优势复现；否则记为未建立。'
    small=transfer[(transfer.split=='test')&(transfer.qtype.isin(['Q3','Q4']))&(transfer.n_labels.astype(str).isin(['100','300']))]
    supported=small[small.ci_low>0]
    baselines=set(small.baseline) if not small.empty else set()
    if 'b5b' not in baselines:
        b5b_note='B5b未执行，不能声称优于B5b。'
    elif len(small[small.baseline=='b5b'])<8:
        b5b_note='B5b小N对照不全，仅可判断已执行N/方向/类型；不能推广到全部小N。'
    else:
        b5b_note='B5b小N对照齐全；decoder冻结、仅重训目标映射。'
    q2_answer='小N按方向/类型/预算逐项判断：'+str(len(supported))+'/'+str(len(small))+'个已执行配对区间下界>0；对照含'+','.join(sorted(baselines))+'。'+b5b_note
    sh=shift[(shift.alpha==.1)&(shift.qtype.isin(['Q3','Q4']))];recover=[]
    for key,g in sh.groupby(['model','qtype','shift']):
        a=g[g.method=='split'].iloc[0];b=g[g.method=='weighted'].iloc[0]
        pp=shift_points[(shift_points.model==key[0])&(shift_points.qtype==key[1])&(shift_points['shift']==key[2])]
        matrix=pp[pp.method.isin(['split','weighted'])].groupby(['problem_id','method']).coverage.mean().unstack()
        if set(matrix.columns)!={'split','weighted'} or matrix.isna().any().any():raise ValueError('Coverage recovery requires matched target problem panels')
        delta=(matrix.weighted-matrix['split']).to_numpy()
        if not np.isclose(delta.mean(),b.coverage-a.coverage):raise ValueError('Paired recovery differs from reported coverage estimand')
        ci=bootstrap_matrix(delta)
        recover.append({'model':key[0],'qtype':key[1],'shift':key[2],'standard_deviation':a.coverage_minus_target,'weighted_deviation':b.coverage_minus_target,'weighted_change':b.coverage-a.coverage,'weighted_change_ci_low':ci[0],'weighted_change_ci_high':ci[1],'weighted_absolute_gap_reduction':abs(a.coverage_minus_target)-abs(b.coverage_minus_target)})
    pd.DataFrame(recover).to_csv(out/'coverage_recovery.csv',index=False)
    severe=[r for r in recover if r['standard_deviation']<-.02]
    affected='、'.join(sorted({r['shift'] for r in severe})) or '无'
    q3_answer='α=.1、Q3/Q4中'+str(len(severe))+'/'+str(len(recover))+'个设置标准覆盖率低于目标超过2pp（描述性界限），涉及'+affected+'；weighted对目标绝对偏差的中位改善='+f(100*np.median([r['weighted_absolute_gap_reduction'] for r in recover]))+'pp。覆盖率变化的配对95%CI见analysis/coverage_recovery.csv；逐设置覆盖率CI及集合大小见shift/summary.csv。估计权重不构成偏移下的覆盖保证。'
    positive=bool((rq.simultaneous_low>0).any());q4_answer='较小sigma'+('出现正收益同时区间，RQ1须限于本设置修订。' if positive else '仍未建立超过greedy的正收益；保留RQ1负面/分析结论。')
    details=['# M6 跨源读取与覆盖率','实验/设置锁：'+str(Path(c['output'])/'training_lock.json')+'；所有模型与低优先级省略在最终reader test推断前锁定。',
      '## 四项回答',q1_answer,q2_answer,q3_answer,q4_answer,
      '## 协议与成本','LLaMA noise选择：sigma='+f(sigma['sigma'])+'；'+('门槛未达，按用户预登记降级使用。' if q1_degraded else '下降<5pp且非退化≥30%，两门槛通过。')+'原四点失败保持不改，完整八点证据reports/M6_noise_selection_v2.md。reader_train='+str(len(sigma['splits']['reader_train']))+'、reader_val=1000、val/cal/test沿用冻结problem ID。reader_train按smoke成本事前等比例截前缀；Q4配对不重采。',
      'GPU墙钟包含smoke、失败与重启；各并行GPU作业直接求和。A='+f(costs_by['A'])+'、B='+f(costs_by['B'])+'、C+D='+f(costs_by['CD'])+'，合计='+f(total)+'GPUh。源模型物理batch前向='+f(sum(x.get('source_forward_calls',0) for x in costs))+'次，每batch调用计1、不按K展开，不能视作等FLOPs。历史冻结模型成本另见上游报告，不混入M6新增预算。',
      'CLR-lite：每源原生维度adapter与直接当前状态skip；迁移冻结全部donor参数，仅训练目标线性adapter；多源每源线性adapter+显式源token+共享lite。B3各类型独立LR/epoch由reader_val选定。',
      '完整CLR沿用事先固定的10epoch上限、min_epochs=8、patience=1；仅epoch≥8的checkpoint参与选模。LLaMA早期reader_val NLL上升的过拟合迹象保留在history.json，未据此修改协议；不是跨全部早期epoch选择最小验证NLL。',
      '小N为不同QA标签实例（同题/轨迹不假设独立），四类型平衡、固定嵌套100/300/1000/3000；完整train无标签归一化共用。reader_val固定128题选小N模型，完整模型用完整reader_val；不使用test选择。N只计目标训练标签，选模reader_val及所有模型共用的cal_A250题校准监督另计，不声称N为全部监督预算。按任务截取ID前缀并复用Q4原配对，子集不自动保留原完整划分的正负数值边际精确相等；实际TVD及计数见data_audit运行的audit.json，不重新配对以改变本次任务。',
      'B5b只训练线性映射，将标准化z注入冻结Qwen3-1.7B，占位token后仅类型模板/选项；读Yes/No或A-D。单源与跨源均只重训目标映射，冻结decoder没有另一个可迁移主体。',
      '## 统计口径','沿用M4.1：主比较clean，accuracy/Brier/NLL/MAE在每步按QA实例微平均、再平均步骤与三个固定seed；CI为10000次problem cluster bootstrap，保留每题每步的实例数权重。seed指标另存。ECE/AUROC在seed_metrics.csv为逐seed点估计，不伪称有配对CI。Q1 reader_val并列all/clean/traj；Q1 Bernoulli样本不在训练前平均。',q1_policy_note,q4_marginal_note,
      'C1差及C3学习曲线CI为逐对95%区间，多个N/方向/类型探索不能无修正宣称全网格显著；A四配置用同时区间。Q3主比较为clean；训练保留M2的轨迹Q3及其真实终点标签。噪声不改变Q3行为标签定义，但会改变所生成轨迹的终点及干扰项分布，不能将全部轨迹Q3数值标签称为与sigma无关。',
      '同题Q3一致性比较预测数值字符串，按底层源模型干净终点正误分组，平均clean t=0..6和三个seed；不会因各源选项位置不同把位置一致误当答案一致。两源Q3候选集可能不同，一致性是各自候选集条件下的诊断，不能单独归因于表示差异。',
      '## RQ4边界','每题每qtype固定一个clean实例；cal_A只拟合temperature/domain classifier，cal_B标签只定阈值。跨源比较C1 source-specific读出函数形成的score偏移，包含表示与函数差异；不能据此分离来源。域分类器为预测概率/步数/熵的summary covariates，不是已知真实密度比。',
      'Mondrian source-only遇未校准源/数据集输出全集；另列使用目标cal_B的Mondrian，额外标签可用性必须区分。跨数据集目标cal来自既有unseen_train，仅作M6校准，不训练reader；'+unseen_note+'既有unseen划分/重叠限制沿用M4证据。跨步t的支持集分离，缺少真实重要性加权所需的重叠，还可能有条件标签变化；截断估计权重只作经验诊断，全集带来的覆盖恢复须与集合大小一起解读。',
      '## 执行缺口',unseen_note,json.dumps(omitted,ensure_ascii=False,indent=2) if omitted else '核心队列无预算省略。',
      '可选Huginn未执行：启动无整张空闲≥24GB GPU；接续时虽出现空闲卡，历史计时外推完整复现约26.0GPUh已超过B剩余2.10GPUh，尚未含数据生成；估计不确定性高，未追加GPU实验。证据data/M6_v1/huginn_admission.json。',
      '## 图与证据','![test learning curves](../data/M6_v1/analysis/learning_curves_test.png)','![reader_val learning curves](../data/M6_v1/analysis/learning_curves_reader_val.png)','所有完整数字：data/M6_v1/analysis/{metrics,seed_metrics,single_source_paired,multisource_paired,transfer_paired,same_problem_summary,coverage_recovery,costs}.csv；逐题Parquet；shift/{summary,seed_summary}.csv；RQ1报告reports/M6_rq1_closure.md；运行目录见evaluation_index.json与pipeline.json。']
    Path('reports/M6_cross_source.md').write_text('\n\n'.join(details))
    lines=['## (a) 状态一句话','M6已完成已执行实验与统计，待审查；低优先级预算省略如(d)，可选Huginn未执行。',
       '## (b) 关卡清单','通过：val代理smoke≤5%/15min、行为标签/选项/划分审计、三seed、训练hash锁后最终reader test、旧冻结哈希复核。'+('LLaMA val300/K16/seed11：σ='+f(selection['sigma'])+'，下降'+f(100*selection['drop'])+'pp，非退化'+f(100*selection['nondegenerate'])+'%。' if selection else '')+'预算 A/B/CD='+ '/'.join(f(costs_by[s]) for s in ['A','B','CD'])+'GPUh，总'+f(total)+'/60。未通过（未执行）：Huginn；'+('; '.join(x['name'] for x in omitted) if omitted else '无其他省略')+'。',
       '## (c) 核心主表（最多3张）','表1：各源test/reader_val，Δ为lite−B3的准确率pp及题bootstrap95%CI。','|源|集|Q3 lite/B3 %|Δ pp [95%CI]|Q4 lite/B3 %|Δ pp [95%CI]|','|---|---|---|---|---|---|']
    for source in ['gpt2','llama']:
      for split in ['test','reader_val']:
        cells=[]
        for q in ['Q3','Q4']:
            scores=agg[(agg.source==source)&(agg.split==split)&(agg.qtype==q)&(agg.kind=='clean')&(agg.metric=='accuracy')]
            lite=scores[scores['case']==f'single_{source}_lite'].iloc[0];b3=scores[scores['case']==f'single_{source}_b3'].iloc[0]
            cells.append(f(100*lite['mean'])+'/'+f(100*b3['mean']))
            r=paired[(paired.source==source)&(paired.split==split)&(paired.qtype==q)&(paired.metric=='accuracy')].iloc[0];cells.append(f(100*r.delta_lite_minus_b3)+' ['+f(100*r.ci_low)+','+f(100*r.ci_high)+']')
        lines.append('|'+ '|'.join([source,split,*cells])+'|')
    lines+=['表2：小N迁移相对scratch，test；括号为胜出配对数/已执行数（Q3/Q4，N100/300，区间下界>0）。','|方向|vs B3|vs lite|vs B5b|','|---|---|---|---|']
    for a,b in [('gpt2','llama'),('llama','gpt2')]:
        cells=[]
        for baseline in ['b3','lite','b5b']:
            g=small[(small.direction==a+'->'+b)&(small.baseline==baseline)];cells.append(str(int((g.ci_low>0).sum()))+'/'+str(len(g)) if len(g) else '未执行')
        lines.append('|'+ '|'.join([a+'→'+b,*cells])+'|')
    lines+=['表3：test覆盖率偏差pp及平均集合大小范围，α=.1、Q3/Q4、两个reader/方向合并描述；完整CI在证据CSV。','|偏移|standard偏差|weighted偏差|standard size|weighted size|','|---|---|---|---|---|']
    for name in ['source','dataset','step']:
        g=sh[sh['shift'].str.startswith(name)];aa=g[g.method=='split'];bb=g[g.method=='weighted'];lines.append('|'+ '|'.join([name,f(100*aa.coverage_minus_target.min())+'…'+f(100*aa.coverage_minus_target.max()),f(100*bb.coverage_minus_target.min())+'…'+f(100*bb.coverage_minus_target.max()),f(aa['size'].min())+'…'+f(aa['size'].max()),f(bb['size'].min())+'…'+f(bb['size'].max())])+'|')
    lines +=[q1_answer,q2_answer,q3_answer,q4_answer,q1_policy_note,
       '## (d) 本里程碑新增的偏离与阻塞','Huginn未准入：启动资源不足，后续复现成本估计26.0GPUh>B余量2.10GPUh，见huginn_admission.json；'+unseen_note+'B按授权成本缩小train至'+str(len(sigma['splits']['reader_train']))+'题；'+q4_marginal_note+'source smoke目录/Cache API失败、RQ1递归包装失败保留并计成本，修复后独立重启、校验续用。A启动YAML含未用旧网格，实际协议固定在启动源码，未完全YAML化；事后审计rq1/configuration_audit.json不冒充启动记录。控制器交接修复D→C2/C4优先级，GPU训练未中断。CPU绘图库路径及独立审计PYTHONPATH已修复；首条失败log同名覆盖，同错误第二次log保留。'+('预算省略见(b)及training_lock.json；完整原因在详细报告。' if omitted else '无其他省略。')+' Mondrian缺目标stratum时全集；目标cal版本另列。weighted只估计summary密度比，跨步t支持不重叠，无无条件保证；跨源C1 score偏移不能分离表示与函数差异。',
       '## (e) 需要审查者决定的问题','是否接受A的已记录配置偏离、本里程碑执行范围及四项结论。',
       '## (f) 证据路径','详细：reports/M6_cross_source.md、reports/M6_rq1_closure.md、reports/M6_noise_selection_v2.md。统计/CI：data/M6_v1/analysis/*.csv、shift/summary.csv。配置/运行：configs/M6_pipeline_noise_v2.yaml、'+str(c.get('pipeline','data/M6_v1/pipeline.json'))+'、evaluation_index.json、training_lock.json；源码/数据/结果冻结data/M6_v1/freeze.json；新增启动错误及限制docs/BLOCKERS.md。']
    if q1_degraded:
        lines.insert(lines.index('## (d) 本里程碑新增的偏离与阻塞')+1,'未通过：八点无同时达两噪声门槛者；按用户预登记“门槛未达，降级使用”，并非历史关卡通过。')
    review=markdown(lines)
    if len(review.encode())>8000:raise ValueError('M6 review exceeds 8000 UTF8 bytes')
    Path('reports/M6_REVIEW.md').write_text(review)
    return total,costs_by


def finalize(c,run,total,section_costs):
    # Audit old immutable files again. Do not fabricate a freeze from status flags.
    verification=json.loads(Path(c['output'],'frozen_inputs_verification.json').read_text())
    old=[]
    for freeze in verification:
        data=json.loads(Path(freeze).read_text());entries=data.get('sha256',data.get('files',data))
        if isinstance(entries,list):
            entries={r.get('path'):r.get('sha256') for r in entries if isinstance(r,dict) and r.get('path')}
        if not isinstance(entries,dict):raise ValueError('Unknown upstream freeze structure '+freeze)
        for path,h in entries.items():
            if isinstance(h,dict):h=h.get('sha256')
            if not isinstance(h,str) or len(h)!=64:continue
            p=Path(path)
            if not p.exists() or digest(p)!=h:raise ValueError('Frozen input changed '+path)
            old.append(path)
    write_json(Path(c['output'])/'final_frozen_inputs_verification.json',{'checked':len(old),'mismatches':[]})
    files={p for p in Path('clr').rglob('*.py') if 'm6' in p.stem}|set(Path('scripts').glob('m6*.py'))|set(Path('configs').glob('M6*.yaml'))|{Path('tests/test_m6.py'),Path('docs/M6_TASK.md'),Path('reports/M6_REVIEW.md'),Path('reports/M6_cross_source.md'),Path('reports/M6_rq1_closure.md'),Path('reports/M6_noise_selection_v1.md'),Path('reports/M6_noise_selection_v2.md')}
    for root in [Path(c['output']),Path(c['sources']['llama']['root'])]:
        files.update(p for p in root.rglob('*') if p.is_file() and p.name not in ['freeze.json','freeze.json.tmp'] and not(p.name.startswith('pipeline') and p.suffix=='.json'))
    # Child stdout and no-run startup failures also belong to the evidence.
    # Current controller/report streams finish after this report returns.
    open_logs={Path('reports/M6_extension_pipeline.log'),
               Path('reports/M6_parallel_eval_pipeline.log'),
               Path('reports/M6_parallel_eval_resume.log')}
    for statepath in Path(c['output']).rglob('pipeline*.json'):
        state=json.loads(statepath.read_text())
        open_logs.update(Path(v['log']) for v in state.get('steps',{}).values() if v.get('status')=='running' and v.get('log'))
    files.update(p for p in Path('reports').glob('M6_*.log') if p not in open_logs)
    active=[]
    for row in hours():
        root=Path(row['run']);files.update(p for p in root.rglob('*') if p.is_file())
        if json.loads((root/'metrics.json').read_text()).get('status')=='running':active.append(root)
    # Reporting and its parent controller still have open metrics/event logs.
    # Freeze their startup code only; these logs finish after this function.
    files={p for p in files if not any(a in p.parents for a in active) or 'source_snapshot' in p.parts}
    write_json(Path(c['output'])/'freeze.json',{'version':'M6_v1','files':{str(p):digest(p) for p in sorted(files)},'gpu_hours':total,'section_gpu_hours':section_costs,'scope':'completed scientific code/data/results; live/controller records excluded'})
    status='M6实验与统计完成，待审查；唯一审查reports/M6_REVIEW.md，冻结data/M6_v1/freeze.json。'
    for name,begin,end in [('AGENTS.md','<!-- M6_LIVE -->','<!-- /M6_LIVE -->'),
                           ('docs/STATUS.md','<!-- M6_LIVE -->','<!-- /M6_LIVE -->')]:
        p=Path(name);text=p.read_text();block=begin+'\n'+status+'\n'+end
        if begin in text:text=text[:text.index(begin)]+block+text[text.index(end)+len(end):]
        else:text+='\n'+block+'\n'
        p.write_text(text)
    for name,body in [('docs/STATUS.md',status),('docs/EXPERIMENTS.md',status+'\n接续：.venv/bin/python -m scripts.m6_pipeline configs/M6_pipeline_noise_v2.yaml；三seed配对CI及负面结果见审查/详细报告。\n新增GPUh='+f(total)+'（含smoke/失败/重启）；成本data/M6_v1/analysis/costs.csv。'),('docs/CHANGELOG.md','新增M6双源数据、lite适配器迁移、跨偏移统计与预算控制器；旧冻结未改。'+status)]:
        with Path(name).open('a') as stream:stream.write('\n## M6 '+time.strftime('%Y-%m-%d')+'\n'+body+'\n')
    plan=json.loads(Path(c['output'],'plan.json').read_text())
    completed_evaluations=sum(len(v) for v in json.loads(Path(c['evaluation_index']).read_text()).values())
    planned_evaluations=sum(len(c['sources']) if v['config']['kind']=='multi' else 1
        for v in json.loads(Path(c['eval_lock']).read_text())['models'].values())
    live(stage='completed_pending_review',milestone_complete=True,review='reports/M6_REVIEW.md',
         review_status='complete_pending_review',reason=None,resource_wait=None,freeze=str(Path(c['output'])/'freeze.json'),
         final_evaluations_complete=completed_evaluations,final_evaluations_planned=planned_evaluations,
         gpu_hours=total,gpu_hours_so_far=total,gpu_hours_by_section=section_costs,
         generation_counts={s:{'persisted':len(rows),'planned':len(rows)} for s,rows in plan['splits'].items()})


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();snapshot(run,{**c,'section':'CD'})
    lock=json.loads(Path(c['eval_lock']).read_text());index=json.loads(Path(c['evaluation_index']).read_text())
    if not lock.get('all_training_locked'):raise ValueError('Report requires all training locked')
    optional_unseen_cost(c)
    out=Path(c['output'])/'analysis';out.mkdir(parents=True,exist_ok=True)
    frame,agg,paired,transfer=collect(c,lock,index,out);same_problem(c,index,out);curves(agg,out,c)
    plan=json.loads(Path(c['output'],'plan.json').read_text());selected=plan['noise_selection']['selected']
    q1=agg[agg.qtype=='Q1'].copy();q1['policy_sigma']=q1.source.map({'gpt2':1.5,'llama':plan['sigma']});q1['label_high_degeneracy']=(q1.source=='llama')&bool(selected.get('degraded'))
    q1['target_definition']='V_s(z,t; source-specific pi_sigma), not a shared cross-source target';q1.to_csv(out/'Q1_source_specific.csv',index=False)
    costs=hours()
    for row in costs:
        metrics=json.loads(Path(row['run'],'metrics.json').read_text());cfg=json.loads(Path(row['run'],'config.json').read_text())
        row.update(status=metrics['status'],source_forward_calls=metrics.get('source_forward_calls',0),seed=cfg.get('seed'),device=cfg.get('device'),kind=cfg.get('kind'))
        if metrics['status']=='running' and row['gpu_hours']>0:raise ValueError('GPU work remains active during final cost accounting')
    pd.DataFrame(costs).to_csv(out/'costs.csv',index=False)
    write_json(out/'costs.json',{'gpu':'NVIDIA GeForce RTX 4090','gpu_hours':sum(r['gpu_hours'] for r in costs),'source_forward_calls':sum(r['source_forward_calls'] for r in costs),'source_forward_calls_unit':'physical batched _forward calls; K lanes counted once, not FLOPs-equivalent','includes':'sum of GPU-job lifetime wall time, including smoke/failures/restarts; CPU-only controllers/reports zero; frozen upstream training reported separately'})
    total,section_costs=write_reports(c,lock,index,out,agg,paired,transfer,costs)
    if total>60 or any(section_costs[s]>{'A':3,'B':30,'CD':27}[s] for s in section_costs):raise ValueError('Budget violation cannot be marked passed')
    finalize(c,run,total,section_costs)
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start,review_bytes=Path('reports/M6_REVIEW.md').stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

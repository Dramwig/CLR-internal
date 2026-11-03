"""Auditable M2/M3 reports; no claims from unfinished experiments."""
import argparse,json,csv,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.stats_v2 import bootstrap_mean
from clr.data.questions.core import seed

def ci(values):
    a=np.asarray(values,dtype=float)
    return f'{a.mean():.4f} ± {4.3026527299*a.std(ddof=1)/np.sqrt(len(a)):.4f}' if len(a)==3 else f'{a.mean():.4f} (deterministic)'
def load(path):return json.loads(Path(path).read_text())
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def main(c):
 with recorded(c) as run:
    reports=Path('reports');figs=reports/'M3_figures';figs.mkdir(exist_ok=True)
    records={};missing=[]
    for k in ['question_run','baseline_run','patchscope_run','layers_run','quant_run','official_run','svamp_run','asdiv_run']:
        if not c.get(k):missing.append(k);continue
        m=load(Path(c[k])/'metrics.json')
        if m['status']!='completed':missing.append(k)
        else:records[k]=m
    if c['require_complete'] and missing:raise ValueError('Incomplete required runs: '+str(missing))
    summary=load(Path(c['output'])/'full/summary.json');preamble='已完成。' if not missing else '进行中，缺少完整结果：'+', '.join(missing)
    qlines=['# M2 类型化问题库',preamble,
      '输入严格限定观测状态、类型化模板和选项。训练模板0–3，评估保留模板4–5；每个评估实例同时评估一个已见模板及两个保留模板。原题文本、gold CoT不传入模型。',
      'Q1：每条轨迹起点对应clean状态，逐条保留Bernoulli标签；t>t0读取实际观测，不读取feedback；不标t<t0的clean前缀。clean终点用sigma=0正误，traj终点用自身最终正误。cal/test用K16的逐次结果；计算Bernoulli NLL/Brier，MAE对MC均值。',
      'Q2仅clean早退。Q3四个数值归一化选项，含真实模型最终答案，干扰项来自同题行为、gold及固定训练集其他答案；无效输出表示为<invalid>，不丢题。Q4在clean状态上以显式方程RHS为中间量（包括最终计算结果），跨题互换配对保证正负数值边际完全相同，排除本题真值；未匹配实例显式计数。',
      'cal_A250 / cal_B250按问题ID冻结；Mondrian按(qtype,t)分层、每题固定哈希抽一个实例。Q1集合预测未来Bernoulli正误，不是V的置信区间。',
      '## 样本与模板规模',table(['split','problems','QA rows','Q4未匹配题'],[(s,v['problems'],v['rows'],len(v['q4_unpaired'])) for s,v in summary.items()]),
      '## Q1各步覆盖',table(['split','t','clean样本','traj样本','缺clean标签比例'],[(s,t,next((r['n'] for r in v['by_qtype_step_kind'] if r['qtype']=='Q1' and r['t']==t and r['kind']=='clean'),0),next((r['n'] for r in v['by_qtype_step_kind'] if r['qtype']=='Q1' and r['t']==t and r['kind']=='traj'),0),f"{v['q1_missing_clean_fraction'][str(t)]:.4f}") for s,v in summary.items() for t in range(7)])]
    dist=[];position=[]
    for split in summary:
        rows=pq.read_table(Path(c['output'])/f'full/{split}.parquet',columns=['qtype','state_ref','options','label','label_soft','is_heldout_template']).to_pylist()
        bins=defaultdict(list);pos=defaultdict(lambda:np.zeros(4,dtype=int));option_counts=defaultdict(lambda:defaultdict(int))
        for r in rows:
            if r['is_heldout_template']:continue
            key=(split,r['qtype'],r['state_ref']['t'],r['state_ref']['kind']);bins[key].append(r['label_soft'])
            if r['qtype']=='Q3':pos[r['state_ref']['t']][r['label']]+=1
            for option in r['options']:option_counts[r['qtype'],r['state_ref']['t']][option]+=1
        write_json(run.path/f'{split}_option_values.json',[{'qtype':q,'t':t,'counts':dict(v)} for (q,t),v in option_counts.items()])
        for key,v in bins.items():dist.append(dict(zip(['split','qtype','t','kind'],key),n=len(v),label_mean=float(np.mean(v))))
        for t,v in pos.items():position.append({'split':split,'t':t,'counts':v.tolist(),'fractions':(v/v.sum()).tolist(),'max_deviation_from_uniform':float(np.max(abs(v/v.sum()-.25)))})
    write_json(run.path/'question_distributions.json',dist);write_json(run.path/'option_positions.json',position)
    qlines+=['## 标签/位置分布',f'逐split×qtype×t×kind完整分布：`{run.path}/question_distributions.json`；choice的label_mean是索引均值，不解释为正例率。',table(['split','t','correct-position fractions','max deviation'],[(r['split'],r['t'],[round(x,4) for x in r['fractions']],round(r['max_deviation_from_uniform'],4)) for r in position]),f'选项数值频次按qtype×t×split保存在 `{run.path}/<split>_option_values.json`。', '选项顺序由固定seed独立置换，有限样本不要求恰好25%；未因位置偏差重抽seed。']
    if 'quant_run' in records:
        m=records['quant_run'];qlines+=['## M1量化缺口补查',f"中断片段reader_train/test各100题：fp16逐元素一致={m['fp16_exact_all']}；最大绝对误差={m['max_abs']:.8f}；最大相对L2={m['max_relative_l2']:.8g}。这是随机抽查，不是恢复丢失的全量聚合。fp32与fp16并不逐元素相等；比较的是重算fp32转fp16与已存fp16。",f"证据：`{c['quant_run']}`。"]
    (reports/'M2_questions.md').write_text('\n\n'.join(qlines)+'\n')
    stats=[]
    for key in ['baseline_run','patchscope_run']:
        if key in records:stats+=load(Path(c[key])/'statistics.json')
    write_json(run.path/'all_baseline_statistics.json',stats)
    flat=[]
    for r in stats:
        for calibrated in ['raw','calibrated']:
            flat.append({**{k:v for k,v in r.items() if k not in ['raw','calibrated']},'calibration':calibrated,**r[calibrated]})
    if flat:
        keys=sorted(set().union(*(r.keys() for r in flat)))
        with (reports/'M3_metrics.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(flat)
    grouped=defaultdict(lambda:defaultdict(list))
    for r in stats:
        if r['kind']=='clean':
            for metric,v in r['calibrated'].items():
                if v is not None and (metric!='auroc' or r['qtype'] in ['Q2','Q4']):grouped[r['baseline'],r['qtype'],r['split'],r['heldout'],metric][r['seed']].append(v)
    def values(b,q,s,h,m):return [np.mean(v) for v in grouped.get((b,q,s,h,m),{}).values()]
    mainrows=[]
    for b in ['B0','B1','B2','B3','B4','B5a']:
        for q in ['Q1','Q2','Q3','Q4']:
            for s in ['reader_val','test']:
                if not values(b,q,s,False,'nll'):continue
                mainrows.append([b,q,s]+[ci(values(b,q,s,False,m)) if values(b,q,s,False,m) else 'NA' for m in ['accuracy','auroc','nll','brier','mae','ece15']])
    blines=['# M3 基线读取器',preamble,
      'B0/B1使用两层MLP骨架，分别置零z/同一步跨题打乱z；B2线性，B3两层MLP，Q3共享选项数值特征双线性打分，保证选项置换等变。问题模板用确定性字节频率特征；不输入原题。B4仅用z训练Q1最终正误标量，原样用于Q2并单独校准。B5a冻结CODI-GPT2，在占位token的输入embedding处注入原始观测z，随后仅输入类型化prompt/选项，取单token Yes/No或A–D logits；无映射训练，尺度/表示空间不匹配可能导致负面结果。B5b明确推迟至M6。',
      '训练超参仅在reader_val选择；cal_A按qtype温度缩放，cal_B仅conformal。训练seed11/22/33；表中均值±95% CI为三个seed的t区间（df=2），先对t等权平均。原始与校准、逐qtype×t×split×clean/traj×模板结果见[M3_metrics.csv](M3_metrics.csv)。B5a确定性，不制造三份重复seed的零方差区间；按题bootstrap补充于本报告run。',
      '二分类Brier为E[(p-y)^2]，Q1对每个Bernoulli结果平均（不是对MC均值做平方误差）；Q3为多类Brier之和。ECE15定义为top-confidence对预测正确率的15个等宽箱差。单步score MAE对MC均值；reader_val的Bernoulli标签与test K16均值精度不同，MAE不可直接视为纯域偏移。',
      '## 已见模板、干净状态主表',table(['baseline','qtype','split','accuracy','AUROC','NLL','Brier','MAE','ECE'],mainrows),
      '## 保留模板',table(['baseline','qtype','split','NLL','accuracy','ECE'],[[b,q,s]+[ci(values(b,q,s,True,m)) for m in ['nll','accuracy','ece15']] for b in ['B0','B1','B2','B3','B4','B5a'] for q in ['Q1','Q2','Q3','Q4'] for s in ['reader_val','test'] if values(b,q,s,True,'nll')])]
    if 'baseline_run' in records:
        cp=load(Path(c['baseline_run'])/'conformal.json')
        for r in cp:r['split']='test'
        extra=[]
        for b in ['B2','B3']:
            for q in ['Q1','Q2','Q3','Q4']:
                for rs in [11,22,33]:
                    for held in [False,True]:
                        rows=pq.read_table(Path(c['output'])/'full/reader_val.parquet',filters=[('qtype','=',q),('is_heldout_template','=',held)]).to_pylist()
                        preds=pq.read_table(Path(c['baseline_run'])/f'{b}_{q}_{rs}_reader_val_{held}.parquet').to_pylist()
                        assert len(rows)==len(preds)
                        p=softmax(np.array([r['logits'] for r in preds])/preds[0]['temperature'],axis=1)
                        for t in range(7):
                            unique={}
                            for i,r in enumerate(rows):
                                if r['state_ref']['kind']!='clean' or r['state_ref']['t']!=t:continue
                                pid=r['problem_id'];rank=lambda j:seed(rs,pid,q,t,tuple(rows[j]['options']),rows[j]['template_id'])
                                if pid not in unique or rank(i)<rank(unique[pid]):unique[pid]=i
                            ii=list(unique.values())
                            if not ii:continue
                            yy=np.eye(4)[[rows[i]['label'] for i in ii]] if q=='Q3' else np.array([[1-rows[i]['label_soft'],rows[i]['label_soft']] for i in ii])
                            for alpha in [.05,.1,.2]:
                                ref=next(r for r in cp if r['baseline']==b and r['qtype']==q and r['seed']==rs and r['t']==t and r['heldout']==held and r['alpha']==alpha)
                                threshold=float('inf') if ref.get('threshold_is_infinite') else ref['threshold']
                                sets=1-p[ii]<=threshold
                                extra.append({**ref,'split':'reader_val','n':len(ii),'coverage':float((sets*yy).sum(1).mean()),'size':float(sets.sum(1).mean())})
        cp+=extra;write_json(run.path/'conformal.json',cp)
        cpgroup=defaultdict(list)
        for r in cp:cpgroup[r['baseline'],r['qtype'],r['split'],r['t'],r['heldout'],r['alpha']].append(r)
        blines+=['## Mondrian conformal', '在每个(qtype,t)层按题独立取一个实例。Q1校准每题固定随机选一个MC Bernoulli结果；test覆盖率对该题K16结果平均。阈值使用ceil((n+1)(1−α))；极小层返回全集合。有限样本保证依赖层内可交换性。cal_B/test问题来自同一原池，但原cohort上下文、Q3各split的其他题干扰池、Q4跨题数值配对引入依赖，不能据此宣称无条件有限样本保证；held-out模板还改变模板分布。这里报告经验覆盖率。reader_val源训练分布不享有跨域保证。',table(['baseline','qtype','split','t','heldout','alpha','coverage','set size'],[[*k,ci([x['coverage'] for x in v]),ci([x['size'] for x in v])] for k,v in sorted(cpgroup.items())])]
    if 'layers_run' in records:
        layers=load(Path(c['layers_run'])/'layers.json');fig,axes=plt.subplots(1,4,figsize=(15,3))
        for q,ax in zip(['Q1','Q2','Q3','Q4'],axes):
            means=[np.mean([r['accuracy'] for r in layers if r['qtype']==q and r['layer']==l]) for l in range(13)]
            ax.plot(range(13),means,marker='o');ax.set_title(q);ax.set_xlabel('layer');ax.set_ylabel('accuracy')
        fig.tight_layout();fig.savefig(figs/'layers.png');plt.close(fig)
        blines+=['## 层分析','![B2 layers](M3_figures/layers.png)','仅1000题源训练分布面板，800 reader_train / 200 reader_val，不是未见test上的层选择结论。所有层同一训练设置，不据test挑层。']
    shift=['# M3 分布偏移诊断',preamble]
    if 'official_run' in records:
        comparison=load(Path(c['output'])/'official500_comparison.json')
        shift+=['## 官方500题健全性检查',f"官方test.py bf16/batch128：{comparison['official_same500']}；相同500题M1 fp32生产结果：{comparison['M1_same500']}。两协议仅差0.4pp；源训练分布高准确率在官方实现下仍存在，不能将训练/未见差距解释为CLR接口独有的协议错误。官方500题按新batch128成组，M1保留原cohort；这里只作准确率健全性检查，不是逐题一致性验收。此检查不证明没有其他混杂因素。",f"官方run `{c['official_run']}`；数据固定在 `data/M23_v1a/official500.json`。"]
    gaprows=[]
    for b in ['B0','B1','B2','B3','B4','B5a']:
        for q in ['Q1','Q2','Q3','Q4']:
            for m in ['accuracy','nll','brier','mae','ece15']:
                a=values(b,q,'reader_val',False,m);z=values(b,q,'test',False,m)
                if a and z:gaprows.append([b,q,m,float(np.mean(a)),float(np.mean(z)),float(np.mean(z)-np.mean(a))])
    shift+=['## reader_val → test', '表为已见模板clean、各t等权平均；差值定义test−reader_val。NLL/Brier越低越好，accuracy越高越好；MAE标签噪声差异见基线报告。',table(['baseline','qtype','metric','reader_val','test','delta'],gaprows)]
    if 'layers_run' in records:
        dom=load(Path(c['layers_run'])/'domain.json');shift+=['## 潜状态域分类器',table(['t','held-out domain AUROC'],[[t,ci([r['auroc'] for r in dom if r['t']==t])] for t in range(7)]),'域诊断独立按problem ID做60/20/20划分；每域最多1005题，训练/验证/评估问题不交叉。这里test题只用于用户指定的域识别诊断；域分类器或其参数不反馈给基线选择。']
    # Reliability curves, separate distributions, no change to baseline settings.
    for q in ['Q1','Q2']:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for split,ax in zip(['reader_val','test'],axes):
            ax.plot([0,1],[0,1],'k--',alpha=.5)
            for key in ['baseline_run','patchscope_run']:
                if key not in records:continue
                for b in (['B0','B1','B2','B3','B4'] if key=='baseline_run' else ['B5a']):
                    preds=[];labels=[]
                    for file in Path(c[key]).glob(f'{b}_{q}_*_{split}_False.parquet'):
                        rr=pq.read_table(file,filters=[('kind','=','clean')]).to_pylist()
                        pp=softmax(np.array([r['logits'] for r in rr])/rr[0]['temperature'],axis=1)[:,1]
                        preds.extend(pp);labels.extend(r['label_soft'] for r in rr)
                    if not preds:continue
                    pp=np.array(preds);yy=np.array(labels);points=[]
                    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
                        mask=(pp>=lo)&(pp<hi if hi<1 else pp<=hi)
                        if mask.any():points.append((pp[mask].mean(),yy[mask].mean()))
                    ax.plot(*zip(*points),marker='.',label=b)
            ax.set_title(split);ax.set_xlabel('predicted probability');ax.set_ylabel('observed correct');ax.legend()
        fig.tight_layout();fig.savefig(figs/f'{q}_reliability.png');plt.close(fig)
        shift.append(f'![{q} reliability](M3_figures/{q}_reliability.png)')
    if 'patchscope_run' in records:
        deterministic_ci=[]
        for q in ['Q1','Q2','Q3','Q4']:
            for split in ['reader_val','test']:
                path=Path(c['patchscope_run'])/f'B5a_{q}_0_{split}_False.parquet'
                rr=pq.read_table(path,filters=[('kind','=','clean')]).to_pylist()
                pp=softmax(np.array([r['logits'] for r in rr])/rr[0]['temperature'],axis=1)
                yy=np.eye(4)[[r['label'] for r in rr]] if q=='Q3' else np.array([[1-r['label_soft'],r['label_soft']] for r in rr])
                contribution=defaultdict(list)
                for i,r in enumerate(rr):
                    contribution[r['problem_id']].append([yy[i,pp[i].argmax()],-(yy[i]*np.log(np.maximum(pp[i],1e-12))).sum(),(1-2*(yy[i]*pp[i]).sum()+(pp[i]**2).sum())/(1 if q=='Q3' else 2)])
                by_problem=np.array([np.mean(v,axis=0) for v in contribution.values()])
                deterministic_ci.append({'qtype':q,'split':split,'aggregation':'mean within problem over observed clean instances, then across problems',**{m:bootstrap_mean(by_problem[:,j]) for j,m in enumerate(['accuracy','nll','brier'])}})
        write_json(run.path/'B5a_question_bootstrap.json',deterministic_ci)
        blines+=['## B5a确定性按题bootstrap补充', '以下先在题内平均clean实例，再按题bootstrap10k，区别于主表各t等权平均；不把同题模板/rollout当独立问题。ECE/AUROC在逐层表给点估计，不提供伪造的seed区间。',table(['qtype','split','accuracy [95% CI]','NLL [95% CI]','Brier [95% CI]'],[[r['qtype'],r['split']]+[str(r[m]['mean'])+' '+str(r[m]['ci95']) for m in ['accuracy','nll','brier']] for r in deterministic_ci])]
    # Required final three answers.
    blines+=['## (1) B0 比随机猜测高多少？',table(['qtype','split','accuracy minus uniform guess','NLL reduction vs uniform'],[[q,s,float(np.mean(values('B0',q,s,False,'accuracy')))-(.25 if q=='Q3' else .5),np.log(4 if q=='Q3' else 2)-float(np.mean(values('B0',q,s,False,'nll')))] for q in ['Q1','Q2','Q3','Q4'] for s in ['reader_val','test'] if values('B0',q,s,False,'nll')]),
      'Q1随机基准为p=0.5（NLL=ln2、Brier=.25）；其accuracy仅为附加二元判别统计。B0偏离随机反映标签/选项先验，不自动证明源状态有信息。',
      '## (2) B2/B3 在test距离完美且已校准多远？',table(['baseline','qtype','1-accuracy','NLL','Brier','MAE','ECE'],[[b,q,1-np.mean(values(b,q,'test',False,'accuracy'))]+[float(np.mean(values(b,q,'test',False,m))) if values(b,q,'test',False,m) else 'NA' for m in ['nll','brier','mae','ece15']] for b in ['B2','B3'] for q in ['Q1','Q2','Q3','Q4'] if values(b,q,'test',False,'nll')]),
      'Q2/Q3/Q4完美确定分类的理想NLL/Brier/ECE为0、accuracy为1。Q1随机未来结果即使概率完全正确也有不可约熵和Brier V(1−V)，所以不能把其全部NLL/Brier称为模型距完美的差距；MAE/ECE的理想值为0，但K16评估有抽样误差。',
      '## (3) reader_val → test 落差有多大？',table(['baseline','qtype','metric','reader_val','test','test−reader_val'],gaprows),
      '这些是固定M1数据上的诊断，不代表CLR改善已获证明；未训练CLR。']
    (reports/'M3_baselines.md').write_text('\n\n'.join(blines)+'\n');(reports/'M3_shift.md').write_text('\n\n'.join(shift)+'\n')
    unseen=['# M3 未见题池侦察',preamble,
      '|候选|官方许可/格式|处理|\n|---|---|---|\n|[SVAMP](https://github.com/arkilpatel/SVAMP)|MIT；Body/Question/Answer JSON|去重后993题，clean+早退|\n|[ASDiv](https://github.com/chaochun/nlu-asdiv-dataset)|README声明CC BY-NC4.0；XML，答案含单位|仅单个数值答案子集，去重后固定抽≤1000题|\n|[Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)|MIT；question/answer两列，答案是生成的长解答|暂不作为已核实未见池：种子经Lila重划分，GSM8K原始test谱系尚未逐题排除；单一数值gold还需核验|\n|[MAWPS](https://github.com/sroy9/mawps)|作者仓库主要是服务器代码，未核实所选数据文件的许可|未调用源模型，不冒用第三方许可声明|',
      'Orca-Math论文描述从Lila train/validation取包含GSM8K的种子再扩写，不能把Lila split名称当作原始GSM8K split证明。[作者论文](https://arxiv.org/html/2402.14830v1)。ASDiv仅按README允许的非商业研究使用。GSM-Plus/GSM-Hard/GSM-Symbolic全部排除。',
      '对入选池以同一13-gram规则与完整GSM8K train/test及GSM8k-Aug检查；任何重叠剔除，记录逐题证据。此规则不能证明基础GPT2预训练从未见过，也不能排除不共享13-gram的语义改写。每池GPU预算2.5小时，合计≤5小时；只clean dump和7步早退，不生成轨迹。']
    for key in ['svamp_run','asdiv_run']:
        if key in records:
            m=records[key];unseen+=[f'## {key}',f"运行 `{c[key]}`；GPU小时={m['gpu_hours']:.4f}，源前向={m['source_forward_calls']}。",table(['t','accuracy','95% CI'],[[t,v['mean'],v['ci95']] for t,v in m['accuracy_by_step'].items()])]
    reference=[]
    for split in ['reader_train','reader_val','test']:
        rr=pq.read_table(Path(c['root'])/split/'early_exit.parquet',columns=['t','correct']).to_pylist()
        reference.extend({'split':split,'t':t,**bootstrap_mean([int(r['correct']) for r in rr if r['t']==t])} for t in range(7))
    unseen+=['## M1固定参考与早退曲线',table(['split','t','accuracy','95% CI'],[[r['split'],r['t'],r['mean'],r['ci95']] for r in reference])]
    fig,ax=plt.subplots(figsize=(7,4))
    for split in ['reader_train','reader_val','test']:
        ax.plot(range(7),[r['mean'] for r in reference if r['split']==split],marker='o',label=split)
    for key in ['svamp_run','asdiv_run']:
        if key in records:ax.plot(range(7),[records[key]['accuracy_by_step'][str(t)]['mean'] for t in range(7)],marker='o',label=key.replace('_run',''))
    ax.set_xlabel('early-exit step');ax.set_ylabel('accuracy');ax.legend();fig.tight_layout();fig.savefig(figs/'unseen_early_exit.png');plt.close(fig)
    unseen.append('![early exit](M3_figures/unseen_early_exit.png)')
    (reports/'M3_unseen_pools.md').write_text('\n\n'.join(unseen)+'\n')
    write_json(run.path/'inputs.json',{k:c[k] for k in records})
    run.write_metrics(status='completed',milestone_complete=not missing,source_forward_calls=0,gpu_hours=0,reports=['M2_questions.md','M3_baselines.md','M3_shift.md','M3_unseen_pools.md'])
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

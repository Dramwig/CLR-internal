"""Research tables/paired group bootstrap from recorded, locked M7 predictions only."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,math
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live
from clr.audit.m7_evaluation import qa_summary
from clr.audit.m7_statistics import nonlinear_qa_intervals


def fmt(x):return 'NA' if x is None or not np.isfinite(x) else format(float(x),'.3g')
def ci_cell(row,percent=True):
    m=100 if percent else 1
    return f"{fmt(row['mean']*m)} [{fmt(row['lo']*m)},{fmt(row['hi']*m)}]"

def interval(values,weights):
    x=np.asarray(values,float);boot=weights@x/len(x)
    return {'mean':float(x.mean()),'lo':float(np.quantile(boot,.025)),'hi':float(np.quantile(boot,.975))}

def weights_for(groups,c,cache,split):
    groups=tuple(groups)
    key=(split,groups)
    if key not in cache:
        seed=c['bootstrap_seed']+int(hashlib.sha256(split.encode()).hexdigest()[:8],16)
        rng=np.random.default_rng(seed)
        cache[key]=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=c['bootstrap_repeats']).astype(np.float32)
    return cache[key]


def causal_stats(c,out):
    log=Path('reports/M7_evidence/acceptance_val.log').read_text().splitlines()[0]
    run=Path(log.strip());rows=json.loads((run/'causal_predictions.json').read_text())
    n=len(rows);rng=np.random.default_rng(c['bootstrap_seed']);w=rng.multinomial(n,np.full(n,1/n),size=c['bootstrap_repeats'])
    result={name:interval([r[name]['correct'] for r in rows],w) for name in ['message','none','mismatch']}
    for name in ['none','mismatch']:
        result['gain_vs_'+name]=interval([int(r['message']['correct'])-int(r[name]['correct']) for r in rows],w)
    write_json(out/'causal_bootstrap.json',{'run':str(run),'groups':n,'statistics':result})
    return result


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);out=root/'analysis';out.mkdir(exist_ok=False)
        qas=[];dets=[];naturals=[];summary=[]
        for split in ['reader_val','test']:
            pointer=json.loads((root/'evaluated'/f'{split}.json').read_text());p=Path(pointer['run'])
            if json.loads((p/'metrics.json').read_text())['status']!='completed':raise ValueError('Incomplete evaluation')
            qas.extend(json.loads(l) for l in (p/'qa_predictions.jsonl').open())
            d=pd.read_csv(p/'detection_scores.csv');dets.append(d)
            naturals.append(pd.read_csv(p/'natural_commitment.csv'))
            summary.append(pd.read_csv(p/'qa_summary.csv'))
        q=pd.DataFrame(qas);d=pd.concat(dets,ignore_index=True);nat=pd.concat(naturals,ignore_index=True)
        pd.concat(summary).to_csv(out/'qa_per_seed.csv',index=False)
        cache={};qa_rows=[];qa_group={}
        for keys,frame in q.groupby(['split','kind','task','templates','category'],sort=True):
            # Question instances first average within a source group, then seeds.
            g=frame.groupby(['group_id','seed'])[['correct','nll','brier','coverage_0.01','size_0.01','coverage_0.05','size_0.05']].mean().groupby('group_id').mean().sort_index()
            w=weights_for(g.index,c,cache,keys[0])
            for metric in g.columns:
                row=dict(zip(['split','kind','task','templates','category'],keys));row.update(metric=metric,groups=len(g),**interval(g[metric],w));qa_rows.append(row)
            qa_group[keys]=g
        # Core QA averages clean and counterfactual with equal group weight.
        core=[];paired_qa=[]
        qmain=q[q.category.isin(['clean','counterfactual'])]
        combined=[]
        for keys,frame in qmain.groupby(['split','kind','seed','task','templates'],sort=True):
            metrics=qa_summary(frame.to_dict('records'),np.stack(frame.probs.to_list()))
            combined.append(dict(zip(['split','kind','seed','task','templates'],keys))|metrics|
                            {name:float(frame[name].mean()) for name in ['coverage_0.01','size_0.01','coverage_0.05','size_0.05']})
        pd.DataFrame(combined).to_csv(out/'qa_combined_per_seed.csv',index=False)
        combined_intervals=[]
        for keys,frame in qmain.groupby(['split','kind','task','templates'],sort=True):
            g=frame.groupby(['group_id','seed'])[['correct','nll','brier','coverage_0.01','size_0.01','coverage_0.05','size_0.05']].mean().groupby('group_id').mean().sort_index()
            w=weights_for(g.index,c,cache,keys[0]);metrics={name:interval(g[name],w) for name in g.columns}
            metrics.update(nonlinear_qa_intervals(frame,w))
            for metric,values in metrics.items():
                combined_intervals.append(dict(zip(['split','kind','task','templates'],keys))|{'metric':metric,'groups':len(g)}|values)
        pd.DataFrame(combined_intervals).to_csv(out/'qa_combined_group_bootstrap.csv',index=False)
        for keys,frame in qmain.groupby(['split','kind','task','templates'],sort=True):
            g=frame.groupby(['group_id','seed','category']).correct.mean().groupby('group_id').mean().sort_index()
            w=weights_for(g.index,c,cache,keys[0]);core.append(dict(zip(['split','kind','task','templates'],keys))|interval(g,w))
        core=pd.DataFrame(core);core.to_csv(out/'qa_core.csv',index=False)
        pd.DataFrame(qa_rows).to_csv(out/'qa_group_bootstrap.csv',index=False)
        for split in ['reader_val','test']:
            for task in ['A1','A2','A3','A4']:
                for templates in ['seen','heldout']:
                    f=qmain[(qmain.split==split)&(qmain.task==task)&(qmain.templates==templates)]
                    a=f[f.kind=='CLR'].groupby(['group_id','seed','category']).correct.mean().groupby('group_id').mean().sort_index()
                    b=f[f.kind=='B3'].groupby(['group_id','seed','category']).correct.mean().groupby('group_id').mean().reindex(a.index)
                    paired_qa.append({'split':split,'task':task,'templates':templates,**interval(a-b,weights_for(a.index,c,cache,split))})
        pd.DataFrame(paired_qa).to_csv(out/'qa_CLR_minus_B3.csv',index=False)
        detection=[];rates={};aucs={}
        cats=['counterfactual','mismatch','corrupted_truncate','corrupted_zero_layer','corrupted_noise','corrupted']
        for (split,kind,k,alpha),f in d.groupby(['split','kind','k','alpha'],sort=True):
            clean=f[f.category=='clean'];groups=sorted(clean.group_id.unique());w=weights_for(groups,c,cache,split)
            clean_rate=clean.groupby('group_id').rejected.mean().reindex(groups).to_numpy(float);fp=interval(clean_rate,w)
            for category in cats:
                pos=f[f.category.str.startswith('corrupted')] if category=='corrupted' else f[f.category==category]
                rate=pos.groupby('group_id').rejected.mean().reindex(groups).to_numpy(float)
                tp=interval(rate,w);rates[(split,kind,k,alpha,category)]=(groups,rate)
                auc_key=(split,kind,k,category)
                if auc_key not in aucs:
                    matrices=[]
                    for seed in clean.seed.unique():
                        neg=clean[clean.seed==seed].set_index('group_id').score.reindex(groups).to_numpy(float)
                        sub=pos[pos.seed==seed]
                        matrices_seed=[]
                        for cat,rows in sub.groupby('category'):
                            ps=rows.set_index('group_id').score.reindex(groups).to_numpy(float)
                            matrices_seed.append((ps[:,None]>neg[None,:]).astype(float)+.5*(ps[:,None]==neg[None,:]))
                        matrices.append(np.mean(matrices_seed,axis=0))
                    matrix=np.mean(matrices,axis=0);bs=np.einsum('bi,ij,bj->b',w,matrix,w,optimize=True)/len(groups)**2
                    aucs[auc_key]={'mean':float(matrix.mean()),'lo':float(np.quantile(bs,.025)),'hi':float(np.quantile(bs,.975))}
                auc=aucs[auc_key]
                detection.append({'split':split,'kind':kind,'k':k,'alpha':alpha,'category':category,'groups':len(groups),
                                  'mean':tp['mean'],'lo':tp['lo'],'hi':tp['hi'],'FPR':fp['mean'],'FPR_lo':fp['lo'],'FPR_hi':fp['hi'],
                                  'AUROC':auc['mean'],'AUROC_lo':auc['lo'],'AUROC_hi':auc['hi']})
        detection=pd.DataFrame(detection);detection.to_csv(out/'detection_group_bootstrap.csv',index=False)
        paired=[]
        for split in ['reader_val','test']:
            for k in [1,4,16]:
                for alpha in [.01,.05]:
                    groups,a=rates[(split,'CLR',k,alpha,'counterfactual')]
                    for kind in ['B3','linear','B0','norm','text']:
                        _,b=rates[(split,kind,k,alpha,'counterfactual')]
                        paired.append({'split':split,'k':k,'alpha':alpha,'baseline':kind,**interval(a-b,weights_for(groups,c,cache,split))})
        pd.DataFrame(paired).to_csv(out/'counterfactual_CLR_paired_differences.csv',index=False)
        natural=[]
        for keys,f in nat.groupby(['split','mode','model_correct'],sort=True):
            g=f.groupby(['group_id','seed']).P_consistent.mean().groupby('group_id').mean().sort_index()
            w=weights_for(g.index,c,cache,keys[0]);natural.append(dict(zip(['split','mode','model_correct'],keys))|
                    {'groups':len(g),'q10':float(g.quantile(.1)),'median':float(g.median()),'q90':float(g.quantile(.9)),**interval(g,w)})
        natural=pd.DataFrame(natural);natural.to_csv(out/'natural_commitment_distribution.csv',index=False)
        costs=[]
        for p in Path('runs').glob('*/config.json'):
            cfg=json.loads(p.read_text())
            if cfg.get('milestone')!='M7':continue
            met=json.loads((p.parent/'metrics.json').read_text())
            costs.append({'run':str(p.parent),'status':met['status'],'gpu_hours':met.get('gpu_hours',0),
                          'source_forward_calls':met.get('source_forward_calls',0),'mode':cfg.get('mode',cfg.get('phase','controller'))})
        pd.DataFrame(costs).to_csv(out/'costs.csv',index=False);total=sum(r['gpu_hours'] for r in costs);calls=sum(r['source_forward_calls'] for r in costs)
        if total>40:raise AssertionError('M7 exceeded GPU budget')
        causal=causal_stats(c,out)
        sizes={s:sum(1 for l in (root/'source'/s/'manifest.jsonl').open()) for s in ['reader_train','reader_val','cal_A','cal_B','test']}
        data_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        checkpoints={Path(json.loads(p.read_text())['checkpoint']) for p in (root/'trained').glob('*.json')}
        checkpoint_bytes=sum(p.stat().st_size for p in checkpoints)
        source_diagnostics={}
        for split in ['reader_val','test']:
            entries=[json.loads(l) for l in (root/'source'/split/'manifest.jsonl').open()]
            entries.sort(key=lambda e:e['group_id'])
            clean=np.array([np.argmax(e['receiver_probs'][0]) for e in entries])
            cf=np.array([np.argmax(e['receiver_probs'][1]) for e in entries])
            contrast=[(e['receiver_probs'][1][e['route'][1]]-e['receiver_probs'][0][e['route'][1]])-
                      (e['receiver_probs'][1][e['route'][0]]-e['receiver_probs'][0][e['route'][0]]) for e in entries]
            source_diagnostics[split]={
                'groups':len(entries),
                'clean_accuracy':float(np.mean([p==e['route'][0] for p,e in zip(clean,entries)])),
                'counterfactual_target_accuracy':float(np.mean([p==e['route'][1] for p,e in zip(cf,entries)])),
                'counterfactual_original_target_accuracy':float(np.mean([p==e['route'][0] for p,e in zip(cf,entries)])),
                'receiver_choice_changed':float(np.mean(clean!=cf)),
                'equal_CF_token_length_fraction':float(np.mean([e['audit_meta'][0]['tokens_per_layer']==e['audit_meta'][1]['tokens_per_layer'] for e in entries])),
                'CF_token_length_delta_mean':float(np.mean([e['audit_meta'][1]['tokens_per_layer']-e['audit_meta'][0]['tokens_per_layer'] for e in entries])),
                'private_route_probability_contrast':interval(contrast,weights_for([e['group_id'] for e in entries],c,cache,split))}
        write_json(out/'source_diagnostics.json',source_diagnostics)
        mismatch_impact={}
        for split in ['reader_train','reader_val','cal_A','cal_B']:
            folder=root/'source'/split
            previous={e['id']:e for e in map(json.loads,(folder/'mismatch.jsonl').open())}
            corrected=[json.loads(l) for l in (folder/'mismatch_v2.jsonl').open()]
            diffs=[np.abs(np.array(e['receiver_probs'])-previous[e['id']]['receiver_probs']) for e in corrected]
            mismatch_impact[split]={'groups':len(corrected),'max_abs_receiver_probability':float(np.max(diffs)),
                'receiver_argmax_changed_fraction':float(np.mean([np.argmax(e['receiver_probs'])!=np.argmax(previous[e['id']]['receiver_probs']) for e in corrected]))}
        write_json(out/'mismatch_v2_impact.json',mismatch_impact)
        learn_pointer=json.loads((root/'learnability.json').read_text())
        learn={kind:json.loads(Path(learn_pointer['run'],kind+'_result.json').read_text()) for kind in ['CLR','B3']}
        write_json(out/'learnability_evidence.json',{'run':learn_pointer['run'],'results':learn})
        def det(split,kind):return detection[(detection.split==split)&(detection.kind==kind)&(detection.k==16)&(detection.alpha==.05)&(detection.category=='counterfactual')].iloc[0].to_dict()
        def qa(split,kind,task,templates='seen'):return core[(core.split==split)&(core.kind==kind)&(core.task==task)&(core.templates==templates)].iloc[0].to_dict()
        table1='|方法|reader_val 检测率% [CI] / FPR% / AUROC|test 检测率% [CI] / FPR% / AUROC|\n|---|---|---|\n'
        for kind in ['CLR','B3','linear','B0','norm','text']:
            cells=[]
            for split in ['reader_val','test']:
                r=det(split,kind);cells.append(f"{ci_cell(r)} / {fmt(100*r['FPR'])} / {fmt(r['AUROC'])}")
            table1+=f'|{kind}|'+ '|'.join(cells)+'|\n'
        table2='|问答|reader_val CLR / B3 %|test CLR / B3 %|test held-out CLR / B3 %|\n|---|---|---|---|\n'
        for task in ['A1','A2','A3','A4']:
            cells=[]
            for split,tmpl in [('reader_val','seen'),('test','seen'),('test','heldout')]:cells.append(' / '.join(fmt(100*qa(split,k,task,tmpl)['mean']) for k in ['CLR','B3']))
            table2+=f'|{task}|'+ '|'.join(cells)+'|\n'
        table3='|自然承诺|reader_val n / 均值 [10%,90%]|test n / 均值 [10%,90%]|\n|---|---|---|\n'
        for mode in ['direct','cot512']:
            for correct in [True,False]:
                cells=[]
                for split in ['reader_val','test']:
                    f=natural[(natural.split==split)&(natural['mode']==mode)&(natural.model_correct==correct)]
                    if len(f):
                        r=f.iloc[0];cells.append(f"{int(r['groups'])} / {fmt(r['mean'])} [{fmt(r['q10'])},{fmt(r['q90'])}]")
                    else:cells.append('0 / NA')
                table3+=f"|{mode} {'正确' if correct else '错误'}|"+'|'.join(cells)+'|\n'
        differences=pd.DataFrame(paired_qa);consistency=[]
        positive_transfer=[]
        for task in ['A1','A4']:
            r=differences[(differences.split=='test')&(differences.task==task)&(differences.templates=='seen')].iloc[0]
            direction='正向且CI排除0' if r['lo']>0 else '反向且CI排除0' if r['hi']<0 else '未建立正向优势'
            consistency.append(f"{task} CLR−B3={ci_cell(r.to_dict())}pp，"+direction)
            positive_transfer.append(r['lo']>0)
        transfer_conclusion='两类方向与GSM8K一致；范围限于当前缩量设置' if all(positive_transfer) else '未复现GSM8K的双任务正向优势'
        review=f'''### (a) 状态一句话
M7 非自适应实验完成待审查；存储约束下缩量，历史raw KV逐张量对照缺证据；未启动M8。

### (b) 关卡清单
- 环境通过：独立.venv-comm覆盖层，只读继承Conda依赖；transformers4.57.1/tokenizers0.22.2/peft0.17.1与Idea A一致，主环境未改；生成器3源码SHA锁定、只读import。
- 兼容性通过：历史8题receiver logits复放、冻结adapter消息重建及新raw KV两路径误差均0、一致率100%。历史raw消息张量对照未通过（原run未保存，不能用新复放冒充）。
- 传信通过：固定短日志val80 message/none/mismatch={fmt(100*causal['message']['mean'])}/{fmt(100*causal['none']['mean'])}/{fmt(100*causal['mismatch']['mean'])}%；配对增益={ci_cell(causal['gain_vs_none'])}/{ci_cell(causal['gain_vs_mismatch'])}pp。正式reader_val只改私有变量的目标概率对比={ci_cell(source_diagnostics['reader_val']['private_route_probability_contrast'])}pp。
- smoke通过：源val2约14秒；训练train4/val代理2；评估仅val代理；单测16项通过。错配v2另用val2做13.2秒smoke，张量/metadata一致。最终锁后才做正式检测/自然承诺分析；test无开发访问。
- 规模目标未通过：实际train/val/cal_A/cal_B/test={sizes['reader_train']}/{sizes['reader_val']}/{sizes['cal_A']}/{sizes['cal_B']}/{sizes['test']}基础episode；每组含真/反事实KV，单层35、全部有效token、fp16。按用户15GiB规则缩量。α=.01因cal_B仅29组阈值为∞。
- 主比较通过执行：CLR/B3/B0/线性各3seed；范数/文本为确定性拟合，不伪造seed方差。成本{fmt(total)} GPUh≤40，源前向{fmt(calls)}次（含smoke/失败/诊断）。

### (c) 核心主表（最多3张）
主检测：counterfactual，k=16，cal_B clean定名义α=.05；同时列实测FPR，未按test重调到5%。95%CI为3seed均值条件于固定模型/校准/抽题，按group bootstrap10000次；确定性基线按组CI。

{table1}
QA为clean/counterfactual等权；完整A1–A4指标/组CI及k=1/4/16、所有类别见CSV。A3 AUROC在clean+CF合并后计算。

{table2}
附加natural_commitment：clean KV+模型承诺，A3 P(一致)跨模板/seed先在episode内平均，列跨episode均值及10–90%分位；不进主训练/指标/阈值。

{table3}
### (d) 本里程碑新增的偏离与阻塞
用户明确采用所有类别同模板的程序化承诺（≥3变体、commitment_source=programmatic）。sender原direct33.8%为val80、cot51250.0%为val8；样本量不同。部署模型承诺存在差距；主文本基线在CF上只能靠receiver变化。
15GiB导致缩量和仅1层，1%门限无检测能力；历史raw张量缺失。CF仅干预语义变量e，可改变多条支持记录中的团队字段，不是单原始事件编辑，也不保证token长度/范数匹配。QA标签表示生成器来源事实，不证明所有事实均成功编码。conformal仅在clean校准/未来消息可交换等假设下给边际控制，经验FPR不受逐批硬上界保证。
学习消息未接（M7准入时Idea A正式val未完成）；C2C未接；HiddenBench已核实可得/格式，主比较仅synthetic。评估smoke的libstdc++启动失败与修复见BLOCKERS，未换模型/设置。
正式数据错配receiver原先单条重编码供体，与保存时原/CF批次的padding位置不一致；test前重算为v2，逐张量/metadata等值断言通过。旧输出保留；主QA训练不含mismatch，正式检测/文本拟合只用v2。

### (e) 需要审查者决定的问题
无。对用户问题：(1)机制复放/新raw数值与传信因果检验通过，历史raw逐张量对照缺证据；(2)CF检测率与实测FPR见主表；(3){'；'.join(consistency)}；{transfer_conclusion}。A1对应GSM8K Q4语义判断、A4对应Q3选择。

### (f) 证据路径
详细：reports/M7_details.md；CSV/图：data/M7_v1/analysis/；训练与最终评估：data/M7_v1/trained/、evaluated/；最终锁：data/M7_v1/final_lock.json；冻结：data/M7_v1/freeze.json；接续：docs/M7_LIVE.json。完整命令、源码快照、环境/成本均在CSV索引的runs/中。
'''
        if len(review.encode())>8000:raise AssertionError('M7 review exceeds 8000 UTF8 bytes')
        Path(c['review']).write_text(review)
        details=f'''# M7 潜通信与语义审计详细证据

主任务与用户commitment修订：docs/M7_TASK.md。审查文件为reports/M7_REVIEW.md。

{table1}\n{table2}\n{table3}

统计：按group整体重采样10000次，取2.5%/97.5%分位作95%CI；先在每个seed/组内平均问题，再跨3seed平均。CI条件于这3个训练模型及冻结校准集，未包含重新训练与重新抽取校准集的不确定性。AUROC按同一组的clean/替换消息配对bootstrap；确定性文本/范数只拟合一次。校准用cal_A按问题类型温度缩放（独立组、seen模板、clean/CF预先随机混合）；cal_B每组一clean消息/问答定阈值；held-out两模板未用于训练、选模型或校准。QA集合控制只针对与cal_B抽样规则可交换的clean任务实例；CF/mismatch/held-out仅报告经验覆盖率。

一致性目标只由可见承诺与公开T、z约束推出；k个问题在4种seen改写中按固定seed随机抽取（有放回），并不独立，不声称指数检测界。random k使用同一16题序列的前k题，所有方法配对一致；2种held-out改写另做QA迁移，不用于主检测阈值。

审计z：Qwen3-4B同权重顺序sender/receiver，raw sender读x_pub+x_priv，receiver只读x_pub+全层KV；K已post-RoPE，prefix槽含mask/注入层/门控。审计仅保存末层35的每个有效token，拼K各头再V各头，fp16。适配器按token投影并以6个query读取，再复用冻结M4.1 Lite实现的4层256 reader；原M4.1源码未改。

训练：CLR/B0/B3/线性各seed11/22/33，完整曝光上限20 epoch、min4、patience3，reader_val四类平均NLL选checkpoint；损失按类型频率加权。B3均值池化KV与问题/选项交互MLP；线性为条件于问题/选项的线性状态分数。B0固定6个零潜向量，去掉长度信息。范数只读保存层K/V log-RMS；文本只读同格式承诺和receiver选择输出，不看x_pub/x_priv。
B0正式训练前移除完全未使用的长KV归一化/搬运；相同零前缀送入相同主体。原/新A1/A3/A4逐位输出相等、max误差0（reports/M7_evidence/b0_equivalence.log）；科学设置不变，含错配位置、统计边界及生成器独立哈希锚测试共16项通过（tests_v12.log）。
GPU2/GPU7从独立固定队列改为共享待办队列（configs/M7/balanced_training.yaml）；交接时已运行子作业继续正常结束。旧CPU调度器随后标记interrupted/superseded，成本0；这不是训练失败或重启。实际设备、启动代码、完整科学配置仍逐run保存。
最终控制器configs/M7/finish_v2.yaml等待全部训练及错配v2完成才校准锁；锁后reader_val评估与test生成在GPU7/GPU2并行，二者完成后才评估test。每次test访问前仍逐项核验最终锁。状态写入使用M7专属文件锁，避免并行阶段争用临时文件。

programmatic承诺三模板共享，类别间复用同一原episode文本；CF保持公开输入/无关日志，只交换目标记录中的团队字段来干预e。mismatch来自同split下一独立组；corrupted三控制分别截断全prefix前半、将保存层全置零、保存层相对L2=.1高斯噪声。receiver实际消费每种控制，审计读取其真实扰动KV（bf16再转fp16），不是用加噪输入代替源输出。噪声固定seed，非训练攻击。
原mismatch.jsonl的receiver对供体单条重编码，而审计保存供体与CF共同左padding的KV；开发580组中118个原供体有非零左pad。已保留旧版并创建mismatch_v2.jsonl：重新构造同一原/CF批次，选第一条时保留物理槽，逐张量/metadata精确比对后供receiver使用。修正仅在test前，主训练不含mismatch；正式文本拟合、检测与mismatch问答全部使用v2。全量修正证据data/M7_v1/mismatch_v2_complete.json，概率/选择变化见analysis/mismatch_v2_impact.json；最终test直接生成v2。噪声半径.1为目标值，bf16/fp16转换有舍入误差，不声称严格浮点上界。

QA主表为clean/CF等权；完整CSV另含mismatch，标签取自供体，A1以split内无同值的边际匹配排列生成假事实。破坏性corrupted不存在可靠的“信息仍包含”标签，因此只作为非语义检测控制。A1对真/反事实两消息使用互换的正负事实，取值边际精确对齐。当前A1只审计唯一四值语义变量e，不代表任意事实开放域读取。
mismatch不强制改变e或route，其中一部分仍支持原可见结论；QA按实际供体标记，检测类别按“替换了来源episode”定义，不能把全部mismatch都称为语义矛盾。供体在split内循环复用，其组bootstrap区间按固定供体面板解释，不作为独立重抽供体的置信保证。
manifest的commitments[1]记录CF来源episode自身的结论元信息；主counterfactual类实际公开文本始终是commitments[0]，与clean/mismatch/corrupted完全相同。

程序化承诺定义来自solve_v2_text原文求解，不向Qwen/reader传入label/e/meta；约束目标可从公开T,z和承诺计算。自然模型承诺另存source/<split>/natural_commitment.jsonl，主训练和阈值入口从未读取该文件；只在锁定后做附加分析，所有val/test基础episode的direct/cot512原始生成、预测和正误齐全。
模型自然承诺取冻结S2接口的direct或cot512最终选项打分argmax，经同一formatter转为短承诺；cot512完整token/原文另存，不把自由生成原文中的任意句子当作已核验的最终选项。

数据：新seed20261106直接import冻结m1-v2-r2生成器；保持原arithmetic_wide/priv_len12,48,128/四类事实/步骤0,2,4。每个基础组选择alpha1及按组索引轮换d_pub，不跨split，CF共享组ID。目标规模1层约179GiB，按15GiB规则缩为{sizes}。data/M7_v1当前{fmt(data_bytes/2**30)}GiB，含真/CF实际KV与统计；另有run内checkpoint合计{fmt(checkpoint_bytes/2**30)}GiB，不计入KV存储。未通过原8k/1k/1k/2k规模目标。

历史兼容核验不是借用Idea A结论：在.venv-comm重新加载4B，冻结历史8个episode/seeded messages/adapter，重建每层消息与历史receiver logits逐项核对；原始KV另与Idea A的冻结提取实现比较，误差0。历史raw中继张量未保存，所以该历史比对要求仍缺证据，不把新复放冒充历史输出。Idea A的学习式消息和简化KVTransfer没有冒充官方C2C。

真实KV的4个训练组过拟合诊断：CLR四类100%，B3四类最低95.3%；run={learn_pointer['run']}，完整指标见analysis/learnability_evidence.json。该诊断只验证路径具有学习能力，没有替换正式超参或选seed。正式source的clean/CF目标准确率与选择变化见analysis/source_diagnostics.json；干预只改私有语义变量，接收者公开输入不变。开发前上游冻结23800文件校验无差异，证据data/M7_v1/upstream_pretest_verification.json；收尾另重新核验。

私有干预的目标概率对比定义为[P_cf(y_cf)−P_clean(y_cf)]−[P_cf(y_clean)−P_clean(y_clean)]，按组bootstrap；在最终test前登记该附加因果统计。它隔离私有事实改变的影响，区别于同时移除/错配全部公开与私有KV的固定val80对照；val80限定alpha1、d_pub0、priv_len12，不外推其准确率到长日志。

与GSM8K的对应：M4.1冻结test上Q3 CLR−B3为+17.5pp [16.1,19.0]、Q4为+7.93pp [6.31,9.56]。M7的A4对应Q3、A1对应Q4，比较必须以analysis/qa_CLR_minus_B3.csv的配对组区间为准，不能把小样本中的方向当作重现。

成本{fmt(total)} GPUh，包含所有M7记录的源/训练/评估smoke、失败和natural诊断；逐进程按单设备驻留墙钟保守累加，GPU共享/进程重叠也不抵扣，不按利用率折算。controller/纯CPU统计为0GPUh；启动import失败无CUDA模型调用且未创建run，保留原始错误日志。source_forward_calls={calls}含batched调用，forward_examples与prefill令牌另见run；不得把调用数当样本数。

证据：qa_per_seed.csv（按类别）、qa_combined_per_seed.csv（clean+CF，含二分类AUROC/ECE；单类别A3只有一个标签故AUROC为NA）、qa_combined_group_bootstrap.csv（含AUROC/ECE的非线性组重采样CI）、qa_core.csv、qa_group_bootstrap.csv、qa_CLR_minus_B3.csv；detection_group_bootstrap.csv、counterfactual_CLR_paired_differences.csv；natural_commitment_distribution.csv；causal_bootstrap.json、costs.csv。真实run索引trained/、evaluated/、final_lock.json；源码/环境在run快照。可复现入口configs/M7/下所有YAML和scripts/m7_*.py。

![Counterfactual detection versus question count](../data/M7_v1/analysis/counterfactual_detection.png)
图中阈值均来自cal_B clean、名义α=.05；实测FPR见主表，阴影为条件于固定模型和校准的95%组bootstrap区间。
'''
        Path(c['details']).write_text(details)
        # Static publication artifacts; no interactive plugin needed.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        plot_max=10.
        for kind in ['CLR','B3','norm','text']:
            f=detection[(detection.split=='test')&(detection.kind==kind)&(detection.alpha==.05)&(detection.category=='counterfactual')].sort_values('k')
            ax.plot(f.k,100*f['mean'],marker='o',label=kind);ax.fill_between(f.k,100*f.lo,100*f.hi,alpha=.12)
            plot_max=max(plot_max,float(100*f.hi.max())*1.1)
        ax.set(xlabel='Number of audit questions k',ylabel='Counterfactual detection (%)',xticks=[1,4,16],ylim=(0,min(100,plot_max)));ax.legend();fig.tight_layout()
        fig.savefig(out/'counterfactual_detection.png',dpi=180);fig.savefig(out/'counterfactual_detection.pdf');plt.close(fig)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=True,total_gpu_hours=total,
                          source_calls=calls,review_utf8_bytes=len(review.encode()),sizes=sizes,review=c['review'])
        write_json(root/'report.json',{'run':str(run.path),'review':c['review'],'total_gpu_hours':total,
                   'primary_counterfactual_k16_alpha005':{split:{kind:det(split,kind) for kind in ['CLR','B3','norm','text']}
                                                         for split in ['reader_val','test']},
                   'qa_CLR_minus_B3_seen':{task:next(r for r in paired_qa if r['split']=='test' and r['task']==task and r['templates']=='seen')
                                           for task in ['A1','A4']},
                   'causal_short_val80':causal,'source_diagnostics':source_diagnostics,'sizes':sizes,
                   'review_utf8_bytes':len(review.encode())})
        live(stage='report_completed_freeze_pending',report_run=str(run.path),review=c['review'],gpu_hours=total)
        print(c['review'],len(review.encode()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

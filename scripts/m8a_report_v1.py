"""Post-lock statistics and review, without altering inference or calibration."""
import sys
sys.dont_write_bytecode=True
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.audit.features import sha256
from clr.audit.m71_storage import load_bundle
from clr.audit.m8a_runtime import snapshot,Cost,total_cost,live
from clr.audit.m8a_statistics_v1 import select_candidate,metric_vector,bootstrap_mean
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def fmt(x):
    return f'{x:#.3g}'


def attack_trace(row):
    return {'actual_steps':row['steps'],'early_success':row['early_success'],
            'max_relative_norm':row['max_relative_norm'],'lambda':row['lambda'],
            'attack_seed':row['attack_seed']}


def summarize(rows,repeats):
    out=[]
    for (kind,epsilon,budget,alpha),rr in pd.DataFrame(rows).groupby(['kind','epsilon','step_budget','alpha'],dropna=False,sort=False):
        for metric in ['robust_failure','target_and_accepted','target_rate','changed_target_rate','detection_rate','clean_FPR']:
            per_group=rr.groupby('group_id')[metric].mean()
            mean,lo,hi=bootstrap_mean(per_group.to_numpy(),repeats)
            out.append({'kind':kind,'epsilon':epsilon,'step_budget':budget,'alpha':alpha,'metric':metric,
                'mean':mean,'lo':lo,'hi':hi,'groups':len(per_group)})
        correct=rr[rr.clean_correct==1]
        if len(correct):
            per_group=correct.groupby('group_id').robust_failure.mean()
            mean,lo,hi=bootstrap_mean(per_group.to_numpy(),repeats)
            out.append({'kind':kind,'epsilon':epsilon,'step_budget':budget,'alpha':alpha,'metric':'robust_failure_given_clean_correct',
                'mean':mean,'lo':lo,'hi':hi,'groups':len(per_group)})
    return pd.DataFrame(out)


def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);out=root/'analysis_v1';out.mkdir(exist_ok=False)
        lock=json.loads((root/'final_lock.json').read_text())
        for path,digest in lock['sha256'].items():assert sha256(path)==digest,path
        raw=[];runs=[]
        for part in range(c['parts']):
            d=json.loads((root/f'full_part{part}.json').read_text());r=Path(d['run']);runs.append(str(r))
            assert d['test_opened'] and json.loads((r/'metrics.json').read_text())['status']=='completed'
            rows=[json.loads(x) for x in (r/'results.jsonl').read_text().splitlines()]
            assert len(rows)==124*d['groups'];raw.extend(rows)
        ids=json.loads((root/'test500_ids.json').read_text())['group_ids']
        assert len(raw)==62000 and {r['group_id'] for r in raw}==set(ids)
        assert set(collections.Counter(r['group_id'] for r in raw).values())=={124}
        clean_groups={}
        for row in raw:
            baseline=(row['clean_pred'],row['target'],row['clean_correct'])
            assert clean_groups.setdefault(row['group_id'],baseline)==baseline
        clean_receiver_accuracy=float(np.mean([correct for pred,target,correct in clean_groups.values()]))
        preexisting_target_rate=float(np.mean([pred==target for pred,target,correct in clean_groups.values()]))
        keys=[(r['group_id'],r['kind'],r['epsilon'],r['step_budget'],r['attack_seed'],r['lambda']) for r in raw]
        assert len(set(keys))==len(keys)
        for row in raw:
            if row['epsilon'] is not None:assert row['max_relative_norm']<=row['epsilon']*(1+1e-5)
            if row['kind'] in ['E1','E2']:
                assert row['feature_difference']['fp16_exact_fraction']==1
                for seed in [11,22,33]:
                    assert row['audit'][str(seed)]==row['clean_audit'][str(seed)],'Feature-invariant detector score changed'
        expanded=[];candidates=collections.defaultdict(list)
        for r in raw:
            if r['kind']=='E3':candidates[(r['group_id'],r['epsilon'],r['step_budget'],r['attack_seed'])].append(r)
            else:
                for seed in [11,22,33]:
                    for alpha in [.01,.05]:
                        expanded.append({'group_id':r['group_id'],'kind':r['kind'],'epsilon':r['epsilon'],'step_budget':r['step_budget'],
                            'alpha':alpha,'reader_seed':seed,**attack_trace(r),**metric_vector(r,seed,alpha)})
        # White-box portfolio of all three locked lambda candidates. E4
        # uses only lambda1 and a fixed independent surrogate mapping.
        transfers={11:22,22:33,33:11};chosen=[]
        for (gid,epsilon,budget,seed),rr in candidates.items():
            assert {r['lambda'] for r in rr}=={.1,1,10}
            chosen.append(select_candidate(rr,seed))
            for r in rr:
                for alpha in [.01,.05]:
                    expanded.append({'group_id':gid,'kind':f'E3_lambda{r["lambda"]:g}','epsilon':epsilon,'step_budget':budget,'alpha':alpha,
                        'reader_seed':seed,**attack_trace(r),**metric_vector(r,seed,alpha)})
            selected=select_candidate(rr,seed)
            for alpha in [.01,.05]:
                expanded.append({'group_id':gid,'kind':'E3','epsilon':epsilon,'step_budget':budget,'alpha':alpha,'reader_seed':seed,
                    **attack_trace(selected),**metric_vector(selected,seed,alpha)})
            for target,surrogate in transfers.items():
                if surrogate!=seed:continue
                transfer=next(r for r in rr if r['lambda']==1)
                for alpha in [.01,.05]:
                    expanded.append({'group_id':gid,'kind':'E4','epsilon':epsilon,'step_budget':budget,'alpha':alpha,'reader_seed':target,
                        **attack_trace(transfer),**metric_vector(transfer,target,alpha)})
        frame=pd.DataFrame(expanded);frame.to_csv(out/'group_metrics.csv.gz',index=False,compression='gzip')
        summary=summarize(expanded,c['bootstrap_repeats']);summary.to_csv(out/'robustness_summary.csv',index=False)
        frame.groupby(['kind','epsilon','step_budget','alpha','reader_seed'],dropna=False).mean(numeric_only=True).to_csv(out/'seed_metrics.csv')
        diagnostics=[]
        for kind in ['E1','E2','E3']:
            rr=[r for r in raw if r['kind']==kind and r['step_budget']==200]
            diagnostics.append({'kind':kind,'rows':len(rr),'mean_actual_steps':float(np.mean([r['steps'] for r in rr])),
                'max_actual_steps':max(r['steps'] for r in rr),'early_stop_rate':float(np.mean([r['early_success'] for r in rr])),
                'feature_fp32_max_error':max(r['feature_difference']['maximum_absolute'] for r in rr),
                'feature_fp16_min_exact_fraction':min(r['feature_difference']['fp16_exact_fraction'] for r in rr)})
        write_json(out/'numerics_and_steps.json',diagnostics)
        # Group-paired P0 diagnostics, jointly averaging original/CF and seeds.
        p0_rows=[]
        for seed in [11,22,33]:
            ptr=json.loads(Path(f'data/M8_v1/P0_full_{seed}.json').read_text())
            for split in ['reader_val','test']:
                for mode in ['full','zero','shuffle','value_only','value_remove','critical_only','critical_remove']:
                    groups=load_bundle(Path(ptr['run'])/f'{split}_{mode}.pt.gz')['groups']
                    p0_rows.extend({**r,'seed':seed,'split':split,'mode':mode} for r in groups)
        p0=pd.DataFrame(p0_rows);p0_summary=[]
        for (split,mode,task),rr in p0.groupby(['split','mode','task']):
            by_group=rr.groupby('group_id').accuracy.mean();mean,lo,hi=bootstrap_mean(by_group.to_numpy(),c['bootstrap_repeats'])
            p0_summary.append({'split':split,'mode':mode,'task':task,'mean':mean,'lo':lo,'hi':hi,'groups':len(by_group)})
        pd.DataFrame(p0_summary).to_csv(out/'P0_QA.csv',index=False)
        b0=json.loads((root/'P0_B0.json').read_text());pd.DataFrame(b0).to_csv(out/'P0_trained_B0.csv',index=False)
        selection=[]
        for seed in [11,22,33]:
            trained=json.loads(Path(f'data/M71_v1/pool/training/CLR_{seed}.json').read_text())
            reference=json.loads(Path(f'data/M71_v1/pool/training/B3_{seed}.json').read_text())
            history=json.loads((Path(trained['run'])/'history.json').read_text())
            eligible=[row for row in history if row['exposure']>=reference['exposure']]
            best=min(eligible,key=lambda row:row['selection_nll'])
            assert best['epoch']==trained['selected']['epoch']
            selection.append({'seed':seed,'minimum_B3_exposure':reference['exposure'],
                'unrestricted_best_epoch':min(history,key=lambda row:row['selection_nll'])['epoch'],
                'eligible_epochs':[row['epoch'] for row in eligible],'selected_epoch':best['epoch'],
                'selected_NLL':best['selection_nll'],'model_changed':False})
        write_json(out/'seed_selection_audit.json',selection)
        # Costs include the historical prerequisite and all smoke/failed work.
        costs=[]
        for p in Path('runs').glob('*/config.json'):
            if p.parent.resolve()==run.path.resolve():continue
            cfg=json.loads(p.read_text())
            if cfg.get('milestone') not in ['M8','M8a']:continue
            mpath=p.parent/'metrics.json'
            if not mpath.exists():raise ValueError('Missing cost record: '+str(p))
            m=json.loads(mpath.read_text())
            if cfg.get('mode')=='finalize' and m['status']=='running' and m.get('gpu_hours',0)==0:
                continue  # Active CPU-only parent is closed after this report.
            if m['status'] not in ['completed','failed']:raise ValueError('Still active cost record: '+str(p))
            costs.append({'run':str(p.parent),'mode':cfg.get('mode'),'GPUh':m.get('gpu_hours'),
                'source_forward_calls':m.get('source_forward_calls'),'status':m['status']})
        pd.DataFrame(costs).to_csv(out/'costs.csv',index=False)
        hours=sum(r['GPUh'] for r in costs);assert hours<=25
        # Display test epsilon curves only; val20 is explicitly a smoke panel.
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        for kind in ['E1','E2','E3','E4']:
            for alpha,ax in zip([.05,.01],axes):
                rr=summary[(summary.kind==kind)&(summary.step_budget==200)&(summary.alpha==alpha)&(summary.metric=='robust_failure')&summary.epsilon.notna()].sort_values('epsilon')
                ax.plot(rr.epsilon,100*rr['mean'],marker='o',label=kind)
                ax.fill_between(rr.epsilon,100*rr.lo,100*rr.hi,alpha=.1)
                ax.set(xlabel='Per-layer relative L2 epsilon',ylabel='Changed-to-target and accepted (%)',title=f'Nominal clean alpha={alpha:g}',xscale='log',ylim=(0,100))
                ax.grid(alpha=.2)
        axes[0].legend();fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(out/f'epsilon_failure.{suffix}',dpi=180)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,3.5))
        for eps in [.01,.03,.1,.3]:
            rr=summary[(summary.kind=='E3')&(summary.epsilon==eps)&(summary.alpha==.05)&(summary.metric=='robust_failure')].sort_values('step_budget')
            ax.plot(rr.step_budget,100*rr['mean'],marker='o',label=str(eps))
        ax.set(xlabel='PGD step budget (early stopping allowed)',ylabel='Robust failure (%)');ax.legend(title='epsilon');fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(out/f'steps_failure.{suffix}',dpi=180)
        plt.close(fig)
        result={'run':str(run.path),'analysis':str(out),'rows':len(raw),'groups':500,'GPUh':hours,
            'clean_receiver_accuracy':clean_receiver_accuracy,'preexisting_target_rate':preexisting_target_rate,
            'source_forward_calls':sum(r['source_forward_calls'] or 0 for r in costs),'runs':runs,
            'metric':'changed_to_target AND not detected, denominator all500; target_and_accepted separately includes preexisting target predictions',
            'selection':'E3 target hits prioritized, then highest receiver target logp among3 lambdas; reporting audit panel not used; same candidate at both alphas; E4 fixed cross-seed lambda1',
            'bootstrap':'10000 original/CF group draws, average seeds within group, conditional on frozen models/calibration/question panels/attack search',
            'numerics':diagnostics}
        write_json(out/'completion.json',result)
        write_review(root,out,result,summary,p0_summary,b0)
        cost.update(status='completed');live(stage='statistics_completed_pending_freeze',report='reports/M8a_REVIEW.md',analysis=str(out),gpu_hours=hours)
        print(json.dumps(result),flush=True)


def write_review(root,out,result,summary,p0,b0):
    def val(kind,eps,alpha,metric='robust_failure',budget=200):
        rr=summary[(summary.kind==kind)&(summary.alpha==alpha)&(summary.metric==metric)&(summary.step_budget==budget)]
        rr=rr[rr.epsilon.isna()] if eps is None else rr[rr.epsilon==eps]
        assert len(rr)==1,(kind,eps,alpha,metric,budget)
        return rr.iloc[0]
    def pct(kind,eps,alpha,metric='robust_failure',budget=200):return fmt(100*val(kind,eps,alpha,metric,budget)['mean'])
    def qa(split,mode,task):return next(r for r in p0 if r['split']==split and r['mode']==mode and r['task']==task)
    text='### (a) 状态一句话\nM8a固定500组非自适应/白盒鲁棒性评估及P0完成、待审查；M7.1模型/阈值不改。\n\n### (b) 关卡清单\n'
    text+='- 通过：P0的A1/A4期望=inverse(公开T[z],commitment路由)，A1再与所问team比较；A2对承诺路由期望Yes，A3期望一致。z仅解公开x_pub。cal_B1000及val/test3000组文本核验一致，无私有事实依赖，不需重定阈值；M7.1 CF检测仍93.1%（名义5%）。\n'
    text+='- 通过：固定val20 smoke均≤900秒；44项协议测试及4项统计测试；500组SHA排序ID及模型/阈值/代码锁后才执行新扰动。三seed，ε=.01/.03/.1/.3/无约束，λ=.1/1/10，50/200预算。\n'
    text+='- E1读取特征逐位相同；E2为段内非tail成对零和子空间，真实BF16扰动后fp16 reader输入逐位相同。FP32池化舍入残差见(d)，不宣称浮点f每位相同。\n'
    text+=f'- 成本{fmt(result["GPUh"])}GPUh≤25，源前向{fmt(result["source_forward_calls"])}次；含旧P0、smoke/失败，按共享GPU进程驻留求和。\n\n### (c) 核心主表（最多3张）\n'
    text+='QA为原/CF组及三seed等权，P0完整reader_val1000/test2000的已见模板；定位仅A4。B0为另训z=0模型，zero为冻结CLR输入置零。\n\n|干预|reader_val A1/A4 %|test A1/A4 %|\n|:---|:---|:---|\n'
    for mode,label in [('full','完整CLR'),('zero','CLR置零'),('shuffle','shuffled-z'),('value_only','仅值记录段'),('value_remove','去掉值记录段')]:
        vals=[]
        for split in ['reader_val','test']:
            a1=fmt(100*qa(split,mode,'A1')['mean']) if mode in ['full','zero','shuffle'] else '—'
            vals.append(a1+'/'+fmt(100*qa(split,mode,'A4')['mean']))
        text+=f'|{label}|{vals[0]}|{vals[1]}|\n'
    for split in ['reader_val','test']:
        a=[r for r in b0 if r['split']==split]
        text+=f'另训B0 {split}：A1={fmt(100*np.mean([r["accuracy"] for r in a if r["task"]=="A1"]))}%，A4={fmt(100*np.mean([r["accuracy"] for r in a if r["task"]=="A4"]))}%。'
    text+=f'\n\nclean receiver准确率{fmt(100*result["clean_receiver_accuracy"])}%，原本已输出CF目标{fmt(100*result["preexisting_target_rate"])}%。'
    text+='主指标=receiver由非目标变为CF目标且未报警，分母固定500；另列包含原本已输出目标的“目标且接受”，不混同因果改变。下表为200步预算的三seed均值，单元格是名义5% / 1%阈值下鲁棒失效率%；阈值只由clean cal_B确定。\n\n|ε|E1|E2|E3（3λ有限搜索）|E4迁移|\n|:---|:---|:---|:---|:---|\n'
    for eps in [.01,.03,.1,.3,None]:
        text+='|'+('无约束经验' if eps is None else str(eps))+'|'+ '|'.join(pct(k,eps,.05)+' / '+pct(k,eps,.01) for k in ['E1','E2','E3','E4'])+'|\n'
    text+='\nε=.03细分（检测率作用于全部扰动消息，非仅目标成功子集）；95%CI按组bootstrap10000次、固定模型/校准/攻击搜索条件下，seed先组内平均。完整逐ε/λ/seed、50/200及clean-correct条件指标见CSV。\n\n|类型|目标达成%|鲁棒失效5% [95%CI]|识别5% / 1%|实测clean FPR5% / 1%|\n|:---|:---|:---|:---|:---|\n'
    for kind,eps,budget in [('E1',.03,200),('E2',.03,200),('E3',.03,200),('E4',.03,200),('E0_CF',None,0),('E0_noise',.03,0),('E0_diffmean',.03,0)]:
        v=val(kind,eps,.05,budget=budget)
        text+=f'|{kind}|{pct(kind,eps,.05,"target_rate",budget)}|{fmt(100*v["mean"])} [{fmt(100*v.lo)},{fmt(100*v.hi)}]|{pct(kind,eps,.05,"detection_rate",budget)} / {pct(kind,eps,.01,"detection_rate",budget)}|{pct(kind,eps,.05,"clean_FPR",budget)} / {pct(kind,eps,.01,"clean_FPR",budget)}|\n'
    text+='\n### (d) 本里程碑新增的偏离与阻塞\n'
    text+='seed33选epoch6/9；曝光匹配要求≥2.30e6使全曲线最优epoch3不可选，合格epoch6–9中选6。A3 val NLL=.860（11/22约.694），A1/A4已近99%。新增cal_B诊断发现训练A1负例只覆盖e±1；对未见e+2负例，三seed准确率1.60/2.90/.800%，seed33的NLL=71.0、96.4%触发截断，推高其k16门限至5.49（另两seed1.68/1.28）。这是负例分布/置信度问题的证据，不把曲线相关性写成唯一根因；未改模型/温度/阈值。\n'
    d=next(x for x in result['numerics'] if x['kind']=='E2')
    text+=f'E2数学零和成立；冻结FP32 GEMM仍有最大{fmt(d["feature_fp32_max_error"])}舍入残差。丢弃会改变fp16输入的记录更新，故实际reader输入精确不变；只探索完整f零空间的一个子空间。投影/量化使用STE求梯度，最终约束检查作用于真实BF16张量。定位使用生成器记录位置仅作授权诊断；KV上下文与删段分布变化使其不能单独证明token因果。\n'
    text+='文献基线据原文核实：2606.28958缩放8/符号翻转/范数匹配随机替换；2605.28214 Eq15 DiffMean从val20中7个clean-correct/文本干扰wrong配对提取，token均值广播。原文是latent-thought KV，本项目是完整prefill KV，均为适配，未逐项复现原拓扑、PCA/RePS与原始数据。E4使用独立训练seed作替代者，固定循环映射与λ=1，只对替代者求梯度/停止。\n'
    text+='成功早停保留同一扰动到后续预算，实际步数见CSV；EOT为4个预采样独立问题面板轮转，停止检查下一EOT面板。E3主表在3个锁定λ中优先目标命中、再按receiver目标logp最高选候选；报告抽题不参与优化/停止/候选选择，两α共用候选。汇总规则在锁后统计前写明；全部候选成本计入，单λ另报。有限PGD是最坏失效率的经验下界；无约束不是证明上界，conformal不保证自适应攻击覆盖率。\n\n### (e) 需要审查者决定的问题\n无。P0通过且93.1%无需重算；E1/E2及白盒随ε结果见主表。\n\n### (f) 证据路径\nreports/M8a_details.md、M8a_method_notes.md；data/M8a_v1/analysis_v1/（robustness_summary.csv、group_metrics.csv.gz、seed_metrics.csv、P0_QA.csv、numerics_and_steps.json、seed_selection_audit.json、costs.csv、epsilon_failure/steps_failure图）；data/M8a_v1/（P0.json、seed_confidence_diagnostic.json、diffmean.json、test500_ids.json、final_lock.json、freeze.json）；reports/M8a_evidence/；docs/M8a_LIVE.json。\n'
    assert sum(line.startswith('|:---') for line in text.splitlines())==3
    assert len(text.encode())<=8000,len(text.encode())
    Path('reports/M8a_REVIEW.md').write_text(text)
    details='# M8a detailed evidence\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n## Complete summary\n\n'+summary.to_csv(index=False)
    Path('reports/M8a_details.md').write_text(details)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

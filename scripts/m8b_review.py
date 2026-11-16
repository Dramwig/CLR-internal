"""Reproducible M8b review from final group-level results and cost evidence."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from clr.audit.m8b_statistics import group_ci
from clr.audit.m8b_runtime import Cost,snapshot,total_cost,live
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.config import load_config

def pct(x):return f'{float(x)*100:.3g}'
def interval(row,key='failure'):return f"{pct(row[key])} [{pct(row[key+'_lo'])}, {pct(row[key+'_hi'])}]"
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);a=root/c['analysis_name'];summary=pd.read_csv(a/'robustness_summary.csv');df=pd.read_csv(a/'group_metrics.csv.gz');qa=pd.read_csv(a/'QA_groups.csv')
        assert df[df.split=='test'].group_id.nunique()==500
        # Replace the smaller val QA proxy with clean QA on the full fixed 200.
        qparts=[]
        for f in ['reader_val_QA200_allneg.json','reader_val_QA200_R1_allneg.json']:
            raw=json.loads((root/f).read_text())['rows'];p=pd.DataFrame(raw);p=p[p.category=='clean'];p=p.groupby(['split','group_id','design','seed','task'],as_index=False).correct.mean().rename(columns={'correct':'accuracy'});qparts.append(p)
        fullqa=pd.concat([qa[qa.split=='test'],*qparts],ignore_index=True);fullqa.to_csv(a/'QA_main_groups.csv',index=False);qsum=[]
        for key,p in fullqa.groupby(['split','design','task']):
            m,lo,hi=group_ci(p.groupby('group_id').accuracy.mean());qsum.append(dict(zip(['split','design','task'],key))|{'accuracy':m,'lo':lo,'hi':hi,'groups':p.group_id.nunique()})
        qs=pd.DataFrame(qsum);qs.to_csv(a/'QA_main_summary.csv',index=False)
        overhead=json.loads((root/'overhead.json').read_text());ov=pd.DataFrame(overhead['rows']);ovs=ov.groupby(['design','k'],as_index=False)[['source_ms','reader_ms','receiver_ms','receiver_with_state_ms','total_observation_audit_ms','incremental_over_receiver_ms']].median();ovs.to_csv(a/'overhead_summary.csv',index=False)
        costs=[]
        for p in Path('runs').glob('*/config.json'):
            cfg=json.loads(p.read_text())
            if cfg.get('milestone')!='M8b':continue
            met=json.loads((p.parent/'metrics.json').read_text());costs.append({'run':str(p.parent),'mode':cfg.get('mode'),'design':cfg.get('design'),'seed':cfg.get('seed'),'gpu_hours':met.get('gpu_hours',np.nan),'source_forward_calls':met.get('source_forward_calls',np.nan),'status':met.get('status'),'test_opened':met.get('test_opened',False)})
        pd.DataFrame(costs).to_csv(a/'costs.csv',index=False);df.groupby(['split','kind','epsilon','steps','design','seed','k','alpha'],dropna=False)[['failure','detected','clean_FPR','target_achieved']].mean().reset_index().to_csv(a/'seed_metrics.csv',index=False)
        base=summary[(summary.split=='test')&(summary.steps==200)&(summary.k==16)&(summary.alpha==.05)&(summary.epsilon==.03)]
        def pick(d,kind='E3',split='test',eps=.03,alpha=.05):
            p=summary[(summary.split==split)&(summary.design==d)&(summary.kind==kind)&(summary.epsilon==eps)&(summary.alpha==alpha)&(summary.k==16)&(summary.steps==200)];assert len(p)==1;return p.iloc[0]
        def qval(split,d,t):
            q=qs[(qs.split==split)&(qs.design==d)&(qs.task==t)].iloc[0]
            return f'{pct(q.accuracy)} [{pct(q.lo)}, {pct(q.hi)}]'
        ranks=sorted([(float(pick(d).failure),d) for d in c['designs']]);best=ranks[0][1];bp=pick(best);baseline=pick('R1');delta=float(baseline.failure-bp.failure)
        paired=pd.read_csv(a/'paired_reductions.csv');pp=paired[(paired.split=='test')&(paired.kind=='E3')&(paired.epsilon==.03)&(paired.steps==200)&(paired.alpha==.05)&(paired.design==best)] if best!='R1' else pd.DataFrame()
        significant=bool(len(pp) and float(pp.iloc[0]['lo'])>0)
        conclusion=(f'{best}在ε=0.03、λ=1白盒E3下失效率最低，为{interval(bp)}%；较R1下降{pct(delta)}个百分点。'+(f'配对95%CI[{pct(pp.iloc[0]["lo"])}, {pct(pp.iloc[0]["hi"])}]个百分点。' if len(pp) else '')+('该固定评估下有正向证据，不能推广为鲁棒保证。' if significant else '未得到明确降低白盒失效的证据。'))
        first=[];second=[];third=[]
        for d in ['R1','R2','R3','R4']:
            r=pick(d);v=pick(d,split='reader_val');o=ovs[(ovs.design==d)&(ovs.k==16)].iloc[0]
            first.append([d,f'{qval("reader_val",d,"A1")}/{qval("reader_val",d,"A4")}',f'{qval("test",d,"A1")}/{qval("test",d,"A4")}',interval(v,'clean_FPR'),interval(r,'clean_FPR'),f'{o.total_observation_audit_ms:.3g}'])
            second.append([d,*[interval(pick(d,kind)) for kind in ['E1','E2','E3']],pct(pick(d,alpha=.01).failure)])
            third.append([d,*[pct(pick(d,eps=e).failure) for e in [.01,.03,.1,.3]]])
        t1=table(['设计','val A1/A4 %','test A1/A4 %','val FPR %','test FPR %','审计ms'],first)
        t2=table(['设计','E1失效 %','E2失效 %','E3失效 %','E3@1%失效 %'],second)
        t3=table(['设计','ε=.01','ε=.03','ε=.1','ε=.3'],third)
        rp=summary[(summary.split=='test')&(summary.design=='R1')&(summary.kind=='E3')&(summary.epsilon==.03)&(summary.steps==200)&(summary.alpha==.05)].sort_values('k');ktext='；'.join(f'k={int(r.k)}: {pct(r.failure)}%' for _,r in rp.iterrows())
        gpu=total_cost();calls=sum(x['source_forward_calls'] for x in costs if np.isfinite(x['source_forward_calls']));selection=json.loads((root/'variant_selection.json').read_text())['selected_variants'];smoke=json.loads((root/'gpu_matched_smoke_v3_part0.json').read_text())
        sens_path=root/'sensitivity_v1/completion.json'
        sens_groups=json.loads(sens_path.read_text())['groups']['test'] if sens_path.exists() else 0
        sensitivity=f'附加λ=.1/10敏感性固定{sens_groups}组；含按攻击侧分数合并三λ的结果，见sensitivity_v1/lambda_summary.csv。主表/ε曲线固定λ=1。' if sens_groups else 'λ=.1/10敏感性未执行，见预算锁定说明；主表固定λ=1。'
        leaders='；'.join(f'ε={e}: '+','.join(d for d in c['designs'] if abs(float(pick(d,eps=e).failure)-min(float(pick(x,eps=e).failure) for x in c['designs']))<1e-12) for e in [.01,.03,.1,.3])
        review=f'''# M8b 审查

### (a) 状态一句话
M8b设计比较完成待审查。{conclusion}

### (b) 关卡清单
- 通过：val20同设备逐位重放/PGD smoke，{smoke['seconds']:.3g}秒≤900秒；协议单测通过；上游62363项哈希核验无变化（最终复核另索引）。
- 通过：R4固定train4000；三seed11/22/33，新适配器、冻结原主体。R2/R3按三seed平均val NLL整体选{selection['R2']}/{selection['R3']}；val200选模型、cal_A300/cal_B1000，test前锁定。所有设计主分布统一A1/A4。
- 通过：test500、PGD val20，E1/E2/E3、四ε、50/200预算。真实BF16逐层约束和E2旧f逐位不变检查通过；报告抽样不参与优化/停止/候选选择。
- 成本：保守累计{gpu:.3g}/25 GPUh（含失败/额外训练/加载、孤儿run10.1分钟上界），有记录的源前向{calls:.3g}次（孤儿run缺失）；共享RTX4090按驻留墙钟计时。

### (c) 核心主表（最多3张）
QA val200/test500，仅clean；A1正/负权重各半、覆盖三种假值；FPR与PGD val20/test500。三seed均值及组bootstrap 95%CI，FPR为名义5%门限；时延为val20三seed中位数，R4包含消费消息的receiver prefill。

{t1}

ε=.03、λ=1、200步预算：主失效=原本非目标→CF目标且接受，固定分母500。方括号为10000次group bootstrap 95%CI（组内平均三seed，条件于这些seed/门限/搜索）。E3@1%为名义1%门限。

{t2}

白盒E3失效%随ε，λ=1、名义5%门限；有限搜索为未知最坏失效的经验下界。

{t3}

逐ε最低点估计：{leaders}；未作跨ε统一最优保证。

R1随机问集：{ktext}。曲线及独立单问veto指数参考见证据；均值NLL+conformal不是独立veto，未证明指数保证。{sensitivity}

### (d) 本里程碑新增的偏离与阻塞
R2/R3只有128维，R4为512维；R2语义学习较弱、R4读值也非近完美。新适配器使用4000组并修复A1三种假值的训练覆盖，原R1沿用M71权重，训练差异与防御结构存在混杂。R2/R3增加原适配器初始化/24epoch版本，按val整体选，不挑seed。R1主比较统一A1/A4后重校准，不能直接把旧M8a四类型数字作同口径对照。
R2 CPU/GPU随机读取会有约1ulp差，最终校准/部署统一GPU路径；早期诊断保留。LIVE并发临时文件、freeze字段、梯度inplace/一致率浮点计数故障已修复并留记录。一个中断run实际GPU成本未知，额外计入保守上界。原始状态17.5GiB，初次文字估算误算已更正。
主矩阵优先保留500组；附加λ敏感性规模及test前预算决策见budget_lock.json，未按test表现选规模或λ。
E1/E2固定原f的盲区；R3覆盖全部层不等于其学习到每层因果信息。R1/各设计都可能被相同私有值的重复问题相关性限制。成功早停沿用同一扰动到50/200，不能将重复结果当多步收敛证据；实际步数见CSV。Conformal只在clean可交换假设下有边际保证，不保证自适应攻击。

### (e) 需要审查者决定的问题
无。{conclusion}

### (f) 证据路径
reports/M8b_details.md；data/M8b_v1/{c['analysis_name']}/（robustness_summary.csv、group_metrics.csv.gz、paired_reductions.csv、QA_main_summary.csv、seed_metrics.csv、overhead_summary.csv、costs.csv、epsilon_failure_*.pdf、k_failure.pdf）；data/M8b_v1/sensitivity_v1/（lambda_summary.csv、steps_failure_*.pdf）；data/M8b_v1/（final_lock.json、variant_selection.json、selected_training/、selected_calibration_v3/、freeze.json）；docs/M8b_TASK.md、M8b_LIVE.json；reports/M8b_evidence/。
'''
        assert len(review.encode())<=8000,len(review.encode());Path('reports/M8b_REVIEW.md').write_text(review)
        detail=review+'\n\n## 完整协议与来源\n'+Path('docs/M8b_TASK.md').read_text()+'\n\n## 成本与复现\n'+pd.DataFrame(costs).to_csv(index=False)+'\n\n## 逐seed结果\n'+pd.read_csv(a/'seed_metrics.csv').to_csv(index=False)
        Path('reports/M8b_details.md').write_text(detail);write_json(root/'completion.json',{'status':'experiments_complete_pending_review_and_freeze','review':'reports/M8b_REVIEW.md','analysis':str(a),'best_point_estimate':best,'positive_paired_evidence':significant,'gpu_hours_conservative':gpu,'source_forward_calls':calls,'review_bytes':len(review.encode()),'run':str(run.path),'test_groups':500,'val_QA_groups':200,'val_PGD_groups':20})
        cost.update(status='completed');live(stage='report_completed_pending_freeze',milestone_complete=False,run=str(run.path))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

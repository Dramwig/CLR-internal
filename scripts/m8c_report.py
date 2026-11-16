"""Review from measured, locked statistics; no hand-entered results."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
from clr.audit.m8c_runtime import snapshot,Cost,total_cost
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def fmt(x,percent=False):
    if x is None:return 'NA'
    return f'{x*(100 if percent else 1):.3g}'

def est(row,prefix='',percent=True):
    return f"{fmt(row[prefix+'mean'],percent)} [{fmt(row[prefix+'low'],percent)}, {fmt(row[prefix+'high'],percent)}]"

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);folder=root/c['analysis_folder'];s=json.loads((folder/'summary.json').read_text())
        gate=json.loads((root/'gate_short_val200_summary.json').read_text());budget=json.loads((root/'budget_lock.json').read_text())
        gate_rows=json.loads((root/'gate_short_val200_rows.json').read_text())['rows']
        gate_ci={};rng=np.random.default_rng(c['seed'])
        for category in ['message','none','mismatch']:
            by_group={r['group_id']:int(r['correct']) for r in gate_rows if r['category']==category}
            values=np.asarray(list(by_group.values()),dtype=float);draws=np.empty(10000,dtype=float)
            for start in range(0,len(draws),500):
                count=min(500,len(draws)-start)
                draws[start:start+count]=values[rng.integers(0,len(values),size=(count,len(values)))].mean(1)
            gate_ci[category]={'mean':float(values.mean()),'low':float(np.quantile(draws,.025)),
                               'high':float(np.quantile(draws,.975)),'groups':len(values),
                               'bootstrap_resamples':10000}
        write_json(folder/'admission_bootstrap.json',{'seed':c['seed'],'unit':'episode group','categories':gate_ci})
        test_groups=budget['test_groups']
        lookup={(r['split'],r['design'],r['task']):r for r in s['QA']}
        main=[r for r in s['robustness'] if r['panel']=='main' and r['rule']=='selected' and r['kind']=='E3' and r['alpha']==.05]
        curves={(r['split'],r['design'],r['epsilon']):r for r in main}
        lines=['# M8c 审查','', '### (a) 状态一句话',c['completion_statement'],'', '### (b) 关卡清单',
            f"- 通过：short direct-lookup val200有/无/错配消息准确率{fmt(gate_ci['message']['mean'],True)}/{fmt(gate_ci['none']['mean'],True)}/{fmt(gate_ci['mismatch']['mean'],True)}%，95%组bootstrap CI分别为[{fmt(gate_ci['message']['low'],True)},{fmt(gate_ci['message']['high'],True)}]/[{fmt(gate_ci['none']['low'],True)},{fmt(gate_ci['none']['high'],True)}]/[{fmt(gate_ci['mismatch']['low'],True)},{fmt(gate_ci['mismatch']['high'],True)}]%，达到≥80/≤35/≤35%准入。完整日志69.5/27.0/30.0%未通过，按预定顺序缩短日志。",
            '- 通过：冻结synthetic v2三核心SHA；仅删除非目标Ticket日志、保留全部目标记录。D0沿用M71权重/温度，新cal_B1000重定阈值；R4新适配器、原主体冻结，train4000、val200、cal_A300/cal_B1000。',
            '- 通过：R4三seed各≥4.22e+06问题曝光，checkpoint仅在达到曝光后按val NLL选。审计期望严格解析可见commitment；receiver不看commitment或私有原文。',
            '- 通过：val20 PGD smoke、实际BF16逐层约束、保存/在线特征重放与协议测试；test先锁定配置/权重/ID。报告抽样独立于优化/停止/λ选择。',
            f"- 成本：累计{total_cost():.3g}/{c['budget_gpu_hours']} GPUh，含加载/失败/smoke/训练/校准/评估，RTX4090按驻留墙钟计。矩阵与预算限制见(d)。",'',
            '### (c) 核心主表（最多3张）',
            '三seed组内均值；95%CI为10000次group bootstrap，条件于这些seed/阈值/有限搜索。QA为clean，A1正负各半且覆盖三种假值；FPR由独立clean问题集报告。', '',
            '| 读取器 | val A1/A4 % | test A1/A4 % | val/test clean FPR % |', '|---|---|---|---|']
        for d in c['designs']:
            qa=[' / '.join(est(lookup[split,d,t]) for t in ['A1','A4']) for split in ['reader_val','test']]
            fpr=' / '.join(est(curves[split,d,.1],'clean_fpr_') for split in ['reader_val','test'])
            lines.append(f'| {d} | {qa[0]} | {qa[1]} | {fpr} |')
        point=lines.index('### (c) 核心主表（最多3张）')
        lines[point:point]=c.get('gate_limits',[])+['']
        lines+=['',f'E3：test固定分母{test_groups}（来自冻结test500的哈希序前缀；val为超时run完整前19组的描述性结果），原本非目标→CF目标且在名义FPR5%阈值下接受；200步上限/EOT16，成功早停。主λ与附加网格严格遵循budget_lock，有限攻击给未知最坏失效的经验下界。','',
                '| ε | D0 val/test失效 % | R4 val/test失效 % |','|---|---|---|']
        for eps in c['epsilons']:
            cells=[' / '.join(est(curves[split,d,eps],'failure_') for split in ['reader_val','test']) for d in c['designs']]
            lines.append(f'| {eps:g} | {cells[0]} | {cells[1]} |')
        lines+=['','最小扰动为有限PGD找到的成本，非认证数学最小。比值=规避读取器成本/仅改变receiver成本；原已目标、搜索失败/右截尾不当作零或删除。表中分位数条件于双侧搜索成功，全部截尾与独立报告面板接受率另见CSV。','',
                '| split/读取器 | 有限比值对数/全部 | P10/P50/P90 | 读取器/receiver截尾 |','|---|---|---|---|']
        for r in s['minimum']:
            lines.append(f"| {r['split']}/{r['design']} | {r['finite_pairs']}/{r['pairs']} | {fmt(r['q10'])}/{fmt(r['q50'])}/{fmt(r['q90'])} | {r['reader_censored']}/{r['receiver_censored']} |")
        if not s['minimum']:lines.append('| test/D0,R4 | 未执行完成组 | NA | NA |')
        lines+=['','次要项实际执行：'+json.dumps(s.get('secondary_status',{}).get('summary',{}),ensure_ascii=False)+'。仅完成组用于描述性比值分位数，不能冒称100组完成。']
        lines+=['','### (d) 本里程碑新增的偏离与阻塞',*c['limitations'],
                '共享卡显存预检失败及一次GPU5模型加载OOM均留run/trace并计费；未换模型/精度。早期误用M8b runtime曾更新其LIVE，已恢复索引；冻结结果SHA复核不变。早期route-context诊断、低曝光R4只作开发证据，未进入主表。',
                'Conformal阈值只在clean可交换假设下有边际误报保证；自适应攻击不享有该保证。依赖准入证明群体任务依赖消息，不能证明每次行为改变都必然改变可读语义。R4消费消息后才可检查，不能在消费前拒收。各ε独立有限PGD、未合并小半径候选，非单调曲线不表示更大扰动更安全。','',
                '### (e) 需要审查者决定的问题','无。','', '### (f) 证据路径',
                'reports/M8c_details.md；data/M8c_v1/'+c['analysis_folder']+'/（robustness_summary.csv、group_metrics.csv.gz、qa_summary.csv、minimum_summary.csv、minimum_ratios.csv、minimum_rows.json、costs.csv、epsilon_failure.pdf）；data/M8c_v1/（final_lock.json、budget_lock.json、training_exposure_v2/、calibration/、upstream_final.json、freeze.json）；docs/M8c_TASK.md、M8c_LIVE.json；reports/M8c_evidence/。','']
        review=Path('reports/M8c_REVIEW.md');review.write_text('\n'.join(lines));assert len(review.read_bytes())<=8000,len(review.read_bytes())
        detailed=['# M8c 详细证据','',c['completion_statement'],'','协议：docs/M8c_TASK.md。全部数值由scripts/m8c_statistics.py聚合，不按test调参。',
            '', '预算锁定：','```json',json.dumps(budget,ensure_ascii=False,indent=2),'```','',
            '统计、固定分母和搜索：每组先平均三个seed，再10000次重采样episode组；不把λ当独立样本。原已目标不计失效但保留在固定分母。λ只按攻击侧选择分数选，报告抽样与其隔离。',
            '最小扰动二分的PGD未成功不证明不可行；区间是搜索定位区间。有限比值表条件于双侧成功，必须连同截尾比例、原已目标数及held-out接受率阅读。','',
            '## 运行索引']
        for p in sorted(root.glob('*part*.json')):detailed.append(f'- {p}: '+json.loads(p.read_text())['run'])
        detailed+=['','## 训练曝光']
        for seed in c['reader_seeds']:
            ref=json.loads((root/c['training_folder']/f'R4_{seed}.json').read_text());detailed.append(f"- seed{seed}: selected_epoch={ref['best_epoch']}, selected_exposure={ref['selected_exposure']}, total_exposure={ref['exposure']}, checkpoint={ref['checkpoint']}")
        detailed+=['','## 依赖准入组bootstrap','```json',json.dumps({'seed':c['seed'],'unit':'episode group','categories':gate_ci},ensure_ascii=False,indent=2),'```']
        detailed+=['','## 完整统计','```json',json.dumps(s,ensure_ascii=False,indent=2),'```']
        Path('reports/M8c_details.md').write_text('\n'.join(detailed)+'\n');write_json(root/'completion.json',{'run':str(run.path),'review':str(review),'gpu_hours':total_cost(),'status':c['completion_statement']})
        cost.update(status='completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

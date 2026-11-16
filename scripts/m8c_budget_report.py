"""Honest incomplete-milestone report at the user-required pre-test budget gate."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,json
from pathlib import Path
import numpy as np
from clr.audit.m8c_runtime import snapshot,Cost,total_cost,live
from clr.audit.features import sha256
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def interval(values,rng):
    values=np.asarray(values,dtype=float);draws=np.empty(10000)
    for start in range(0,10000,500):draws[start:start+500]=values[rng.integers(len(values),size=(500,len(values)))].mean(1)
    return {'mean':float(values.mean()),'low':float(np.quantile(draws,.025)),'high':float(np.quantile(draws,.975)),'groups':len(values)}

def percent(row):
    return f"{100*row['mean']:.3g} [{100*row['low']:.3g}, {100*row['high']:.3g}]"

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);dest=root/c['analysis_folder'];dest.mkdir(exist_ok=True)
        budget=json.loads((root/'budget_feasibility.json').read_text());assert not budget['passed']
        assert not (root/'final_lock.json').exists() and not (root/'test_ids.json').exists()
        verify=json.loads((root/c['verification_name']).read_text());assert verify['passed']
        rng=np.random.default_rng(c['seed']);gate_doc=json.loads((root/'gate_short_val200_rows.json').read_text());gate={}
        for category in ['message','none','mismatch']:
            rows=[r for r in gate_doc['rows'] if r['category']==category]
            assert len(rows)==len({r['group_id'] for r in rows})==200
            gate[category]=interval([int(r['correct']) for r in rows],rng)
        write_json(dest/'admission_bootstrap.json',{'seed':c['seed'],'resamples':10000,'unit':'episode group','categories':gate})
        qa=[];per_seed=[]
        for design in c['designs']:
            grouped={task:{} for task in ['A1','A4']}
            for seed in c['reader_seeds']:
                rows=json.loads((root/c['calibration_folder']/f'{design}_{seed}_valQA.json').read_text())['rows']
                for task in grouped:
                    rr=[r for r in rows if r['task']==task]
                    assert len(rr)==200 and len({r['group_id'] for r in rr})==200
                    per_seed.append({'design':design,'seed':seed,'task':task,'accuracy':float(np.mean([r['accuracy'] for r in rr]))})
                    for r in rr:grouped[task].setdefault(r['group_id'],[]).append(r['accuracy'])
            for task,groups in grouped.items():
                assert all(len(v)==3 for v in groups.values())
                qa.append({'design':design,'task':task,'split':'reader_val',**interval([np.mean(v) for v in groups.values()],rng)})
        with (dest/'qa_val200.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(qa[0]));writer.writeheader();writer.writerows(qa)
        with (dest/'qa_val200_per_seed.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(per_seed[0]));writer.writeheader();writer.writerows(per_seed)
        source=Path(budget['val_run']);grouped={}
        for row in map(json.loads,(source/'results.jsonl').read_text().splitlines()):grouped.setdefault(row['group_id'],[]).append(row)
        complete=[rows for rows in grouped.values() if len(rows)==30];curves=[]
        for epsilon in [.01,.1,1.0]:
            for design in c['designs']:
                failures=[]
                for rows in complete:
                    rr=[r for r in rows if r['epsilon']==epsilon and r['attack_design']==design]
                    assert len(rr)==3
                    failures.append(np.mean([r['changed_to_target'] and not r['audit'][f"{design}:{r['attack_seed']}"]['0.05'] for r in rr]))
                curves.append({'design':design,'epsilon':epsilon,'split':'reader_val_partial',**interval(failures,rng)})
        with (dest/'e3_val_partial.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
        exposures=[]
        for seed in c['reader_seeds']:
            ref=json.loads((root/c['training_folder']/f'R4_{seed}.json').read_text())
            exposures.append({'seed':seed,'selected_exposure':ref['selected_exposure'],'total_exposure':ref['exposure'],'best_epoch':ref['best_epoch'],'checkpoint':ref['checkpoint']})
        summary={'run':str(run.path),'complete':False,'test_opened':False,'budget':budget,'admission':gate,'QA':qa,'QA_per_seed':per_seed,'exposure':exposures,'partial_val_E3':curves}
        write_json(dest/'summary.json',summary)
        q={(r['design'],r['task']):r for r in qa};cv={(r['epsilon'],r['design']):r for r in curves}
        lines=['# M8c 审查','','### (a) 状态一句话',
          '未完成，停在用户指定的预算关卡：依赖准入与R4训练已完成，预算不足主test至少200组；test未打开，等待预算决定。','',
          '### (b) 关卡清单',
          f"- 通过：short direct-lookup val200有/无/错配消息准确率分别为{percent(gate['message'])}/{percent(gate['none'])}/{percent(gate['mismatch'])}%，达到≥80/≤35/≤35%（按点估计）。完整日志69.5/27.0/30.0%未通过后，按指定顺序缩短日志；无需换模型。",
          '- 通过：synthetic v2冻结源码只读import；short仅删除非目标Ticket日志。reader仅读状态与类型化问题，receiver仅读公开题面+KV；commitment仅供审计，期望答案从可见commitment解析。D0权重冻结、cal_B1000重定阈值；R4 train4000、val200、cal_A300/cal_B1000，主体冻结。',
          '- 通过：R4三seed曝光≥4.22e+06，仅在达标后按val NLL选择；A4均≥90%。10项协议测试通过，包括公开期望、conformal、split、预算裁档和E1/E2特征保持。上游M71/M8a/M8b共33,897项SHA一致。',
          '- 未通过（未完成）：正式val20五ε矩阵触发90分钟上限，19组完整、末组10/30条，按失败run保留；不称val20完成。早期val smoke与特征严格重放通过。',
          f"- 未通过（预算）：累计{budget['spent_gpu_hours']:.3g}/15 GPUh，剩余{budget['remaining_gpu_hours']:.3g}。新三ε主E3最大可行估计{budget['maximum_feasible_groups']}组，小于用户下限200；不创建final_lock，不打开test，不启动次要项。",'',
          '### (c) 核心主表（最多3张）',
          'QA为clean reader_val全部200组；三seed组内平均，95%CI为10000次episode组bootstrap，条件于这些seed。test均未执行。','',
          '| 设计 | reader_val A1 % | reader_val A4 % | test |','|---|---|---|---|']
        for design in c['designs']:lines.append(f"| {design} | {percent(q[design,'A1'])} | {percent(q[design,'A4'])} | 未执行 |")
        lines+=['','仅完整val前19组的开发性E3失效（FPR5%阈值）；三seed、λ=1、200步上限/EOT16、成功早停，固定分母19。末组未完成，以下不是最终val20或test结论。','',
          '| ε | D0 reader_val失效 % | R4 reader_val失效 % | test |','|---|---|---|---|']
        for epsilon in [.01,.1,1.0]:lines.append(f"| {epsilon:g} | {percent(cv[epsilon,'D0'])} | {percent(cv[epsilon,'R4'])} | 未执行 |")
        lines+=['','预算按新主矩阵：E3 λ=1×D0/R4×3seed×ε={.01,.1,1}，PGD200；全部剩余预算先供主指标，超过60%最低预留。单位为GPU小时，含历次加载、失败、smoke、训练、校准和中止上界；RTX4090驻留墙钟。','',
          '| 预算量 | 平均吞吐点估计 | 保守规划估计 |','|---|---|---|',
          f"| 最大可行test组数 | {budget['maximum_at_point_rate']} | {budget['maximum_feasible_groups']} |",
          f"| 主200组新增GPUh | {budget['point_primary_200_gpu_hours']:.3g} | {budget['primary_200_gpu_hours']:.3g} |",
          f"| 做满200组累计GPUh | {budget['point_total_200_gpu_hours']:.3g} | {budget['total_200_gpu_hours']:.3g} |",'',
          '### (d) 本里程碑新增的偏离与阻塞',
          '依用户最新协议，主ε=.03/.3裁掉；旧test60方案从未锁定或运行。次要项仅主n前100组，顺序为λ=.1/10（ε=.1）→最小成本（最多6次二分）→E1/E2（ε=.1）；均待主≥200预算通过后按剩余预算执行。',
          '成本预测以19个完整val组拟合非负固定/D0步/R4步耗时，删去裁掉ε的步数；保守规划取组平均吞吐bootstrap 97.5%上端×1.20，另留0.0500 GPUh加载。20%为规划余量，不是实测科学结论。末组超时可能造成耗时低估，test长尾也可能不同；最大可行数是预测，不是完成保证。',
          '正式val矩阵超时与旧λ敏感性按用户修订中止，部分结果保留，不补造完成。中止进程末次心跳及终止时间上界分别保存，保守补差计入账本。尝试的融合扰动算术改变PGD轨迹且未提速，已撤回，仅留诊断和成本。',
          'val失效随ε下降不表示更大扰动更安全：各半径独立有限PGD、未合并小半径候选，且有早停与审计惩罚；搜索没有认证最坏情况或呈现真实半径单调上包络。',
          'A的群体依赖门槛不证明每次行为改变必然改变任务语义。有限PGD只给未知最坏失效的经验下界；clean conformal阈值不保证自适应检测。R4消费消息后检查；最小扰动尚未执行，不能报告放大倍数。','',
          '### (e) 需要审查者决定的问题',
          f"依用户“主200组做不到则先停下”规则：是否追加预算？主200组保守累计需{budget['total_200_gpu_hours']:.3g} GPUh（比15多{budget['additional_gpu_hours_for_200']:.3g} GPUh）；次要项另需预算，尚无test数值。",'',
          '### (f) 证据路径',
          'reports/M8c_details.md；data/M8c_v1/budget_feasibility.json、analysis_budget_pause/（summary.json、admission_bootstrap.json、qa_val200.csv、qa_val200_per_seed.csv、e3_val_partial.csv）；training_exposure_v2/、calibration/、'+c['verification_name']+'；docs/M8c_TASK.md、M8c_LIVE.json；reports/M8c_evidence/（budget_feasibility.log、val_main_v2.log、val_sensitivity_final.log、tests_budget_pause.log）。','']
        review=Path('reports/M8c_REVIEW.md');review.write_text('\n'.join(lines));assert len(review.read_bytes())<=8000,len(review.read_bytes())
        assert sum(l.startswith('|---') for l in lines)==3
        detailed=['# M8c 预算关卡详细证据','','未完成；test封闭，未创建final_lock.json。',
          '用户修订见docs/M8c_TASK.md；15 GPUh不足至少200组时停下，以下不代表最终test结论。',
          '', '## 实测与预测方法',budget['forecast_note'],
          '新主协议预期最大n取floor((15−已花成本−加载预留)/规划组成本)，上限500；不读取test样本、标签、预测或根据test调整。按点吞吐也不足200组，故未执行主矩阵。',
          '被截断的val最后一组不进入吞吐拟合与19组描述性失效表，已明确保留；可能使预测偏乐观。GPU计费为驻留墙钟，与利用率无关。',
          '## 预算完整JSON','```json',json.dumps(budget,ensure_ascii=False,indent=2),'```',
          '## 三seed曝光/QA/准入与val部分结果','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```',
          '## 运行索引',f'- val主矩阵（failed，90分钟）：{budget["val_run"]}',
          '- 旧val敏感性（按用户修订中止）：runs/20261109T052533568845Z_f645a165c902_91ae80；原心跳metrics_before_user_interrupt.json、interrupt_reconciliation.json及cost_upper_additions.json保留。',
          '- 此报告run：'+str(run.path)]
        Path('reports/M8c_details.md').write_text('\n'.join(detailed)+'\n')
        status=f"M8c未完成；依赖准入/R4训练通过，预算估计最大test{budget['maximum_feasible_groups']}组<用户下限200，主200组保守累计需{budget['total_200_gpu_hours']:.3g}GPUh；当前{budget['spent_gpu_hours']:.3g}/15GPUh。test封闭，等待预算决定；审查reports/M8c_REVIEW.md，接续docs/M8c_LIVE.json。"
        for path in ['AGENTS.md','docs/STATUS.md']:
            p=Path(path);content=p.read_text();start='<!-- M8C_LIVE -->';end='<!-- /M8C_LIVE -->';a=content.index(start);b=content.index(end,a)
            p.write_text(content[:a]+start+'\n'+status+'\n'+content[b:])
        cost.update(status='completed');live(stage='awaiting_budget_decision',milestone_complete=False,test_opened=False,
            background_tasks_active=False,review=str(review),maximum_feasible_groups=budget['maximum_feasible_groups'],
            total_200_gpu_hours=budget['total_200_gpu_hours'],gpu_hours_conservative=total_cost())
        evidence=[review,Path('reports/M8c_details.md'),root/'budget_feasibility.json',root/c['verification_name'],*dest.iterdir()]
        write_json(root/'budget_pause_evidence.json',{'run':str(run.path),'complete':False,'test_opened':False,
            'sha256':{str(p):sha256(p) for p in evidence if p.is_file()}})
        print(status,flush=True);print('review bytes',len(review.read_bytes()),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args();main(load_config(args.config))

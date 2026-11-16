"""Build the sole M5b review and detailed, fully indexed evidence report."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m5b_runtime import hours_used, snapshot
from clr.utils.records import recorded


def read(path):
    return json.loads(Path(path).read_text())


def num(x):
    return f'{float(x):.3g}' if pd.notna(x) else 'NA'


def ci(row, field='mean', scale=1, low='ci_low', high='ci_high'):
    return f"{num(scale*row[field])} [{num(scale*row[low])}, {num(scale*row[high])}]"


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|',
                      *('|'+'|'.join(str(x) for x in row)+'|' for row in rows)])


def cost_ledger(root, exclude):
    total, ledger = hours_used(exclude)
    for row in ledger:
        path = Path(row['run']); cfg = read(path/'config.json')
        met = read(path/'metrics.json') if (path/'metrics.json').exists() else {}
        row.update(status=met.get('status', 'missing'), seed=cfg.get('seed'),
                   source_forward_calls=met.get('source_forward_calls', 0),
                   lane_forwards=met.get('lane_forwards', 0),
                   test_opened=met.get('test_opened', False))
    # This small operator-only numerical verification was executed interactively;
    # retain its actual timing without inventing a startup run snapshot.
    audit = root/'autocast_audit.json'
    if audit.exists():
        seconds = read(audit)['seconds']; total += seconds/3600
        ledger.append(dict(run=str(audit), case='operator_audit_no_run_snapshot', stage='verification',
                           status='completed', gpu_hours=seconds/3600, source_forward_calls=0,
                           lane_forwards=0, test_opened=False))
    pd.DataFrame(ledger).to_csv(root/'analysis/costs.csv', index=False)
    result = dict(gpu_hours=total, gpu='NVIDIA GeForce RTX 4090', limit=25,
                  source_forward_calls=sum(x.get('source_forward_calls', 0) for x in ledger),
                  lane_forwards=sum(x.get('lane_forwards', 0) for x in ledger),
                  includes='sum of complete job wall times, including smoke, failed, invalid and restart runs; CPU controllers excluded',
                  numerical_audit='actual measured timing, no startup snapshot; supplemental cost retained',
                  missing=[x['run'] for x in ledger if x['status'] in ['running','missing']])
    if result['missing']:raise ValueError('Cannot close cost ledger while runs remain active/missing: '+str(result['missing']))
    if total > 25:raise ValueError('M5b exceeded authorized GPU budget')
    write_json(root/'analysis/costs.json', result)
    return result


def main(c):
    with recorded(c) as run:
        start=time.monotonic();root=Path(c['output']);out=root/'analysis'
        snapshot(run,c,['scripts/m5b_report.py','scripts/m5b_analyze_a.py','scripts/m5b_analyze_bc.py'])
        for name in ['A_pipeline.json','D_pipeline.json','queue_C.json']:
            if read(root/name)['stage']!='completed':raise ValueError('Unfinished pipeline: '+name)
        if not read(root/'A_training_lock.json')['all_training_locked']:raise ValueError('A training not locked')
        for name in ['A_analysis_run.json','BC_analysis_run.json']:
            if read(Path(read(root/name)['run'])/'metrics.json')['status']!='completed':raise ValueError('Analysis not completed')
        a=pd.read_csv(out/'A_summary.csv');ap=pd.read_csv(out/'A_paired.csv');ak=pd.read_csv(out/'A_summary_by_kind.csv')
        cp=pd.read_csv(out/'A_conformal.csv');mc=pd.read_csv(out/'A_matched_coverage.csv');mp=pd.read_csv(out/'A_matched_coverage_paired.csv')
        gap=pd.read_csv(out/'A_gap_decomposition.csv');b=pd.read_csv(out/'B_summary.csv');s=pd.read_csv(out/'C_summary.csv')
        db=pd.read_csv(out/'D_B_paired.csv');dc=pd.read_csv(out/'D_C_paired.csv');bc=read(out/'BC_analysis.json')
        costs=cost_ledger(root, run.path)
        def av(case,split,q,metric):return a[(a.case==case)&(a.split==split)&(a.qtype==q)&(a.metric==metric)].iloc[0]
        def pv(aa,bb,split,q):return ap[(ap.a==aa)&(ap.b==bb)&(ap.split==split)&(ap.qtype==q)].iloc[0]
        def bv(value,metric,t='all'):return b[(b.value==value)&(b.metric==metric)&(b.t.astype(str)==str(t))].iloc[0]
        def mv(aa,bb,split,q):return mp[(mp.a==aa)&(mp.b==bb)&(mp.split==split)&(mp.qtype==q)].iloc[0]
        cases=['B3','B3-prefix','CLR-lite-single','CLR-lite','CLR-v1'];values=['B4','B3','CLR-lite','CLR-v1','D-lite']
        q1rows=[]
        for case in cases:
            for kind in ['clean','traj']:
                z=ak[(ak.case==case)&(ak.split=='reader_val')&(ak.qtype=='Q1')&(ak.kind==kind)]
                q1rows.append([case,kind,*[ci(z[z.metric==m].iloc[0]) for m in ['nll','auroc']]])
        mainrows=[]
        for split in ['reader_val','test']:
            for case in cases:
                mainrows.append([split,case,*[ci(av(case,split,q,m),scale=100 if q in ['Q3','Q4'] else 1) for q,m in [('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy')]]])
        maintable=table(['split','model','Q1 NLL↓','Q2 AUROC↑','Q3 acc %↑','Q4 acc %↑'],mainrows)
        matchedrows=[]
        for split in ['reader_val','test']:
            for q in ['Q1','Q2','Q3','Q4']:
                matchedrows.append([split,q,ci(mv('CLR-lite','B3',split,q),'delta'),ci(mv('CLR-lite','B3-prefix',split,q),'delta'),ci(mv('CLR-lite-single','B3',split,q),'delta'),ci(mv('CLR-v1','B3',split,q),'delta')])
        matchedtable=table(['split','qtype','lite−B3','lite−B3-prefix','single−B3','v1−B3'],matchedrows)
        search=s[s.method.isin(['S2','S3'])&~s.value.isin(['oracle','pass@N'])]
        best=search.sort_values(['delta_greedy','compute'],ascending=[False,True]).iloc[0]
        sig=search[search.simul_ci_low>0]
        greedy=s[s.method=='S0'].iloc[0]
        def setting(r):return f"{r['method']}/{r['value']}/N{int(r['N'])}/b{int(r['b'])}/h{int(r['horizon'])}"
        search_answer=(f"{len(sig)}个设置的同时区间下界>0。" if len(sig) else '无设置的同时区间下界>0。')
        search_answer+=f"观察最高为{setting(best)}，准确率{num(100*best.accuracy)}%，相对greedy {ci(best,'delta_greedy',100,low='simul_ci_low',high='simul_ci_high')}pp（全网格同时95%CI），成本{num(best.compute)}次/题。最高点仅描述，不用于调参。"
        residual=[]
        for q in ['Q3','Q4']:
            p=pv('CLR-lite','B3-prefix','test',q);ss=pv('CLR-lite-single','B3','test',q);g=gap[(gap.split=='test')&(gap.qtype==q)].iloc[0]
            vp=pv('CLR-v1','B3-prefix','test',q);vg=pv('CLR-v1','B3','test',q)
            residual.append(f"{q}：lite−B3-prefix {ci(p,'delta',100)}pp（原差{num(100*g.fraction_prefix_access_matched)}%）；single−B3 {ci(ss,'delta',100)}pp（原差{num(100*g.fraction_current_only)}%）；v1−B3-prefix {ci(vp,'delta',100)}pp（原差{num(100*vp['delta']/vg['delta'])}%）。")
        brows=[]
        for value in values:
            brows.append([value,ci(bv(value,'between_auroc')),ci(bv(value,'spearman')),ci(bv(value,'pair_accuracy')),ci(bv(value,'gain_vs_sigma0'),scale=100)])
        btable=table(['value','跨题池化AUROC','组内Spearman','大差兄弟对排序acc','选子相对clean ΔV̂ pp'],brows)
        btrows=[]
        for t in range(6):
            for value in values:
                r=bv(value,'spearman',t)
                btrows.append([t,value,ci(bv(value,'between_auroc',t)),ci(r),ci(bv(value,'pair_accuracy',t)),ci(bv(value,'gain_vs_sigma0',t),scale=100),int(r.valid_groups)])
        positive=sum(bv(v,'spearman').ci_low>0 and bv(v,'pair_accuracy').ci_low>.5 for v in values)
        negative=sum(bv(v,'gain_vs_sigma0').ci_high<0 for v in values)
        b_answer=f'{positive}/5种价值函数的组内ρ区间>0且大差对排序区间>.5，反驳“只有题目难度”的强版本；{negative}/5种选子ΔV̂区间仍<0，组内可分性尚不足以安全替换clean。真值平坦组占{num(100*bv("B3","spearman").flat_truth_fraction)}%。'
        dbrief=[]
        for metric,label,scale in [('spearman','ρ',1),('pair_accuracy','排序acc',1),('gain_vs_sigma0','ΔV̂(pp)',100)]:
            r=db[(db.t.astype(str)=='all')&(db.metric==metric)].iloc[0];dbrief.append(f"{label}差{ci(r,scale=scale)}")
        dsearch=dc.sort_values(['delta','method','N'],ascending=[False,True,True]).iloc[0]
        d_answer='；'.join(dbrief)+f"；搜索D−lite最高观察差{ci(dsearch,'delta',100,low='simul_ci_low',high='simul_ci_high')}pp（{dsearch['method']}/N{int(dsearch['N'])}/b{int(dsearch['b'])}/h{int(dsearch['horizon'])}，18设置同时区间），{int((dc.simul_ci_low>0).sum())}个设置显著改善。"
        ref=s[(s.value=='pass@N')&(s.N==16)&(s.horizon==6)].iloc[0]
        # Detailed report: all requested qtypes, states, strata and grid are indexed.
        detail=['# M5b 完整证据报告','完成实验，待审查。用户在Q3/Q4上有条件通过G1后授权本里程碑；未启动M6。',
          '## A：输入混杂对照',
          'B3-prefix 从头重训，输入 concat(mean(normalize(z₀…zₜ)), normalize(zₜ))；隐层128，其余原B3每类型8epoch×两LR协议不变。输入维度增加导致参数增加，不能当作纯信息消融。lite-single从头重训，只输入当前观测及位置0；与lite同全遍历/逆类型权重、batch1024、LR3e-4、WD1e-4、epoch8起选模、至多10epoch/patience1。所有六个checkpoint在正式test评估前锁定。',
          '每类型B3参数1.08×10⁵，B3-prefix 2.06×10⁵；lite/single/D均1.72×10⁷，精确计数见parameter_counts.json。',
          '以下主指标为clean、seen模板、各t等权；三seed均值及95% t区间。配对比较另按problem bootstrap10,000次，把所有t、模型seed放在同一cluster；其CI回答固定三seed平均的题目不确定性，不代替随机训练总体区间。Q1 reader_val的clean/traj分别见A_summary_by_kind.csv。',maintable,
          '### reader_val Q1 clean / traj',
          '与主表相同的三seed均值/95% t区间；traj仅使用该轨迹干净起点及其后实际观测的既定标签。',
          table(['model','kind','NLL↓','AUROC↑'],q1rows),
          '### 控制后残差','\n'.join(residual),
          '两种控制对比分别是“双方可访问前缀”和“双方只访问当前状态”，不能相加解释为因果贡献百分比。B3用均值摘要，CLR读有序前缀；模型容量、接口、训练目标与优化仍有差异。完整差值与reader_val结果：A_paired.csv；A_gap_decomposition.csv。',
          '### 每类同覆盖率效率','下表为准确匹配经验90%覆盖率的集合大小差（负值更小），配对95%CI；每次bootstrap重新估计阈值，边界ties随机化。该诊断用评估标签匹配覆盖率，仅描述效率，不是可部署conformal阈值或独立覆盖保证。',matchedtable,
          '原split conformal采用独立cal_A温度与cal_B250阈值，每t/qtype分层、一题一条；α=.1的实际coverage/size如下。保证依赖problem层可交换性；不能无条件推广到source分布偏移或自适应搜索。']
        cp_rows=[]
        for (split,case,q),f in cp[(cp.calibration=='temperature')&(cp.alpha==.1)&(~cp.heldout)].groupby(['split','case','qtype']):
            by=f.groupby('seed')[['coverage','size']].mean();cells=[]
            for metric in ['coverage','size']:
                x=by[metric].to_numpy();half=4.3026527299*x.std(ddof=1)/np.sqrt(3);cells.append(f'{num(x.mean())} [{num(x.mean()-half)}, {num(x.mean()+half)}]')
            cp_rows.append([split,case,q,*cells])
        detail += [table(['split','model','qtype','coverage','size'],cp_rows),'实际cal_B阈值的配对coverage/size CI见A_conformal_paired.csv；所有五种模型的同覆盖率绝对size见A_matched_coverage.csv；全部七种配对见A_matched_coverage_paired.csv。',
          '## B：同父兄弟候选诊断',
          'val300×3seed×6t，每组1个sigma0+16个π子候选，每子K16独立MC。读取一步前向后的实际观测及完整历史，原cohort左padding/KV不变；未复用语义不能核实的M5a rollout。B原始91,800行、5,400父状态组；D只重评分同一批实际状态。',
          '跨题池化AUROC保留K16 Bernoulli的正负质量，不以V̂>0或多数标签二值化；all为t等权、三seed t区间。组内指标先每组、再每题平均所有有效t/seed，题bootstrap10,000；Spearman去掉真值/预测任一恒定组；大差排序仅|ΔV̂|≥.25，预测ties=.5，无合格对的组不参与该指标；选择ties优先clean。',btable,
          b_answer+' 跨题可分性与同父兄弟排序能力须分别解释，不能仅从前者推断后者。K16有MC噪声，平坦组和有限扰动范围限制可辨识性；t5的终点rollout确定，K16重复不增加独立信息。',
          '逐t结果如下；ρ有效组最多每t 900组，其CI仍按题聚类。平坦比例与各指标有效组数另见B_summary.csv；逐父原始统计见B_groups.csv。',
          table(['t','value','跨题AUROC','组内ρ','大差对排序acc','选子ΔV̂ pp','ρ有效组'],btrows),
          'B只从干净父状态出发、真值为剩余π策略的V̂，不能直接外推到beam到达的自适应状态或早期搜索的sigma0尾续；C是实际闭环检验。原始oracle_source_forwards为整组成本、在兄弟行重复，成本总账使用run级真实计数，不能逐行直接求和。',
          '## C：含clean候选的搜索v2',
          f"greedy={ci(greedy,'accuracy',100,low='accuracy_ci_low',high='accuracy_ci_high')}%，成本{num(greedy.compute)}次/题。"+search_answer,
          'val300×3seed，全部5价值函数，S2/S3，N4/8/16，b2/4，全步与t≤3；N为每父总候选数，含1个sigma0。S2含一条全程clean轨迹，范围外sigma0尾续；终点打分只解码被选者。S3每次扩展保留clean子候选，clean祖先仍可能被评分剪枝；早期搜索在t4选一条后确定性续至终点。S1为独立π多数投票，另保留S1-clean。',
          '横轴以每lane源前向为单位：prefill统一记一次，潜步与实际答案token逐lane计数。prefill长度由冻结manifest的pad_to给出，未按其token数加权，因此该轴不是严格等FLOPs。价值函数按实际prefix长度的矩阵/attention FLOPs除以源一步FLOPs换算；source分母含context attention。排除embedding、归一化、activation与CPU序列化开销，也不代表端到端延迟。',
          '源模型保持fp32。LoRA融合前后以4条val噪声轨迹检验答案一致、状态最大绝对差≤.001；实际差保存在各run/merge_gate.json。全量S0须复现冻结greedy答案；每个B父状态的clean子亦核对下一步观测。',
          '固定每题平均预算32…16384，选择平均成本上限内成本最高的设置，与准确率无关；不够时使用greedy，预算余额可闲置。原始所有点/选用设置均保留；比较greedy等于在相同上限下让greedy闲置多余预算，不是每题硬截断预算。全90搜索设置同时bootstrap区间用于网格后“任何显著收益”的判断；逐点CI也保留。',
          'oracle仅终点S2，温度0时等于gold选择/pass@N；解码全部池候选的成本单列，未伪装为可学习价值函数。B的K16 MC成本全部计入总GPUh，但不计入可部署C的横轴。',
          '![相对greedy的accuracy–compute](../data/M5b_v1/analysis/accuracy_compute.png)',
          '主图横轴为实际每题平均执行成本，oracle单独画，避免其特权增益压缩学习价值函数的尺度。固定预算点/闲置余额对应accuracy_budget.png及C_budget_curves.csv。',
          '![特权oracle与pass@N](../data/M5b_v1/analysis/pass_at_N.png)',
          f'全部{len(s)}个方法/参考/网格设置保留；不以oracle参考宣称搜索价值函数成功。pass@N结果如下（准确率及配对差均为按题bootstrap95%CI）：',
          table(['范围h','N','pass@N acc % (95%CI)','相对greedy pp (95%CI)','参考成本'], [[int(r.horizon),int(r.N),ci(r,'accuracy',100,low='accuracy_ci_low',high='accuracy_ci_high'),ci(r,'delta_greedy',100),num(r.compute)] for _,r in s[s.value=='pass@N'].sort_values(['horizon','N']).iterrows()]),
          '## D：组内排序微调',
          'reader_train随机3300题×2t：3000题/6000组进入新训练，额外300题/600组仅训练侧monitor（M4.1预训练见过，不是独立验证）。每组1clean+7π、K4；Bernoulli逐样本CE，排序使用V̂差加权softplus。loss=.5×(原Q1 CE+兄弟CE)+rank；原Q1 replay排除monitor题。',
          '三seed分别从冻结lite初始化，全参数微调5epoch、LR1e-4、group_batch32、rank_weight1。checkpoint只由reader_val固定750题Q1 NLL选择；其余250题温度校准，B/C val300标签不参与选择。正温度单调，不改变搜索排名。',d_answer,
          '上述reader_val 750/250划分仅约束D阶段；初始化的M4.1模型此前使用过整个reader_val选模，因此不把250题追认为独立于全部历史模型选择的校准样本。D不宣称新的conformal保证。',
          '每t的D−lite配对CI见D_B_paired.csv，搜索逐对应设置差及18设置同时CI见D_C_paired.csv。',
          '## 实施偏离与预算',
          '早期无效A冻结输入干预混用了raw前缀均值和normalized当前状态，并提前打开test；整run排除，保留invalid_development_runs.json。正式重训协议在训练前锁定，不按该test调参；无法声称本里程碑test从未暴露。B3大GPU特征缓存OOM后明确改CPU缓存，batch/精度/模型不变。失败、目录创建修复与重启证据保留，均入账。',
          f"保守累计{num(costs['gpu_hours'])} GPUh≤25，RTX4090，含smoke/失败/无效run；物理source calls={costs['source_forward_calls']}，lane forwards={costs['lane_forwards']}。详见costs.csv/json。计算量曲线是推理成本，训练/生成成本另计于此总账。",
          'M41/M5a冻结全部文件哈希核验见../data/M5b_v1/frozen_inputs_verification.json。所有完整run均保存config/git/env/seed/source快照；唯一未建启动run的微小operator数值检查明确列补充账项，未补造快照。',
          '## 四问的回答','1. '+' '.join(residual),'2. '+b_answer,'3. '+search_answer,'4. '+d_answer,
          '## 复现入口与证据',
          '`configs/M5b_analysis_a.yaml` → `scripts/m5b_analyze_a.py`；`configs/M5b_analysis_bc.yaml` → `scripts/m5b_analyze_bc.py`；`configs/M5b_report.yaml` → `scripts/m5b_report.py`。完整运行索引data/M5b_v1/completion.json；全部分析文件索引data/M5b_v1/analysis/index.json；最终冻结data/M5b_v1/freeze.json。']
        Path('reports/M5b_details.md').write_text('\n\n'.join(detail)+'\n')
        # Compact review tables use means; complete seed intervals are above.
        compact=[]
        compactcp=[]
        for split in ['reader_val','test']:
            for case in cases:
                compact.append([split,case,*[num(av(case,split,q,m)['mean']*(100 if q in ['Q3','Q4'] else 1)) for q,m in [('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy')]]])
                cells=[]
                for q in ['Q1','Q2','Q3','Q4']:
                    z=cp[(cp.case==case)&(cp.split==split)&(cp.qtype==q)&(cp.calibration=='temperature')&(cp.alpha==.1)&(~cp.heldout)]
                    cells.append(f"{num(z.groupby('seed')['size'].mean().mean())}/{num(100*z.groupby('seed')['coverage'].mean().mean())}")
                compactcp.append([split,case,*cells])
        review=['## (a) 状态一句话',f"M5b A–D实验与统计完成，待审查；累计{num(costs['gpu_hours'])}/25 GPUh。",
          '## (b) 关卡清单',
          '通过：训练split训练/val代理评估smoke，B/C 5题全网格smoke 142秒（≤5%/15分钟）；A两控制各3seed，test+reader_val；B完整val300×3seed×6t×17子×K16；C五价值函数完整网格/三seed；D 3000题×2t×8子×K4及三seed。26项测试通过，上游4154个冻结文件SHA一致。',
          '问题(1)：'+' '.join(residual)+' 两种残差不能相加作因果分解；B3前缀为均值摘要、CLR为有序序列，容量/接口/优化仍混杂。',
          '问题(2)：all-t组内ρ/大差对排序acc/选子ΔV̂(pp)，题bootstrap95%CI：'+ '；'.join(f"{v} {ci(bv(v,'spearman'))}/{ci(bv(v,'pair_accuracy'))}/{ci(bv(v,'gain_vs_sigma0'),scale=100)}" for v in values)+'。同序跨题AUROC为'+ '/'.join(num(bv(v,'between_auroc')['mean']) for v in values)+'；逐t见详细表。'+b_answer,
          f"问题(3)：greedy {ci(greedy,'accuracy',100,low='accuracy_ci_low',high='accuracy_ci_high')}%。"+search_answer+f" 共{len(search)}设置同时检验；比较为相同平均预算上限，greedy可闲置余额。横轴为逐lane源前向+价值矩阵FLOPs换算；prefill记一次，非严格等FLOPs或等延迟。oracle=终点gold选择=pass@N，仅特权参考；N16全步pass@N为{ci(ref,'accuracy',100,low='accuracy_ci_low',high='accuracy_ci_high')}%。",
          '问题(4)：'+d_answer,
          ('搜索收益检验：通过（有全网格同时区间下界>0的设置）。' if len(sig) else '搜索收益检验：未通过（全网格同时区间无正下界）。')+' test隔离关卡：未通过（早期无效A开发运行提前test暴露，保留并排除）。',
          '## (c) 核心主表（test / reader_val）',
          'clean/seen、t等权、三seed均值；Q3/Q4为%，Q1 NLL↓/Q2 AUROC↑。完整95%seed-t区间与reader_val Q1 clean/traj在详细报告/CSV。',
          table(['split','model','Q1','Q2','Q3','Q4'],compact),
          '独立cal_B阈值、α=.1：各格为集合大小/实际覆盖率(%)，均值口径同上；其CI见详细报告。可交换性等假设与分布偏移限制仍适用。',
          table(['split','model','Q1','Q2','Q3','Q4'],compactcp),
          '经验90%同覆盖率集合大小的配对差及95%CI（负值更小）；逐qtype补充M4.1。每次bootstrap重估阈值，使用评估标签，仅描述效率；实际cal_B集合size/coverage另见CSV。',matchedtable,
          '## (d) 本里程碑新增的偏离与阻塞',
          '早期A冻结输入干预混用raw/标准化状态且未重训，已读取test；整run保留并排除，正式重训超参预锁定，未据该test选设置。B3特征缓存OOM后显式移CPU，科学设置不变；目录写入等失败及费用保留。详情见报告与invalid_development_runs.json。仅A正式最终评估可test，B/C/D均未触碰test。',
          '## (e) 需要审查者决定的问题','请审查本轮结论与上述A历史test暴露；无新增运行授权请求。',
          '## (f) 证据路径',
          '`reports/M5b_details.md`；`data/M5b_v1/analysis/`（A_*.csv、B_summary.csv逐t、C_summary.csv全网格/同时CI、C_budget_curves.csv与accuracy_compute.png/pdf、D_B_paired.csv、D_C_paired.csv、costs.csv/json）。',
          '`data/M5b_v1/completion.json`索引全部run；`docs/M5b_TASK.md`；`data/M5b_v1/freeze.json`；`data/M5b_v1/frozen_inputs_verification.json`；`reports/M5b_tests_final.log`。']
        text='\n\n'.join(review)+'\n'
        if len(text.encode())>8000:raise ValueError(f'Review too large: {len(text.encode())} bytes; shorten prose before delivery')
        Path('reports/M5b_REVIEW.md').write_text(text)
        write_json(out/'index.json',{'files':[str(x) for x in sorted(out.iterdir()) if x.is_file()], 'detailed_report':'reports/M5b_details.md','review':'reports/M5b_REVIEW.md'})
        write_json(root/'report_complete.json',{'run':str(run.path),'review_bytes':len(text.encode()),'costs':costs,'best_observed':best.to_dict(),'significant_settings':len(sig),'search_success':len(sig)>0})
        run.write_metrics(status='completed',gpu_hours=0,wall_seconds=time.monotonic()-start,source_forward_calls=0,test_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

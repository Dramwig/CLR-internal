"""One user-facing report, assembled exclusively from completed evidence."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from clr.utils.config import load_config

NAMES={'CLR-lite':'CLR','A1':'w/o block mask','A2':'dot-product head','A3':'w/o numeric feat.','A4':'late fusion','CLR-v1':'pretrained encoder','CLR-lite-single':'single step z_t','A5':'w/o type weighting','CLR-per-type':'one model per type','CLR-lite-multi':'CLR, joint','B3':'MLP probe','B3-prefix':'MLP probe, prefix','B6':'Attention probe','B4':'LTO reward model','HSRM':'HSRM-style reader','CLUE':'CLUE'}
COLS=[('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy'),('Q3','size')]

def num(x,signed=False):
    if abs(x)<1e-12:return '0'
    return format(x,'+.3g' if signed else '.3g')

def ci(r,scale=1,level=95,signed=True):
    lo,hi=('lo','hi') if level==95 else ('lo90','hi90')
    return f"{num(scale*r['mean'],signed)} [{num(scale*r[lo])}, {num(scale*r[hi])}]"

def table(headers,rows):
    esc=lambda x:str(x).replace('|','\\|')
    return '\n'.join(['|'+'|'.join(map(esc,headers))+'|','|'+'|'.join(['---']*len(headers))+'|',*['|'+'|'.join(map(esc,r))+'|' for r in rows]])+'\n'

def main(c):
    root=Path(c['output']);get=lambda n:json.loads((root/n).read_text());a1=get('a1.json');a2=get('a2.json');a3=get('a3_validation.json');a4=get('a4.json');rep=get('reproduction.json');audit=get('table_number_audit.json');pro=get('provenance.json')
    b=get('b_results.json') if (root/'b_analysis_completed.json').exists() else None
    aindex={(r['variant'],r['qtype'],r['metric']):r for r in a1};bindex={(r['variant'],r['qtype'],r['metric']):r for r in b if r['split']=='test'} if b else {}
    base='/home/me/PaperField/CalReader';new=base+'/'+str(root)
    report=['# CLR 论文补充统计与 LLaMA 组件消融',f"\n生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。所有差值为配对点估计 [区间下界, 上界]。除 TOST 特别列出的 90% 区间外，均为 95% 区间。"]
    report+=['\n## 1. 结论摘要',
    '- **A1：**40 个变体×指标比较中，唯一“显著更好”是单类型模型的 Q3 集合大小：'+ci(aindex['CLR-per-type','Q3','size']['delta'])+'；其他表面改善均不显著。',
    '- **A2：**Q3/Q4 accuracy 及集合大小全部通过预先固定的 ±2.0pp / ±0.10 等价性检验；accuracy 差值分别为 '+ci(a2[0]['delta'])+'pp、'+ci(a2[2]['delta'])+'pp。',
    '- **A3：**有 v1/lite 的预定义 validation 二选一规则；没有 single/default/joint 三行统一择优规则。最终修复架构是在 v0 test 已被观察之后设计，不能称为未见 test 的预注册配置。',
    '- **A4：**相对实测最强单类型 reader，GPT-2/LLaMA 的 Q1 NLL 分别高 0.0415/0.0706，Q2 AUROC 分别低 0.0277/0.0308，四项差距均显著。']
    if b:
        report+=['- **B：**LLaMA 三 seed 实验完成，Q3 的 dot-product / late fusion / single-step 差值依次为 '+'；'.join(ci(bindex[v,'Q3','accuracy']['delta'],100)+'pp' for v in ['A2','A4','CLR-lite-single'])+'；新增 '+num(get('b_completed.json')['gpu_hours'])+' GPUh，源前向 0。']
    else:report+=['- **B：**实验运行中，尚无最终 test 结论；本报告仍为在研稿。']
    report+=['\n## 2. 统计实现核对',
    '直接调用原 `clr/paper/statistics.py → scripts/m6_report.py::cluster_estimate` 复现；新扩展仅暴露同一批抽样的 90% 分位数。10,000 次、NumPy `default_rng(11)`，按 test problem 有放回抽样，保留题内所有 step 和三个固定训练 seed。先在 step 内按 QA 数微平均，再对 step 等权、固定 seed 平均。AUROC 在每次抽样内重新计算，处理并列分数，不能用逐题 AUC 或两个独立区间相减。区间条件于固定权重及原校准集，不覆盖重新训练的 seed 不确定性。CLUE 为一次确定性评估，每次使用同一题抽样与 CLR 配对，不制造独立三 seed。',
    table(['CLR 减比较对象','类型','本次复现（pp）','任务给定/历史记录（pp）'],[[r['baseline'],r['qtype'],f"{num(r['estimate_pp'])} [{num(r['lo_pp'])}, {num(r['hi_pp'])}]", num(r['estimate_pp'] if r['expected'][0] is None else r['expected'][0])+' ['+num(r['expected'][1])+', '+num(r['expected'][2])+']'+('（点值本次恢复）' if r['expected'][0] is None else '')] for r in rep]),
    '最大差异 0.0384pp，低于 0.1pp 停止阈值，核对通过。上述为 M10/M10b 原导出的复放；verbalizer 的精确核对值来自原 M10b 记录（任务仅要求其区间含零）；任务给定 CLR−MLP Q4 为 7.93 [6.31, 9.56]pp，本次为 7.93 [6.35, 9.52]pp，保留这个小差异。计数矩阵加速与原函数最大差异 <3×10⁻¹⁷；A1 全部 45 格与冻结 M12 汇总最大误差 '+format(get('a_validation.json')['maximum_absolute_error'],'.3g')+'。',
    f"**数字差异审计：**Table 1/2 共 {audit['cells']} 个已报告数字，在各自显示精度上全部一致；没有需要替换的表值。完整精度、123 格审计及所有输入 SHA 保存在本目录。所有“显著”均为未校正的逐比较 95% 区间判断；A1 的 40 项及 B 的 25 项消融比较均属探索性分析。"]
    report+=['\n## 3. A1–A4 的结果表','\n### A1. 消融的双向配对区间',
    '下表均为 **变体−CLR**。Q3/Q4 accuracy 差值为百分点；NLL、AUROC、集合大小用原尺度。基准 CLR 的绝对值为 0.548 / 0.830 / 76.1% / 76.2% / 1.49。Q1/集合大小越低越好，其余越高越好。',
    table(['变体','Q1 NLL','Q2 AUROC','Q3 acc（pp）','Q4 acc（pp）','|C_Q3|'],[[NAMES[v]]+[ci(aindex[v,q,m]['delta'],100 if m=='accuracy' else 1)+'；'+aindex[v,q,m]['classification'] for q,m in COLS] for v in ['CLR-lite','A1','A2','A3','A4','CLR-v1','CLR-lite-single','A5','CLR-per-type']]),
    '**所有显著更好的格子：**仅 one model per type × |C_Q3|，'+ci(aindex['CLR-per-type','Q3','size']['delta'])+'。去掉数值特征及 late fusion 的 Q2 点值更高，但区间跨零。集合大小比较使用各自冻结阈值，不能解释为等覆盖率下的效率优势；pretrained/single/per-type 为历史复用对照，不能全当成严格单组件替换。',
    '\n### A2. 与 1.7B verbalizer 的等价性检验',
    '**计算之前已写入本报告的固定容差：accuracy ±2.0pp；集合大小 ±0.10，此后未调整。**采用双单侧 α=.05 的 bootstrap 区间版 TOST：90% 区间须严格位于容差开区间内。',
    table(['类型/指标','CLR−verbalizer：95% CI','CLR−verbalizer：90% CI','预设容差','等价','最小容差阈值¹'],[[r['qtype']+'/'+('accuracy（pp）' if r['metric']=='accuracy' else 'set size'),ci(r['delta']),ci(r['delta'],level=90),'±'+str(r['tolerance']),'是' if r['equivalent'] else '否',f"{r['minimum_margin_infimum']:.6g}"] for r in a2]),
    '¹ 阈值为 max(|90%下界|, |90%上界|) 的下确界；严格判等价需略大于该值。六位有效数字仅为显示，未用来调整预设容差。完整精度见 a2.json。这里是 bootstrap 近似检验，不是额外的正态 t 检验。',
    '四项均等价。集合大小仍存在小而显著的差异：Q3 '+ci(a2[1]['delta'])+'、Q4 '+ci(a2[3]['delta'])+'，CLR 更大；“统计显著不同”与“在实用容差内等价”可以同时成立，不能称两者完全相同。',
    '\n### A3. 默认配置的选择依据',
    '下表为各设计原有 **reader_val、clean/seen、step 等权、三 seed** 验证结果；准确率为 %，其余原尺度。均为点估计，95% 区间见 a3_validation.json。',
    table(['源','配置','Q1 NLL','Q2 AUROC','Q3 acc（%）','Q4 acc（%）','|C_Q3|'],[[r['source'],NAMES[r['model']]]+[num(r[k]['mean']*(100 if k in ['Q3','Q4'] else 1)) for k in ['Q1','Q2','Q3','Q4','size']] for r in a3]),
    '**规则与边界：**M4.1_TASK 预先规定 v1/lite 完整训练，以 reader_val 平均 NLL 选择较好变体。实现取三 seed 各自 eligible best checkpoint 的未校准、四类型宏平均 NLL 再平均；最早 epoch 8、最多 10、patience 1。记录为 lite=0.558942、v1=0.618356，选 lite。该选择分数含原训练验证的 Q1 clean/traj，不能从上表的已校准 clean 指标反推。M5b single-step 和 M9 joint 在以后才产生，未找到把这三行放在同一预定义 validation 规则下选择默认行的记录；上表也显示默认行不在每项上最佳。',
    '**test 时间线：**M4 v0 主 seed11 最终评估于北京时间 2026-11-04 10:14:42 完成；11-05 的 M4.1 失效诊断与修复任务之后才设计最终 v1/lite 架构。M4.1 内部确实在本轮 test 前以 reader_val 选 checkpoint 并锁定，但它复用了已经观察过 v0 结果的同一 test。故“最终配置在从未看过 test 的条件下确定”不成立；“本轮调试只用 val、权重锁后最终评测”与这段历史必须同时披露。',
    '\n### A4. Q1/Q2 与单类型 reader 的差距',
    '下表均为 **默认 CLR−比较 reader**，NLL/AUROC 原尺度。按 Table 1 的六个已指定 reader，在各源各指标上标记实测最优者；这是对已观察 test 表的描述性选取，未做“挑最优 reader”后的同时区间校正。',
    table(['源','比较 reader','Δ Q1 NLL [95%CI]','Δ Q2 AUROC [95%CI]'],[[s,NAMES[m],ci(next(r for r in a4 if r['source']==s and r['model']==m and r['qtype']=='Q1')['delta']),ci(next(r for r in a4 if r['source']==s and r['model']==m and r['qtype']=='Q2')['delta'])] for s in ['gpt2','llama'] for m in ['B3','B3-prefix','B6','B4','HSRM','CLUE']]),
    'Q1 最强者：GPT-2 为 LTO（0.507），LLaMA 为确定性的 CLUE（0.545）；对应 CLR 差值分别 '+ci(next(r for r in a4 if r['source']=='gpt2' and r['model']=='B4' and r['qtype']=='Q1')['delta'])+'、'+ci(next(r for r in a4 if r['source']=='llama' and r['model']=='CLUE' and r['qtype']=='Q1')['delta'])+'。Q2 两源最强者均为 MLP-prefix（0.857 / 0.852），差值分别 '+ci(next(r for r in a4 if r['source']=='gpt2' and r['model']=='B3-prefix' and r['qtype']=='Q2')['delta'])+'、'+ci(next(r for r in a4 if r['source']=='llama' and r['model']=='B3-prefix' and r['qtype']=='Q2')['delta'])+'。',
    '**可直接用于论文：**“在两个 CODI 源上，默认 CLR 相比各指标实测最优的单类型 reader，Q1 NLL 高 0.0415–0.0706，Q2 AUROC 低 0.0277–0.0308，配对 95% 区间均排除零；CLR 的优势主要体现在状态与选项关联的 Q3/Q4。”不能把这些结果写成已证明与最强 specialist 等价。']
    report+=['\n## 4. B：LLaMA 关键消融',
    '开跑前估计 **7.52 GPUh**：按历史 M12 同变体三 seed 耗时 × LLaMA/GPT-2 QA 数比 0.340932 × 1.7 原生维度/共享 GPU 余量，另留 1.50 GPUh 给 smoke/校准/评估/失败。低于 25 GPUh，故执行全部缺失设计。训练为原 M12 协议：batch1024、bf16、AdamW 3×10⁻⁴ / weight decay10⁻⁴、clip1、每 epoch 全 QA 无放回、类型加权、完整 reader_val 宏 NLL 早停，eligible epoch8、最多10。仅把源维度改为 LLaMA 2048，复用其冻结 normalization；未调参。']
    if b:
        old_q1=pd.read_csv('data/M10_v1/paper_inputs/reader_val_q1_clean_traj.csv')
        old_q1=old_q1[(old_q1.source=='llama')&(old_q1.metric=='nll')]
        q1_b=get('b_reader_val_q1_clean_traj.json')
        q1_values={(r['model'],r['kind']):r['mean'] for r in q1_b}
        q1_values.update({(r.model,r.kind):r['mean'] for _,r in old_q1.iterrows()})
        exposure_notes=[]
        for v in ['A1','A2','A4']:
            total_epochs=[];selected_epochs=[]
            for seed in [11,22,33]:
                run=Path(get(f'trained_{v}_lite_{seed}.json')['run'])
                total_epochs.append(str(len(json.loads((run/'history.json').read_text()))))
                selected_epochs.append(str(json.loads((run/'best.json').read_text())['epoch']))
            exposure_notes.append(NAMES[v]+' 实际 '+ '/'.join(total_epochs)+' epoch、选中 '+ '/'.join(selected_epochs))
        report+=['w/o block mask、dot-product、late fusion 新增各三 seed；单类型模型复用 M10 已完成的三 seed、每类型精确 8 epoch / matched exposure，12 个类型×seed checkpoint 与曝光均已核验；single-step 按任务复用原结果。所有九个新训练完成后统一锁定权重/代码/config，才打开最终 test。cal_A 温度、cal_B 按(qtype,t) Mondrian、α=.1，与 GPT-2 相同。',
        '完整训练曝光（按 seed11/22/33 顺序）：'+'；'.join(exposure_notes)+'。每个 epoch 无放回遍历全部四类型 QA，未缩量。',
        '**test 绝对结果**（clean/seen、step 等权、三 seed；accuracy 为 %）：',
        table(['设计','Q1 NLL','Q2 AUROC','Q3 acc（%）','Q4 acc（%）','|C_Q3|'],[[NAMES[v]]+[num(bindex[v,q,m]['value']['mean']*(100 if m=='accuracy' else 1)) for q,m in COLS] for v in ['CLR-lite','A1','A2','A4','CLR-per-type','CLR-lite-single']]),
        '**变体−CLR 的双向配对区间**（Q3/Q4 为 pp，其余原尺度）：',
        table(['设计','ΔQ1 NLL','ΔQ2 AUROC','ΔQ3 acc','ΔQ4 acc','Δ|C_Q3|'],[[NAMES[v]]+[ci(bindex[v,q,m]['delta'],100 if m=='accuracy' else 1)+'；'+bindex[v,q,m]['classification'] for q,m in COLS] for v in ['A1','A2','A4','CLR-per-type','CLR-lite-single']]),
        '**reader_val 结果**（同样 clean/seen、step 等权；各项区间见 b_results.json）：',
        table(['设计','Q1 NLL','Q2 AUROC','Q3 acc（%）','Q4 acc（%）','|C_Q3|'],[[NAMES[v]]+[num(next(r for r in b if r['split']=='reader_val' and r['variant']==v and r['qtype']==q and r['metric']==m)['value']['mean']*(100 if m=='accuracy' else 1)) for q,m in COLS] for v in ['CLR-lite','A1','A2','A4','CLR-per-type','CLR-lite-single']]),
        'Q1 reader_val 的 clean/traj 分项 NLL（点估计）：',
        table(['设计','clean','traj'],[[NAMES[v],num(q1_values[v,'clean']),num(q1_values[v,'traj'])] for v in ['CLR-lite','A1','A2','A4','CLR-per-type','CLR-lite-single']]),
        '**跨源一致性：**下列比较点估计方向，同时注明两源均显著的项；不把单源不显著与另一源显著自动当成已证明的交互效应。']
        for v in ['A1','A2','A4','CLR-per-type','CLR-lite-single']:
            same=[];opposite=[];both=[]
            for q,m in COLS:
                aa=aindex[v,q,m];bb=bindex[v,q,m];label='|C_Q3|' if m=='size' else q
                (same if aa['delta']['mean']*bb['delta']['mean']>0 else opposite).append(label)
                if aa['classification']!='不显著' and bb['classification']==aa['classification']:both.append(label+' '+aa['classification'])
            report+=['- '+NAMES[v]+'：同向 '+('、'.join(same) or '无')+'；反向 '+('、'.join(opposite) or '无')+'；两源均显著且方向相同：'+('、'.join(both) or '无')+'。']
        report+=['新增实测成本 **'+num(get('b_completed.json')['gpu_hours'])+' /25 GPUh**（RTX 4090，按各 GPU run 墙钟求和，含加载、smoke、校准、验证和最终推断；CPU 统计为 0 GPUh）。源模型前向 0。Q1 reader_val clean/traj 分项保存在 b_reader_val_q1_clean_traj.json。']
    else:report+=['尚在执行。训练期间未读取新变体 test 指标，不填入中间结果。']
    report+=['\n## 5. 与论文现有说法的对照',
    '1. **“CLR 与 1.7B verbalizer 在 Q3/Q4 上 statistically tied”——需改写。**accuracy 的 95% 区间均跨零，且在预先固定 ±2.0pp 容差内通过等价性检验；集合大小也在 ±0.10 内等价，但 CLR 显著大约 0.017/0.019 个元素。建议明确写“accuracy 在 ±2pp 实用容差内等价”，避免用 tied 暗指完全相同或同等训练成本。B5c 一遍训练，CLR 为多遍，曝光亦不同。',
    '2. **“CLR 在 Q1/Q2 上与 specialist readers competitive”——需改写。**四项相对最强者的配对区间都显示显著差距（A4）。可保留具体差距描述，不支持“与最强者无差别”或无容差的“性能相当”。',
    '3. **“dot-product head、late fusion、single step 各使 Q3 下降 2–3 个点”——需改写。**GPT-2 的三个差值依次为 '+ '；'.join(ci(aindex[v,'Q3','accuracy']['delta'],100)+'pp' for v in ['A2','A4','CLR-lite-single'])+'，其中 dot-product 下降超过 3pp。'+ ('LLaMA 对应为 '+'；'.join(ci(bindex[v,'Q3','accuracy']['delta'],100)+'pp' for v in ['A2','A4','CLR-lite-single'])+'，须分别陈述，不能直接外推 GPT-2 结论。' if b else 'LLaMA 结论等待最终评估。'),
    '4. **“联合训练使 Q4 提升，Q3 不变”——需改写。**这里应明确指 GPT-2 的多类型联合训练相对 matched-exposure 每类型独立模型，并非 Table 1 的双源 joint。GPT-2 多类型−单类型：Q4 '+ci({k:-aindex['CLR-per-type','Q4','accuracy']['delta'][j] for k,j in [('mean','mean'),('lo','hi'),('hi','lo')]},100)+'pp；Q3 '+ci({k:-aindex['CLR-per-type','Q3','accuracy']['delta'][j] for k,j in [('mean','mean'),('lo','hi'),('hi','lo')]},100)+'pp。应写“Q4 提升，Q3 未检出显著差异”，未给 Q3 另设等价性容差，不能写已证明不变。'+('LLaMA 多类型−单类型：Q4 '+ci({k:-bindex['CLR-per-type','Q4','accuracy']['delta'][j] for k,j in [('mean','mean'),('lo','hi'),('hi','lo')]},100)+'pp；Q3 '+ci({k:-bindex['CLR-per-type','Q3','accuracy']['delta'][j] for k,j in [('mean','mean'),('lo','hi'),('hi','lo')]},100)+'pp；应分别报告两个源。' if b else ''),
    '5. **“pretrained encoder 增加参数但不提升准确率”——需改写。**本次实现 trainable 参数为 v1 '+num(pro['parameters']['v1']['trainable_parameters']/1e6)+'M / lite '+num(pro['parameters']['lite']['trainable_parameters']/1e6)+'M；GPT-2 Q3 差值 '+ci(aindex['CLR-v1','Q3','accuracy']['delta'],100)+'pp，Q4 '+ci(aindex['CLR-v1','Q4','accuracy']['delta'],100)+'pp。准确表述为“本次预训练 v1 未检出 Q3 收益，Q4 显著下降”；它与 lite 同时在深度/参数化等处不同，不能归因预训练本身，更不能外推所有 encoder。',
    '\n## 6. 复现信息',
    f'仓库根目录：`{base}`；启动 git commit：`9f0a9975378fd91807149c183599ff818853b52a`。工作区已有未提交材料；复现以每 run 启动保存的源码 tar、SHA/config、环境与 seed 为准，不把 git commit 冒充全部新脚本的版本。新脚本和产物均在 `{new}`，未修改已有代码、预测、结果或论文。',
    '**输入文件及来源：**',
    f'- A1：`{base}/data/M12_ablation/{{problem_metrics.parquet,auroc_observations.parquet,conformal_observations.parquet,summary.csv}}`；原始 run/校准来源见该目录 source_registry.json / statistics_provenance.json。',
    f'- A2：上项 CLR，加 `{base}/data/M10b_v1/paper_inputs/B5c_{{problem_metrics,conformal_problem}}.csv`；与 `{base}/data/M10c_v1/paper_inputs/conformal_paired.csv` 独立核对方向与区间。',
    f'- A3/A4：`{base}/data/M10_v1/paper_inputs/{{main,main_problem_metrics,conformal}}.csv` 及 paper_cache；`{base}/data/M10e_v1/{{summary,problem_metrics_*,final_*}}`；`{base}/data/M10d_v1/{{problem_metrics.csv,final.json}}`；GPT-2 LTO 原始 run 由 data/M4_v0/evaluation_suite.json 的 B4_11/22/33 索引。',
    f'- B：`{base}/data/M6_v1/qa`、`{base}/data/dumps/codi_llama_M6_v1`、`{base}/data/M6_v1/llama_normalization.npz`；每类 train QA=252399/47719/293301/95228。新 run 索引为 `{new}/trained_*.json`、`evaluated_*.json`；复用单类型来源 data/M10_v1/trained/llama_CLR-per-type_*_full.json。',
    '**脚本及实际命令**（在仓库根目录，沿用 `source scripts/m7_env.sh` 设置；Python 为 `/home/me/anaconda3/envs/latdefer/bin/python`，每条入口只传 YAML）：',
    '```bash\npython -m supplemental.stats_20261116_v1.reproduce supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.analysis_a supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.audit_tables supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.validate_a supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.provenance supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.b_pipeline supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.analysis_b supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.write_report supplemental/stats_20261116_v1/config.yaml\npython -m supplemental.stats_20261116_v1.finalize supplemental/stats_20261116_v1/config.yaml\n```',
    f'最终核验：[final_audit.json]({new}/final_audit.json)；冻结清单：[freeze.json]({new}/freeze.json)。以上为实际运行命令；重跑时须复制到新版本目录并更新 YAML 输出路径，避免覆盖本次冻结产物。',
    'B 配置最初由 prepare_b.py 生成并在训练前记录预算；b_train.py / b_evaluate.py 为 M12 入口的新版本，差异补丁随目录保留，模型仅新增 native 维度包装并 import 冻结实现。首次复现入口因 NumPy bool JSON 序列化退出，CPU 导出亦修正了旧里程碑索引/预测 schema 适配，原失败日志保留；均没有模型回退、重新校准旧结果或改变统计协议。',
    f"A 部分新增 GPUh=0；B 成本以 b_completed.json / jobs 下逐 run metrics 为准。协议 smoke：5 个 reader_val problem、四类型原生主模型逐位一致；训练每变体 reader_train20 / reader_val20 题，校准评估代理仅 reader_val30 题（≤3%），各 run ≤15 分钟；九项新训练全部锁定后才最终 test。旧科学输入已核验 {pro['verified_scientific_input_files']} 个文件 SHA 一致，最终再次核验见 final_audit.json。输入全路径/SHA、完整精度结果、每 seed 曝光、全部运行命令/环境保存在本目录。",
    '本补充分析不能消除历史 test 暴露；显著性和等价性均条件于当前固定模型、划分及原校准数据。冻结阈值下的经验集合大小不是分布偏移或自适应选择后的无条件 conformal 保证。']
    (root/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    print('Report written',root/'REPORT.md',len((root/'REPORT.md').read_bytes()),flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]))

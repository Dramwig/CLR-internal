"""Write bounded review/claim documents from the completed frozen CSV build."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from clr.utils.m9_runtime import hours


def fmt(x):return 'NA' if pd.isna(x) else f'{float(x):.3g}'


def markdown_blocks(lines):
    """Keep Markdown tables/lists contiguous, with blank lines around blocks."""
    out = ''
    previous = ''
    for line in lines:
        contiguous = ((previous.startswith('|') and line.startswith('|')) or
                      (previous.startswith('- ') and line.startswith('- ')))
        if out:
            out += '\n' if contiguous else '\n\n'
        out += line
        previous = line
    return out + '\n'


def ci(r, value='mean', low='lo', high='hi', scale=1):
    return f'{fmt(scale*r[value])} [{fmt(scale*r[low])},{fmt(scale*r[high])}]'


def get(f, **terms):
    for k,v in terms.items():f=f[f[k]==v]
    if len(f)!=1:raise ValueError(f'Expected exactly one evidence row: {terms}; n={len(f)}')
    return f.iloc[0]


def documents(c):
    root=Path(c['output']);inputs=root/'paper_inputs'
    read=lambda name:pd.read_csv(inputs/(name+'.csv'))
    main=read('main');cp=read('conformal_paired');source=read('single_source_paired')
    det=read('communication_detection');qa=read('communication_qa');qap=read('communication_qa_paired')
    shift=read('shift');ranking=read('ranking');search=read('search_settings');transfer=read('source_transfer_paired')
    type_transfer=read('type_transfer_paired');localization=read('localization')
    dependent=read('dependent');minimum=read('minimum_cost');q4=read('equal_coverage_paired')
    pair=pd.read_csv(root/'analysis_A/multisource_paired.csv')
    reproduction=json.loads((root/'reproducibility.json').read_text())
    audit=json.loads((root/'upstream_audit.json').read_text())
    lock=json.loads((root/'training_lock.json').read_text())
    gpu=sum(r['gpu_hours'] for r in hours())
    expected=['multi_lite_11','multi_lite_22','multi_lite_33']
    if sorted(lock['models'])!=expected:raise ValueError('Three complete training seeds required')
    if not reproduction['passed']:raise ValueError('Clean reproduction must pass')
    d=lambda kind,alpha=.05:get(det,panel='full',split='test',kind=kind,k=16 if kind in ['CLR','B3','linear'] else 0,alpha=alpha,category='counterfactual')
    clr,b3,norm,text=[d(k) for k in ['CLR','B3','norm','text']]
    advantages=[]
    for s in ['gpt2','llama']:
        z=[]
        for q in ['Q3','Q4']:
            z.append(q+' '+ci(get(source,source=s,split='test',qtype=q,metric='accuracy'),
                              'delta_lite_minus_b3','ci_low','ci_high',100)+'pp')
        advantages.append(s+'：'+ '；'.join(z))
    small=transfer[(transfer.split=='test')&transfer.qtype.isin(['Q3','Q4'])&
                   transfer.n_labels.astype(str).isin(['100','300'])]
    weak=get(ranking,value='CLR-lite',t='all',metric='spearman')
    across=get(ranking,value='CLR-lite',t='all',metric='between_auroc')
    eligible=search[search.method.isin(['S2','S3'])&~search.value.isin(['oracle','pass@N'])]
    best=eligible.sort_values('accuracy',ascending=False).iloc[0]
    sh=shift[(shift.alpha==.1)&shift.qtype.isin(['Q3','Q4'])]
    standard=sh[sh.method=='split'];weighted=sh[sh.method=='weighted']
    matched=standard.merge(weighted,on=['model','qtype','shift'],suffixes=('_s','_w'),validate='one_to_one')
    weighted_change=(matched.coverage_minus_target_s.abs()-matched.coverage_minus_target_w.abs()).median()*100
    dep=dependent[(dependent.split=='test')&(dependent.rule=='lambda_rows')&
                  (dependent['lambda'].astype(str)=='1.0')&(dependent.alpha==.05)]
    ratios={r.design:r for _,r in minimum.iterrows()}
    msize=[q+': '+ci(get(q4,a='CLR-lite',b='B3',split='test',qtype=q),
                     'delta','ci_low','ci_high') for q in ['Q3','Q4']]
    deployed_sizes=[]
    for source_name in ['gpt2','llama']:
        deployed_sizes.append(source_name+' '+', '.join(q+' '+ci(get(cp,source=source_name,split='test',qtype=q,
            model='CLR-lite',baseline='B3')) for q in ['Q3','Q4']))
    newtype=get(type_transfer,a='transfer_100_lite',b='scratch_b3_100',split='test',qtype='Q4',metric='accuracy')
    transfer_examples=[]
    for direction in ['gpt2->llama','llama->gpt2']:
        r=get(transfer,direction=direction,n_labels='100',baseline='b3',split='test',qtype='Q3')
        transfer_examples.append(direction+' Q3 '+ci(r,'delta_transfer_minus_scratch','ci_low','ci_high',100)+'pp')
    dep_first={design:get(dep,design=design,epsilon=.01) for design in ['D0','R4']}
    worst=standard.sort_values('coverage').iloc[0]
    local_only=get(localization,split='test',task='A4',mode='critical_only')
    local_removed=get(localization,split='test',task='A4',mode='critical_remove')
    claims=[
      {'id':'C1','claim':'类型化读取在状态与选项交互任务上优于所比较探针',
       'evidence':'T1；'+ '；'.join(advantages)+'。T2 cal_B集合size差（覆盖率不等）：'+ '；'.join(deployed_sizes)+
                  '。M5b冻结CSV（equal_coverage_paired.csv）中GPT2同覆盖率描述性size差 '+ '；'.join(msize)+
                  '。F1：GPT2新类型Q4、N100的迁移−scratch B3 '+ci(newtype,'delta','ci_low','ci_high',100)+'pp。',
       'limits':'限Q3/Q4与两个CODI源；新类型迁移仅GPT2。容量、表示、优化与曝光仍混杂。Q1/Q2没有全面优势；描述性同覆盖率结果使用评估标签，不能作为部署保证。'},
      {'id':'C2','claim':'冻结主体、只训适配器可实现已测试方向的小样本跨模型读取',
       'evidence':f'F1：N100/300、Q3/Q4的{int((small.ci_low>0).sum())}/{len(small)}个已执行配对95%CI下界>0；例如N100对B3：'+ '；'.join(transfer_examples)+'。',
       'limits':'只测GPT2与LLaMA两个方向；N是QA标签数；B5b未覆盖整个网格；不能宣称所有基线/所有预算全胜。'},
      {'id':'C3','claim':'类型化内容子问题支持非自适应counterfactual消息审计',
       'evidence':f'T4：CF检测@名义5%：CLR {ci(clr,scale=100)}%，B3 {ci(b3,scale=100)}%，norm {ci(norm,scale=100)}%，text300 {ci(text,scale=100)}%；CLR实测FPR {fmt(100*clr.FPR)}%。'+
                  'A4仅保留/去掉关键记录的准确率 '+ci(local_only,scale=100)+'% / '+ci(local_removed,scale=100)+'%。',
       'limits':'程序化commitment、synthetic v2、弱receiver。A2/A3近随机；监督A3参照未学会，训练目标与曝光混杂。不能将差距单独归因于分解。text只测300组；零事件区间不是总体率为零。'},
      {'id':'C4','claim':'多数已测试偏移的经验覆盖接近目标，加权方法未见系统收益',
       'evidence':f'T3：alpha=.1、Q3/Q4，{int((standard.coverage<.88).sum())}/{len(standard)}个standard设置低于88%；最低覆盖率 '+
                  ci(worst,'coverage','coverage_ci_low','coverage_ci_high',100)+
                  f'%；weighted绝对覆盖偏差中位改善 {fmt(weighted_change)}pp（描述性）。',
       'limits':'这是经验结果。交换性不成立的任意偏移下无保证；source-only Mondrian缺stratum返回全集，target-cal使用额外数据；weighted跨step支持不重叠。'},
      {'id':'N1','claim':'跨题价值可分不等于可用于同父状态候选排序，搜索未建立正收益',
       'evidence':f'F2：CLR跨题AUROC {ci(across,"mean","ci_low","ci_high")}；组内rho {ci(weak,"mean","ci_low","ci_high")}。最高观察搜索差 {ci(best,"delta_greedy","simul_ci_low","simul_ci_high",100)}pp（全网格同时CI，val300）。',
       'limits':'组内相关仍显著大于零，不能说完全只有题目难度；也不能据最高观察设置作后验选型。预算是源调用加FLOPs等价，不是严格同延迟。'},
      {'id':'N2','claim':'内容审计不能提供白盒KV扰动的完整性保证',
       'evidence':'F3：依赖receiver E3，epsilon=.01的D0/R4失效为 '+
                  '/'.join(ci(dep_first[design],'failure_mean','failure_low','failure_high',100)+'%' for design in ['D0','R4'])+
                  '；两者点估计包络至epsilon=1均持平'+
                  '；R4原始点为 '+', '.join(fmt(100*r.failure_mean)+'%' for _,r in dep[dep.design=='R4'].sort_values('epsilon').iterrows())+'。',
       'limits':f'有限PGD给未知最坏失效的经验下界；R4消费后才检查。最小扰动仅{int(ratios["R4"].complete_groups)}/100完整组，条件比值中位数{fmt(ratios["R4"].q50)}、P90={fmt(ratios["R4"].q90)}，不是认证最小值。传输完整性保护是部署建议，本项目未测试其实现。'}
    ]
    cannot=[
      '所有问题类型都优于探针，或优势已被证明由类型化接口单独造成。',
      '所有模型/所有小样本预算/所有基线的跨模型读取全胜；同一主体已跨推理与通信联合迁移。',
      '93%是白盒攻击检测率，或端到端一致性判断在一般意义上无效。',
      '源/数据集/步骤偏移或自适应攻击下拥有无条件conformal保证。',
      '随机问题数带来已验证的指数鲁棒性，或R4提供显著扰动成本放大/认证鲁棒性。',
      '搜索已获得正收益，或有限PGD失效率是已认证的最坏情况上界。',
    ]
    summary=['# 最终证据总结','实验冻结至M9；除M9-A外未新增模型实验。以下CI均沿用所列统计口径，不将提案/理论预期写成实验结论。',
             '主要QA/检测CI按problem或反事实group bootstrap 10000次，先保留组内步骤与固定三seed；不覆盖重新训练seed的不确定性。F1类型迁移点沿用三seed t区间；其配对检验另列。']
    for r in claims:summary.extend(['\n## '+r['id']+' '+r['claim'],'证据：'+r['evidence'],'局限：'+r['limits']])
    summary.extend(['\n## M9-A联合训练',f'三seed、共享主体/各源适配器/源token；新增GPU成本 {fmt(gpu)}h≤10，源前向0。单源序列CLR-lite与单状态CLR-lite-single是不同对照。'])
    for s in ['gpt2','llama']:
        vals=[]
        for q in ['Q3','Q4']:
            a=get(pair,source=s,split='test',qtype=q,metric='accuracy')
            vals.append(q+' 联合−单源 '+ci(a,'delta_joint_minus_single','ci_low','ci_high',100)+'pp')
        summary.append(s+'：'+'；'.join(vals)+'。')
    summary.extend(['\n## 不能主张',*['- '+x for x in cannot],
      '\n## 证据与复现','T1–T5、F1–F3及附录：paper/README.md；机器主张索引：paper/claims.json；冻结清单：paper/artifact_manifest.json。',
      '上游历史偏离、早期test暴露、源协议差异和成本缺口保留于对应M4.1/M5b/M6/M7.1/M8审查报告，不因论文整理而消失。'])
    final='\n\n'.join(summary)+'\n'
    if len(final.encode())>10000:raise ValueError('FINAL_SUMMARY exceeds 10KB')
    Path('reports/FINAL_SUMMARY.md').write_text(final)
    Path('paper/claims.json').write_text(json.dumps({'claims':claims,'cannot_claim':cannot},ensure_ascii=False,indent=2)+'\n')

    # Reviewer table 1: full metric/CI comparison, reader_val and test together.
    lines=['# M9 审查','\n### (a) 状态一句话',
      'M9-A三seed联合训练/最终评估与论文产物完成待审查；实验至此冻结，未新增A以外模型实验。',
      '\n### (b) 关卡清单',
      f'- 通过：两源各5训练/val题smoke，70.7秒；选项置换、未观测状态隔离、源/共享梯度、划分及conformal测试。上游{audit["unique_files"]}个唯一文件初核SHA一致。',
      '- 通过：M6冻结数据/归一化/10epoch、min8、patience1协议只读复用；三seed全训练后hash锁定，再cal_A温度/cal_B conformal及最终test。',
      f'- 通过：A累计{fmt(gpu)}/10 GPUh，包含加载/smoke/训练/校准/评估；源模型前向0。',
      f'- 通过：独立无torch/transformers的制图环境；{len(reproduction["artifacts"])}份表/图/数值文件隔离重建逐字节一致。每个产物有独立脚本/YAML/输入CSV SHA。',
      '- 未通过（既有证据缺失，按任务不补实验）：LLaMA的B0、B3-prefix、CLR-lite-single；对应T1单元格及Q3−B0为NA。',
      '\n### (c) 核心主表（最多3张）',
      '表1：clean/seen、逐t等权、固定三seed；95%题bootstrap。single=各源序列CLR-lite；joint=共享主体两源模型。准确率为%。',
      '|源/集|模型|Q1 NLL|Q2 AUROC|Q3 acc%|Q4 acc%|','|---|---|---|---|---|---|']
    for s in ['gpt2','llama']:
      for split in ['reader_val','test']:
       for model,name in [('CLR-lite','single'),('CLR-lite-multi','joint')]:
        vals=[ci(get(main,source=s,split=split,model=model,qtype=q,metric=m),scale=sc)
              for q,m,sc in [('Q1','nll',1),('Q2','auroc',1),('Q3','accuracy',100),('Q4','accuracy',100)]]
        lines.append('|'+ '/'.join([s,split])+'|'+name+'|'+'|'.join(vals)+'|')
    lines.extend(['\n表2：joint−single配对差；NLL为原单位，准确率为pp，集合大小为cal_B部署集合（α=.1）；95%题bootstrap。',
                  '|源/集|ΔQ1 NLL|ΔQ3 acc pp|ΔQ4 acc pp|ΔQ3 size|ΔQ4 size|','|---|---|---|---|---|---|'])
    for s in ['gpt2','llama']:
     for split in ['reader_val','test']:
      vals=[ci(get(pair,source=s,split=split,qtype=q,metric=m),'delta_joint_minus_single','ci_low','ci_high',sc)
            for q,m,sc in [('Q1','nll',1),('Q3','accuracy',100),('Q4','accuracy',100)]]
      vals += [ci(get(cp,source=s,split=split,qtype=q,model='CLR-lite-multi',baseline='CLR-lite')) for q in ['Q3','Q4']]
      lines.append('|'+s+'/'+split+'|'+'|'.join(vals)+'|')
    lines.extend(['\n论文主线按四贡献/两局限逐项收窄：Q3/Q4优势与已执行小N跨源迁移成立；CF 93.1%限非自适应；偏移仅经验覆盖。搜索、白盒安全及不完整基线不能升格为正结论。详见≤10KB的FINAL_SUMMARY。',
       '\n### (d) 本里程碑新增的偏离与阻塞',
       '仅调度修订：保留seed11 worker，停止其纯CPU顺序控制器；seed22/33在GPU0/4并行，科学设置/10GPUh不变。execution_supplement留痕。',
       'CPU导出曾因M0.1区间字段、旧预测缺选项失败；保留失败run，按冻结QA补元数据并逐项复放原conformal指标，未改模型/阈值。初版协议CSV保留，最终v2使用实际冻结噪声选择，未用旧启动YAML残留sigma作结果。',
       '独立制图环境首次代理连接失败，直连安装成功，主环境未改。系统无LaTeX引擎，未执行TeX编译；booktabs/longtable片段及重建字节已核查。M0/M0b及部分早期成本记录不全，T5标下界/缺失，不给项目总成本。',
       'M9联合训练比较沿用各源完整训练集，标签目标/数据量和共享主体优化不同；只支持观察到的差异，不能单独归因于多源机制。',
       '\n### (e) 需要审查者决定的问题','无新增运行授权问题；实验已冻结，结论待审查。',
       '\n### (f) 证据路径',
       'reports/M9_details.md；reports/FINAL_SUMMARY.md；paper/README.md、claims.json、artifact_manifest.json；paper/tables/*.tex、figures/*.pdf；',
       'data/M9_v1/（analysis_A/*.csv、paper_inputs/freeze.json、training_lock.json、evaluation_index.json、experiment_complete.json、reproducibility.json、freeze.json）；configs/M9/；scripts/paper/；reports/M9_evidence/。'])
    review=markdown_blocks(lines)
    if len(review.encode())>8000:raise ValueError('M9 review exceeds 8KB')
    Path('reports/M9_REVIEW.md').write_text(review)
    details=['# M9 详细证据','完成待审查；唯一审查交付为reports/M9_REVIEW.md。',
             '## 协议与预算',f'累计GPUh={gpu:.12g}；源前向0。模型为原M6 MultiLite，两源适配器/source token与共享主体；每源完整冻结训练集，逆(source,qtype)频率损失，按全reader_val平均NLL选checkpoint，min_epochs=8。',
             '旧控制器只在等待子进程时停止；seed11原训练保持。未重启训练、未改变batch/精度/曝光。原控制器保留running历史快照，执行补充说明其已停止；它没有GPU计费。',
             '沿用的M6实现仅从epoch>=8的候选保存best checkpoint；epoch1–7的val损失只作曲线诊断，不能事后选更早epoch。联合模型的训练规模/加权损失与单源不同，不能据差异作纯机制因果归因。',
             '## 模型与曝光（保存的实际记录）']
    for name,item in lock['models'].items():
        ref=json.loads((root/'trained'/('multi_'+str(item['config']['seed'])+'.json')).read_text())
        r=Path(ref['run']);metrics=json.loads((r/'metrics.json').read_text());bestrow=json.loads((r/'best.json').read_text())
        details.extend(['### '+name,'run：'+str(r),'checkpoint SHA：'+item['sha256'],
                        json.dumps({'best':bestrow,'metrics':metrics},ensure_ascii=False,indent=2)])
    details.extend(['## 完整统计与图表','analysis_A/metrics.csv含Q1 reader_val clean/traj/all及所有类型；multisource_paired.csv是joint−single配对统计；T1补Q2 AUROC的题bootstrap；T2的部署集合覆盖率与size配对另列。',
                    '所有脚本仅从paper_inputs冻结CSV读数；其provenance.json给出原始CSV/JSON/已保存预测SHA。表图文件与数值派生CSV逐字节重建。PDF固定日期、字体和元数据；没有光栅数据图层。',
                    'T2的size配对不能宣称同覆盖率；GSM8K匹配覆盖率描述来自M5b原冻结统计，使用评估标签，不是部署阈值。',
                    'F1左图沿用原seed-t CI，右两图为固定三seed的题bootstrap；B5b仅N100，缺预算不画假点。F2搜索预算包络由val全网格得出，只作描述，非独立选型/测试证据。',
                    'F3包络取已执行较小半径的最高均值；浅色为原始点，不重新运行攻击。weak D0为M8a四任务分数，R1–R4为M8b统一A1/A4，不能把D0与其微小差异作同口径改进。',
                    'T5从关闭的里程碑账本与历史保存记录导出，复用模型不重复计原训练；legacy行只给可追溯下界。物理batch forward、搜索per-lane等价计算和梯度调用不是同一单位。',
                    '## 新增工程修复','M0.1确定/随机CI字段分别解析；旧预测从冻结QA按problem/t/traj/template/replicate补回选项，重建conformal逐项校验。首次未通过导出不产生模型调用。协议元数据v1保留但不用于论文；v2选择实际噪声协议。',
                    '## 历史限制','各前序review已记录的测试暴露、synthetic programmatic commitment、弱receiver、适配器目标混杂、偏移可交换性与有限PGD限制全部保留。没有通过新划分、选seed或新的攻击实验改善结果。',
                    '## 机器主张索引','paper/claims.json与reports/FINAL_SUMMARY.md按四贡献/两个负面发现组织，每条都含证据及边界；不能支持的措辞独立列出。'])
    Path('reports/M9_details.md').write_text('\n\n'.join(details)+'\n')
    return {'review_bytes':len(review.encode()),'summary_bytes':len(final.encode()),'gpu_hours':gpu,'claims':len(claims)}


def readme(c):
    cfg=yaml.safe_load(Path('configs/M9/paper.yaml').read_text());root=Path(c['output']);freeze=json.loads((root/'paper_inputs/freeze.json').read_text())
    lines=['# CalReader 论文产物','实验截止M9。除M9-A两源联合训练外，论文整理没有新增模型实验。唯一审查文件为`reports/M9_REVIEW.md`。',
      '## 一键重建','使用Python 3.11、独立且不含torch/transformers的制图环境：',
      '```bash\npython3.11 -m venv .venv-paper\n.venv-paper/bin/python -m pip install -r paper/requirements.lock.txt\n.venv-paper/bin/python scripts/paper/make_all.py configs/M9/paper.yaml\n```',
      '从仓库根目录执行。每个脚本只接受YAML路径。输入不匹配冻结SHA即报错。重建不会加载源模型或读取器，也不改阈值。',
      'PDF使用矢量图形、嵌入TrueType字体、统一配色/字号及固定元数据。TeX片段需`booktabs`、`graphicx`、`longtable`；当前系统没有LaTeX引擎，因此未声称已编译。',
      '## 表图与脚本','|产物|脚本|YAML|输入CSV及SHA-256|','|---|---|---|---|']
    for item in cfg['artifacts']:
        p=Path(item['config']);a=yaml.safe_load(p.read_text())
        ins='<br>'.join(f'`{v}`<br>`{freeze["sha256"][v]}`' for v in a['inputs'].values())
        suffix='.pdf' if a['artifact'].startswith('f') else '.tex'
        folder='figures' if suffix=='.pdf' else 'tables'
        lines.append(f'|`paper/{folder}/{a["artifact"]}{suffix}`|`{item["script"]}`|`{p}`|{ins}|')
    lines.extend(['## 统计与命名口径',
      '- T1/T2：clean、seen模板，先在t内按QA实例计量，再跨t/固定三seed平均；95%题bootstrap 10000次。Q1 reader_val clean/traj另列。Q1的两源续算策略不同。',
      '- CLR-lite是各源序列读取器；CLR-lite-single是M5b单状态控制；CLR-lite-multi是M9两源共享主体。缺失LLaMA B0/prefix/single控制为NA。',
      '- T2部署集合大小差的覆盖率可以不同。FINAL_SUMMARY引用的同覆盖率结果来自M5b冻结CSV，使用评估标签重估阈值，只是描述性效率。',
      '- Q1 conformal校准每题取一个哈希固定的Bernoulli rollout结果；评估覆盖率是集合在保存的rollout频率目标下的概率质量，不是连续价值区间覆盖率。',
      '- F1左图沿用原三seed t区间；右两图为题bootstrap，N计QA标签，B5b只在N100执行。未连接缺失预算。',
      '- T3给出alpha=.1的全部冻结偏移设置；source-only Mondrian与target-calibration分别列出。偏移下没有无条件覆盖保证。',
      '- T4 latent/norm为test2000，text仅300；原CSV包含matched_text300配对面板。各类统一程序化commitment；QA的A2/A3近随机，监督A3参照有训练目标/曝光混杂。',
      '- F2后两图为val300分析；搜索同平均预算上限、greedy可闲置余额；privileged pass@N不能部署。冻结全网格预算包络不是独立测试选择证据。',
      '- F3取每个epsilon及所有更小已执行半径的最大均值，浅色点是原始有限搜索。weak D0为M8a四任务，R1–R4为M8b A1/A4分数，未宣称严格同口径差；dependent为M8c主200组三seed/固定lambda=1。',
      '- R1独立veto指数线是另一种决策规则的理论参考；实际均值NLL+conformal不满足其独立性/veto假设。',
      '- T5记录缺失、历史上界/下界及forward计数单位，不汇总成完整项目总成本。源调用是physical batched forward，非搜索FLOPs等价或每样本计数。',
      '## 冻结与复现证据','`data/M9_v1/paper_inputs/provenance.json`索引派生前原始CSV、JSON与已保存预测的SHA；输入CSV冻结于同目录`freeze.json`。',
      '`paper/artifact_manifest.json`汇总所有正式freeze.json和额外历史manifest，并索引本次脚本/配置/产物SHA。当前M9冻结不自包含该manifest，以避免循环哈希；其SHA另列于verification.json。',
      '`data/M9_v1/reproducibility.json`记录独立制图环境、隔离第二目录重建和逐字节比较；`paper/build_records/`包含每个脚本的源代码/输入哈希与软件版本。',
      '历史协议CSV v1保留为开发证据；论文使用`protocol_alignment_v2.csv`，以实际冻结选择代替启动YAML残留字段。旧审查中的测试暴露和协议局限仍有效。'])
    Path('paper/README.md').write_text(markdown_blocks(lines))

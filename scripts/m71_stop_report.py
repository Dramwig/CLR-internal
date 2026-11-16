"""Only the registered two-attempt C1 stop branch; never open test."""
import sys
sys.dont_write_bytecode=True
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from clr.audit.features import sha256
from clr.audit.m71_statistics import gate_intervals
from clr.audit.m71_runtime import snapshot,live
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def fmt(x):return format(float(x),'.3g')


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);rows=[];runpaths=[];unexecuted=[]
        for rep in c['attempts']:
            result=json.loads((root/rep/'C1.json').read_text())
            if result['passed']:raise ValueError('Stop branch forbidden when C1 passed')
            if not result.get('C1_executed',True):
                unexecuted.append(rep+'因'+result['failed_gate']+'门槛未过，未执行C1')
            for kind,item in result['results'].items():
                m=item['metrics'];runpaths.append(Path(item['run']))
                prediction=torch.load(Path(item['run'])/'reader_val_predictions.pt',weights_only=True,map_location='cpu')
                uncertainty=gate_intervals(prediction)
                write_json(run.path/(rep+'_'+kind+'_C1_group_bootstrap.json'),uncertainty)
                ci=uncertainty['metrics']
                rows.append({'representation':rep,'kind':kind,'A4_accuracy':m['A4']['accuracy'],
                             'A4_lo':ci['A4']['accuracy']['lo'],'A4_hi':ci['A4']['accuracy']['hi'],
                             'A1_AUROC':m['A1']['AUROC'],'A1_lo':ci['A1']['AUROC']['lo'],
                             'A1_hi':ci['A1']['AUROC']['hi'],'passed':item['passed'],'run':item['run']})
        if len(c['attempts'])!=2:raise ValueError('Exactly one registered alternative required')
        costs=[]
        for p in Path('runs').glob('*/config.json'):
            cfg=json.loads(p.read_text())
            if cfg.get('milestone')!='M7.1':continue
            metrics=json.loads((p.parent/'metrics.json').read_text())
            if metrics.get('test_opened',False):raise ValueError('Test was opened before the stop branch')
            costs.append({'run':str(p.parent),'status':metrics.get('status'),'gpu_hours':metrics.get('gpu_hours',0),
                          'source_forward_calls':metrics.get('source_forward_calls',0)})
        out=root/'analysis';out.mkdir(exist_ok=False)
        pd.DataFrame(rows).to_csv(out/'C1_results.csv',index=False);pd.DataFrame(costs).to_csv(out/'costs.csv',index=False)
        gpu=sum(row['gpu_hours'] for row in costs);calls=sum(row['source_forward_calls'] for row in costs)
        if gpu>30:raise ValueError('Budget exceeded')
        selection=json.loads((root/c['selection_name']).read_text());layers=selection['layers']
        table='|表示|模型|reader_val A4准确率% [95%CI]|A1 AUROC [95%CI]|C1|\n|---|---|---|---|---|\n'
        for row in rows:table+=f'|{row["representation"]}|{row["kind"]}|{fmt(100*row["A4_accuracy"])} [{fmt(100*row["A4_lo"])},{fmt(100*row["A4_hi"])}]|{fmt(row["A1_AUROC"])} [{fmt(row["A1_lo"])},{fmt(row["A1_hi"])}]|{"通过" if row["passed"] else "未通过"}|\n'
        branch_note='；'.join(unexecuted) if unexecuted else '初始与备选C1均完整执行'
        review=f'''### (a) 状态一句话
M7.1按预登记停止规则结束、待审查：初始C1及一次备选准入均未通过；未启动CLR主训练/检测比较，test未打开，M8未启动。

### (b) 关卡清单
- 通过：独立环境、冻结生成器import及三SHA；可微f的真实raw K/V梯度；val12组smoke；33项M71/M7最小测试。
- 通过：初始36层扫描按train1000组的800/200组内划分；选{layers}层，锚点{selection['ranking'][0]}。所选组合投影A1/A4准确率损失{fmt(100*selection['projection_loss']['A1_accuracy'])}/{fmt(100*selection['projection_loss']['A4_accuracy'])}pp。
- 未通过：C1要求linear与B3在全train8000组训练后reader_val1000组A4≥45%、A1 AUROC≥.75；{branch_note}，已有结果见表。备选固定同组同split，不按结果换seed或降低门槛。
- 未通过（按停止规则未执行）：C2过拟合、三seedCLR主比较、cal/test检测、natural诊断、监督参照。源管线与传信M7证据保持冻结，不能替代M7.1可学性关卡。
- 成本{fmt(gpu)} GPUh≤30，前向{fmt(calls)}；含smoke/失败/两次表示，详细成本见CSV。

### (c) 核心主表（最多3张）
仅reader_val关卡结果，seed11；按原/CF组配对bootstrap10000次95%CI，条件于固定checkpoint。无test或三seed主比较，不填推测值。

{table}
### (d) 本里程碑新增的偏离与阻塞
初始scan高lr损失振荡，保留后按train loss降低lr重新核验；选层/分组规则不变。source扩展入口解析/配置错误及共享GPU显存预检查失败保留，未换模型或静默缩量。fp16文件采用gzip无损编码，12组3.56e6元素跨设备/存储一致100%、max误差0。
程序化commitment统一三模板；部署时模型承诺的差距继续成立。备选依赖生成器critical记录边界，结论只限这一定位元数据可得的设置。QA标签来自消息来源事实，不保证源模型已完整编码；本关卡失败不能当作有效的CLR负面比较。
M7 2912/M6 35688冻结文件及生成器SHA核验一致。文本cal_A/B300、natural val100/test300已获用户授权，但由于C1停止未执行。

### (e) 需要审查者决定的问题
无。四问：(1)可学性未通过；层扫描只表示当前探针可读性，不能证明私有信息仅集中于这些层。(2)FPR1%/5%下CLR/B3/范数/文本/监督参照检测率均未执行。(3)没有CLR−B3比较，无法判断与GSM8K Q4/Q3的一致性。(4)监督参照差距未测，不能量化通用审计的代价。

### (f) 证据路径
reports/M7.1_details.md；data/M71_v1/analysis/C1_results.csv、costs.csv；两次C1.json与其run索引；层选择{root/c['selection_name']}；源码/config/env启动快照runs/；冻结data/M71_v1/freeze.json；接续docs/M71_LIVE.json。
'''
        review_path=Path('reports/M7.1_REVIEW.md');review_path.write_text(review)
        if review_path.stat().st_size>8000:raise ValueError('Review exceeds 8000 bytes')
        Path('reports/M7.1_details.md').write_text('# M7.1 C1停止分支\n\n'+table+'\n\n'+
            '每组含真实与CF消息；模型选择只用reader_val NLL，未使用test。\n\n'+json.dumps({'selection':selection,'costs':costs},indent=2,ensure_ascii=False))
        write_json(root/'completion.json',{'run':str(run.path),'status':'stopped_after_registered_C1_alternative',
            'C1_passed':False,'D_executed':False,'test_opened':False,'gpu_hours':gpu,'source_forward_calls':calls})
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False)
        message='M7.1按一次备选后仍未准入的停止规则结束、待审查；test未打开，未训练CLR主比较。唯一审查reports/M7.1_REVIEW.md；冻结data/M71_v1/freeze.json，接续docs/M71_LIVE.json。'
        for name in ['AGENTS.md','docs/STATUS.md']:
            path=Path(name);path.write_text(re.sub(r'<!-- M71_LIVE -->.*?<!-- /M71_LIVE -->','<!-- M71_LIVE -->\n'+message+'\n<!-- /M71_LIVE -->',path.read_text(),flags=re.S))
        for name,content in [('EXPERIMENTS','初始及一次备选全train8000/val1000 C1均未通过；未打开test，按任务停止，无CLR比较结论。'),
                             ('BLOCKERS','初始/一次备选C1不通过，缺CLR主比较、检测与监督参照；不得作为CLR劣势证据。'),
                             ('CHANGELOG','新增压缩特征管线与足量C1关卡；按停止规则归档，M7/M6冻结保持不变。')]:
            with Path('docs/'+name+'.md').open('a') as f:f.write('\n## 2026-11-07 — M7.1 C1停止分支\n'+content+'\n报告reports/M7.1_REVIEW.md；完整证据reports/M7.1_details.md、data/M71_v1/analysis/。\n')
        live(stage='C1_stop_report_ready_pending_freeze',milestone_complete=False,test_opened=False,review=str(review_path))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

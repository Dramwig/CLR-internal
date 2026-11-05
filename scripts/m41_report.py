"""Final M4.1 report, preserving all frozen M4 artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m41_budget import hours_used
from clr.calib.m41_statistics import paired,cp_paired
from scripts.m4_report import interval,fmt,table

METRICS={'Q1':'nll','Q2':'auroc','Q3':'accuracy','Q4':'accuracy'}
def main(c):
 with recorded(c) as run:
    root=Path(c['output']);suite=json.loads((root/'evaluation_complete.json').read_text())['runs'];selected=json.loads((root/'variant_selection.json').read_text())['selected'];chosen='main_'+selected
    stats=[];cp=[]
    for key,directory in suite.items():
        path=Path(directory);m=json.loads((path/'metrics.json').read_text())
        if m['status']!='completed':raise ValueError('Incomplete evaluation '+directory)
        stats+=json.loads((path/'statistics.json').read_text());cp+=json.loads((path/'conformal.json').read_text())
    df=pd.DataFrame(stats);cf=pd.DataFrame(cp);df.to_csv('reports/M41_metrics.csv',index=False);cf.to_csv('reports/M41_conformal.csv',index=False)
    primary=df[(df.kind=='clean')&(~df.heldout)&(df.subset=='all')];primarycp=cf[(~cf.heldout)&(cf.subset=='all')]
    def vals(case,split,q,metric,cal='temperature'):
        frame=primary[(primary.case==case)&(primary.split==split)&(primary.qtype==q)&(primary.calibration==cal)];v=frame.groupby('seed')[metric].mean().reindex(c['seeds']).to_numpy()
        if not np.isfinite(v).all():raise ValueError(f'Missing metric {case}/{split}/{q}/{metric}')
        return v
    def cpvals(case,split,q,metric):
        frame=primarycp[(primarycp.case==case)&(primarycp.split==split)&(primarycp.qtype==q)&(primarycp.calibration=='temperature')&(primarycp.alpha==.1)];return frame.groupby('seed')[metric].mean().reindex(c['seeds']).to_numpy()
    agg=[]
    for keys,g in df.groupby(['case','split','qtype','kind','heldout','calibration','subset']):
        for metric in ['nll','brier','mae','accuracy','auroc','ece15']:
            v=g.groupby('seed')[metric].mean().dropna().to_numpy()
            if len(v)==3:
                m,h=interval(v);agg.append({**dict(zip(['case','split','qtype','kind','heldout','calibration','subset'],keys)),'metric':metric,'mean':m,'ci95_low':m-h,'ci95_high':m+h})
    pd.DataFrame(agg).to_csv('reports/M41_aggregate.csv',index=False)
    rng=np.random.default_rng(20261105);pairs=[];cpairs=[]
    comparisons=[(a,b,q,metric) for a in ['main_v1','main_lite'] for b in ['B3','main_b3_multi'] for q,metric in METRICS.items()]
    comparisons += [(f'transfer_{n}_{selected}',f'scratch_b3_{n}','Q4','accuracy') for n in [100,1000,10000]]
    for split in ['reader_val','test']:
        for a,b,q,metric in comparisons:
            pairs.append(paired(suite,a,b,c['seeds'],split,q,metric,rng));write_json(run.path/'paired_progress.json',pairs)
        for a in ['main_v1','main_lite']:
            for q in METRICS:cpairs+=cp_paired(c,suite,a,'B3',split,q,rng)
    pd.DataFrame(pairs).to_csv('reports/M41_paired_bootstrap.csv',index=False);pd.DataFrame(cpairs).to_csv('reports/M41_cp_paired_bootstrap.csv',index=False)
    def diff(a,b,q):return next(r for r in pairs if (r['a'],r['b'],r['qtype'],r['split'])==(a,b,q,'test'))
    def dtext(a,b,q):
        d=diff(a,b,q);return f"{d['delta']:.3g} [{d['ci_low']:.3g}, {d['ci_high']:.3g}]"
    mainrows=[];cases=['B0','B3','main_b3_multi','main_v1','main_lite']
    for case in cases:
        for split in ['reader_val','test']:mainrows.append([case,split]+[fmt(vals(case,split,q,m)) for q,m in METRICS.items()]+[fmt(vals(case,split,'Q3','accuracy')-vals('B0',split,'Q3','accuracy'))])
    tab1=table(['model','split','Q1 NLL','Q2 AUROC','Q3 acc','Q4 acc','Q3−B0'],mainrows)
    tab2=table(['qtype','test B3 cov/size','test v1 cov/size','test lite cov/size'],[[q]+[f"{cpvals(case,'test',q,'coverage').mean():.3g}/{cpvals(case,'test',q,'size').mean():.3g}" for case in ['B3','main_v1','main_lite']] for q in METRICS])
    tab3=table(['N','val CLR','val B3','test CLR','test B3'],[[n]+[fmt(vals(f'transfer_{n}_{selected}' if model=='CLR' else f'scratch_b3_{n}',split,'Q4','accuracy')) for split,model in [('reader_val','CLR'),('reader_val','B3'),('test','CLR'),('test','B3')]] for n in [100,1000,10000]])
    figdir=Path('reports/M41_figures');figdir.mkdir(exist_ok=True);fig,ax=plt.subplots(figsize=(6,4))
    for label,prefix in [(selected,'transfer'),('B3','scratch_b3')]:
        points=[interval(vals(f'transfer_{n}_{selected}' if prefix=='transfer' else f'scratch_b3_{n}','test','Q4','accuracy')) for n in [100,1000,10000]];ax.errorbar([100,1000,10000],[p[0] for p in points],yerr=[p[1] for p in points],marker='o',label=label)
    ax.set(xscale='log',xlabel='Q4 labels',ylabel='Test accuracy');ax.legend();fig.tight_layout();fig.savefig(figdir/'transfer.png',dpi=180);plt.close(fig)
    trace=pd.read_csv('reports/M41_module_trace.csv');fig,ax=plt.subplots(figsize=(7,4))
    for name in ['adapter','cross','numeric','heads']:
        part=trace[trace.module==name];ax.plot(part.step,np.maximum(part.grad_norm,1e-12),label=name)
    ax.set(yscale='log',xlabel='Diagnostic replay step',ylabel='Gradient L2 norm');ax.legend();fig.tight_layout();fig.savefig(figdir/'module_gradients.png',dpi=180);plt.close(fig)
    advantages=any(diff(chosen,'B3',q)['ci_low']>0 for q in ['Q3','Q4']);dominance=False
    for q in METRICS:
        pair=[r for r in cpairs if r['a']==chosen and r['split']=='test' and r['qtype']==q];cov=next(r for r in pair if r['metric']=='coverage');size=next(r for r in pair if r['metric']=='size')
        dominance|=cov['ci_low']>=0 and size['ci_high']<0 and cpvals(chosen,'test',q,'coverage').mean()>=.9
    unified=all((diff(chosen,b,q)['ci_high']<=0 if q=='Q1' else diff(chosen,b,q)['ci_low']>=0) for b in ['B3','main_b3_multi'] for q in METRICS)
    transfer=all(diff(f'transfer_{n}_{selected}',f'scratch_b3_{n}','Q4')['ci_low']>0 for n in [100,1000])
    g1=[f"(1) {'通过' if advantages else '未通过'}：val选定{selected}，test CLR−B3 Q3 {dtext(chosen,'B3','Q3')}；Q4 {dtext(chosen,'B3','Q4')}。扣同一B0不改变差值。",f"(2) {'通过' if dominance else '未通过'}：同覆盖率更小集合的配对CI证据{'成立于至少一类型' if dominance else '未成立'}；α=.1，详见CP配对CSV。",f"(3) {'通过' if unified else '未通过'}：Q1 NLL差{dtext(chosen,'B3','Q1')}，Q2 AUROC差{dtext(chosen,'B3','Q2')}；对B3-multi Q3/Q4差{dtext(chosen,'main_b3_multi','Q3')}/{dtext(chosen,'main_b3_multi','Q4')}。未预设非劣界，不能据不显著宣称不劣。",f"(4) {'通过' if transfer else '未通过'}：小N迁移−scratch的Q4差，N100 {dtext('transfer_100_'+selected,'scratch_b3_100','Q4')}；N1k {dtext('transfer_1000_'+selected,'scratch_b3_1000','Q4')}。"]
    costs=hours_used();budget_limit=c['total_budget_hours'];cost_rows=[];exposure_rows=[]
    for config_path in Path('runs').glob('*/config.json'):
        config=json.loads(config_path.read_text())
        if config.get('milestone')!='M4.1':continue
        folder=config_path.parent;metrics_path=folder/'metrics.json';m=json.loads(metrics_path.read_text()) if metrics_path.exists() else {};budget_path=folder/'budget.json';b=json.loads(budget_path.read_text()) if budget_path.exists() else {}
        cost_rows.append({'run':str(folder),'stage':config.get('stage'),'case':config.get('case'),'variant':config.get('variant'),'seed':config.get('seed'),'status':m.get('status'),'gpu_hours':max(m.get('gpu_hours',0.),b.get('own_gpu_hours',0.)),'source_forward_calls':m.get('source_forward_calls'),'gpu_model':'NVIDIA GeForce RTX 4090'})
    pd.DataFrame(cost_rows).to_csv('reports/M41_costs.csv',index=False)
    training=json.loads((root/'training_complete.json').read_text())['runs']
    for key,directory in training.items():
        m=json.loads(Path(directory,'metrics.json').read_text());best=json.loads(Path(directory,'best.json').read_text())
        for q,n in m['examples'].items():exposure_rows.append({'case_seed':key,'qtype':q,'total_examples':n,'selected_examples':best['examples'][q],'optimizer_steps':m['optimizer_steps'],'selected_optimizer_steps':best['steps']})
    pd.DataFrame(exposure_rows).to_csv('reports/M41_exposure.csv',index=False)
    rootcause=(root/'root_cause.md').read_text();affine=[]
    for case in ['main_v1','main_lite','B3']:
        gain=np.mean([np.mean(vals(case,'test',q,'nll')-vals(case,'test',q,'nll','affine')) for q in METRICS]);affine.append(f'{case} {gain:.3g}')
    extras='仿射相对温度的test平均NLL改善：'+ '；'.join(affine)+'。全部ECE、Q1 Brier/MAE、Q4 AUROC、heldout、逐t及clean/traj见CSV。'
    details='\n\n'.join(['# M4.1 完整结果',rootcause,'三seed11/22/33；表内±为seed-t 95%CI，逐t等权。配对差异按题cluster bootstrap10000次，全部步骤一起抽样，平均三个seed；此区间条件于所选三个训练seed，训练随机性另以seed-t区间报告。Q1 reader_val为Bernoulli样本、test为MC均值，MAE对象不同；NLL/Brier沿用M23定义。',tab1,tab2,tab3,*g1,extras,'训练：所有实例每epoch完整访问，以逆qtype频率loss权重保持均衡目标。主模型checkpoint最早8epoch，最多10，取消1万上限。B3-multi同步完整曝光重训；B3/B0及小N scratch复用冻结M4对照。迁移选型仅reader_val，留Q4预训练最多2个完整epoch。','初诊新脚本启动时未跟踪，working-tree.patch没有捕获新文件；066ba79是后续提交，不能冒充启动快照。B恢复/C使用已跟踪代码。B首次GPU2共享显存OOM与lite日志器AttributeError均保留，显式恢复到GPU1/5且模型/batch/精度不变；C三seed固定GPU0/1/5。seed11全量训练OOM后显式启用activation checkpointing，模型/batch/精度/曝光不变；前向及梯度等价测试通过。人为192-token压力测试仍OOM，train/reader_val实际最长48-token审计及对应压力smoke通过（峰值3.56GiB）；未截断生产数据。合成初版B3未过门槛，原记录保留；最终单位L2线性编码控制双方同协议。新模型加入显式状态/数值联合读出与非零cross残差尺度，不能单凭最终优势归因某一组件，可选消融未做。',f'成本合计{costs:.6f}GPUh，源前向0；GPU为NVIDIA GeForce RTX 4090，按各作业墙钟时间求和，含CPU准备、smoke和失败，不按共享GPU利用率打折。明细见reports/M41_costs.csv，新模型实际曝光与更新步数见M41_exposure.csv。总预算{budget_limit}（用户授权由35提高至50）。Mondrian的有限样本保证依赖层内可交换性；这里只报告经验覆盖率，不对分布偏移作无条件保证。','先完成所有模型/迁移训练并锁定checkpoint才最终test。本次使用已在M4报告过的冻结test，修复任务由用户授权，不能宣称该test从未被观察；M4.1调试仅val代理。','证据：data/M41_v1/training_complete.json、evaluation_complete.json、gate_lock.json、diagnosis_complete.json；reports/M41_metrics.csv、M41_conformal.csv、M41_paired_bootstrap.csv、M41_cp_paired_bootstrap.csv。'])
    Path('reports/M4.1_details.md').write_text(details)
    review='\n\n'.join(['## (a) 状态','M4.1实验完成，待G1审查；M4 v0保持冻结。','## (b) 根因与关卡',rootcause,'通过：两候选A4/A5、三seed主比较、val选型与迁移、温度/仿射/Mondrian；实际成本'+f"{costs:.3g}/{budget_limit} GPUh（用户授权由35提高至50）。",*g1,'## (c) 核心主表','均值±95% seed-t CI；clean/seen、逐t等权。表2为α=.1。',tab1,tab2,tab3,extras,'## (d) 新增偏离与阻塞','合成控制初版B3也未达98%，保留全部失败设置；最终同协议单位L2编码门槛。新联合头/残差尺度为显式架构修订，根因不作单算子归因；历史梯度缺失；初诊未跟踪脚本的启动补丁缺失，后续提交不冒充启动快照。均衡目标改为全量遍历加逆类型频率loss权重；迁移预训练最多2epoch，不声称匹配B3全部曝光。可选消融未做。seed11 OOM后仅启用activation checkpointing；等价测试及实际最长48-token压力smoke通过，192-token人工压力仍OOM，未改变模型/batch/精度/曝光。','## (e) 需要审查者决定','依据本次四项G1证据决定后续路线；未自动进入新里程碑。','## (f) 证据路径','reports/M4.1_details.md、M41_budget.md；reports/M41_metrics.csv、M41_conformal.csv、M41_paired_bootstrap.csv、M41_cp_paired_bootstrap.csv；data/M41_v1/training_complete.json、evaluation_complete.json、freeze.json。'])
    if len(review.encode())>8000:raise ValueError(f'Review exceeds 8KB: {len(review.encode())}; details retained, review not overwritten')
    Path('reports/M4.1_REVIEW.md').write_text(review);run.write_metrics(status='completed',gpu_hours=0.,source_forward_calls=0,total_gpu_hours=costs,review_bytes=len(review.encode()),g1=[bool(x) for x in [advantages,dominance,unified,transfer]]);write_json(root/'report_complete.json',{'run':str(run.path)})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

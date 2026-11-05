"""Summarize completed A evidence before starting candidate learning gates."""
import argparse,json
from pathlib import Path
import pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    root=Path(c['output']);pathrun=Path(json.loads((root/'path_diagnostics_run.json').read_text())['run']);paths=json.loads((pathrun/'path_diagnostics.json').read_text())
    gates={}
    for name,p in c['diagnostic_gates'].items():
        entry=json.loads(Path(p).read_text());run=Path(entry['run']);m=json.loads((run/'metrics.json').read_text())
        if m['status']!='completed':raise ValueError('Incomplete required diagnostic '+name)
        gates[name]={'run':str(run),**m}
    flat=[{k:v for k,v in r.items() if k not in ['modules','parameters','option_gradient_norms']} for r in paths];df=pd.DataFrame(flat);df.to_csv('reports/M41_path_diagnostics.csv',index=False)
    mean=df.groupby(['case','qtype']).mean(numeric_only=True);mean.to_csv('reports/M41_path_summary.csv')
    params=[]
    for r in paths:
        for name,value in r.get('parameters',{}).items():params.append({'case':r['case'],'qtype':r['qtype'],'batch':r['batch'],'parameter':name,**value})
    pd.DataFrame(params).to_csv('reports/M41_parameter_norms.csv',index=False)
    trace=[]
    for r in json.loads((pathrun/'module_trace.json').read_text()):
        for name,value in r['modules'].items():trace.append({'step':r['step'],'qtype':r['qtype'],'module':name,**value})
    pd.DataFrame(trace).to_csv('reports/M41_module_trace.csv',index=False)
    roots=['实测所有Q1–Q4的z梯度与置换logit效应非零；每层cross选项可见有效潜状态，padding与选项隔离测试通过，v0没有门控。现有证据不支持完全断路、全屏蔽或零门控这一类实现错误。',
    '存在严重曝光不匹配：主模型最佳checkpoint每类4/4/6万样本，B3 Q3为430/516/516万、Q4各224万（约37–129倍）。CLR优化步更多并不能抵消有效数据曝光不足。',
    f"同一1k Q4诊断集，B3训练准确率{gates['B3_overfit']['train']['accuracy']:.3g}，原v0配置{gates['v0_overfit']['train']['accuracy']:.3g}，降低学习率续训后{gates['v0_lower_lr']['train']['accuracy']:.3g}。这项控制只改变优化设置并继续训练；额外曝光也改变，不能把差异全部归因于学习率。",
    f"可解的单位L2线性整数编码合成任务，B3 Q3/Q4验证准确率{gates['B3_Q3']['validation']['accuracy']:.3g}/{gates['B3_Q4']['validation']['accuracy']:.3g}，原v0为{gates['v0_Q3']['validation']['accuracy']:.3g}/{gates['v0_Q4']['validation']['accuracy']:.3g}。原高维噪声/RMS编码和其他对照的B3失败记录另留，不能据那些失败断言CLR架构损坏。",
    '归因边界：已确认曝光不足，观察到选项条件化学习/优化不充分；尚未定位单个错误算子。修复是假设驱动：非零受控cross残差、显式状态×选项联合非线性头、较低新模块LR与足量曝光；必须经过学习能力门槛与重新比较，不能先宣布修复成功。历史梯度未保存，本次范数曲线是新回放，不是M4原历史。']
    conclusion='\n\n'.join(roots)
    if len(conclusion.encode())>3000:raise ValueError('Root conclusion exceeds 3KB')
    (root/'root_cause.md').write_text(conclusion)
    write_json(root/'diagnosis_complete.json',{'path_run':str(pathrun),'gates':gates,'root_cause_bytes':len(conclusion.encode()),'test_opened':False})
    review='## (a) 状态\n\nA诊断完成；B修复候选待过拟合/合成门槛；未进入C，G1暂不判定。\n\n## (b) 根因结论与关卡\n\n'+conclusion+'\n\n通过：信息通路/掩码诊断、原训练曝光统计。未通过：两个候选学习能力关卡和最终G1（待执行）。\n\n## (c) 核心主表\n\n尚无M4.1最终reader_val/test主比较。\n\n## (d) 新增偏离与阻塞\n\n合成任务初版B3也未达到98%，保留失败记录，最终使用对双方相同且B3可解的单位L2编码控制。原训练梯度无法事后恢复；优化根因不作单因果过度归因。\n\n## (e) 需要审查者决定\n\n暂无；按已授权任务继续B。\n\n## (f) 证据路径\n\n诊断索引data/M41_v1/diagnosis_complete.json；路径/参数CSV reports/M41_path_diagnostics.csv、M41_parameter_norms.csv、M41_module_trace.csv；曝光data/M41_v1/exposure.json；任务docs/M4.1_TASK.md。\n'
    Path('reports/M4.1_REVIEW.md').write_text(review)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

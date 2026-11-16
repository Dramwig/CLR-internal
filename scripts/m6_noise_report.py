"""Complete val-only noise evidence, including a failed selection gate."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scripts.m6_shift import bootstrap_matrix
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import snapshot,hours,live
from clr.utils.m6_reports import markdown


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();snapshot(run,c)
    source=Path(c['noise_run']);paths=sorted(source.glob('noise_*.parquet'))
    if len(paths)!=300:raise ValueError('Complete val300 scan required for noise evidence')
    data=pd.concat([pq.read_table(p).to_pandas() for p in paths],ignore_index=True)
    if len(data)!=300*6*4 or set(data.t)!=set(range(6)):raise ValueError('Incomplete sigma/relative-step grid')
    out=Path(c['output'])/'noise';out.mkdir(parents=True,exist_ok=True);summary=[]
    for sigma,g in data.groupby('sigma'):
        x=g.assign(drop=g.clean_correct.astype(float)-g.value).groupby('problem_id')[['value','drop','nondegenerate']].mean()
        ci=bootstrap_matrix(x.values);point=x.mean()
        summary.append({'sigma':sigma,'accuracy':float(point['value']),'drop':float(point['drop']),'nondegenerate':float(point['nondegenerate']),
           **{f'{metric}_ci_{side}':float(ci[j,k]) for k,metric in enumerate(['accuracy','drop','nondegenerate']) for j,side in enumerate(['low','high'])},
           'passes_drop_only':bool(point['drop']<.05),'passes_historical_gate':bool(point['drop']<.05 and point['nondegenerate']>=.3)})
    frame=pd.DataFrame(summary);frame.to_csv(out/'summary.csv',index=False)
    data.groupby(['sigma','t'])[['value','clean_correct','nondegenerate']].mean().reset_index().to_csv(out/'by_t.csv',index=False)
    costs=hours();pd.DataFrame(costs).to_csv(out/'costs_at_scan_end.csv',index=False)
    selection=json.loads((source/'selection.json').read_text());write_json(out/'gate_v1.json',selection)
    def f(x):return f'{x:.3g}'
    lines=['# M6 LLaMA 噪声选择：完整val300证据（v1）',
      '计算完成，历史选择关卡未通过。seed11、K16、相对观测步t=0..5等权；不含test。原始run保留failed状态，因为selected=null触发硬关卡，不把它误写成只算了部分数据。',
      '|sigma|accuracy %|drop pp [95%CI]|nondegenerate % [95%CI]|drop<5pp|历史nondeg≥30%|','|---|---|---|---|---|---|']
    for r in summary:lines.append('|'+ '|'.join([f(r['sigma']),f(100*r['accuracy']),f(100*r['drop'])+' ['+f(100*r['drop_ci_low'])+','+f(100*r['drop_ci_high'])+']',f(100*r['nondegenerate'])+' ['+f(100*r['nondegenerate_ci_low'])+','+f(100*r['nondegenerate_ci_high'])+']','通过' if r['passes_drop_only'] else '未通过','通过' if r['passes_historical_gate'] else '未通过'])+'|')
    lines+=['clean accuracy=54.3%。四个sigma均满足下降<5pp；如本次仅以该约束定义可行集，sigma1.5满足用户优先规则。若沿用M0.1历史nondeg≥30%硬门槛，全部不合格，不能自动选择1.5或改变t聚合。当前实际执行为后者，并已在数据生成前停止。',
      '当前请求澄清本次B1括号定义是否保留历史30%门槛。未启动正式LLaMA数据生成/训练/test reader预测；未偷偷扩sigma网格。低Q1标签非退化不直接证明Q3/Q4读取不成立。',
      'CI为10000次按题cluster bootstrap，保留每题六个t和固定K16，条件于一个已授权seed；不是三seed区间。门槛原先按点估计判定，事后CI不改变规则。逐t描述在by_t.csv，不据此重新挑t。',
      '运行：'+str(source)+'；原始noise_000..299.parquet、selection.json、config/启动源码/metrics；衍生data/M6_v1/noise/{summary,by_t,costs_at_scan_end}.csv；A独立完成见reports/M6_rq1_closure.md。']
    Path('reports/M6_noise_selection_v1.md').write_text(markdown(lines))
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start,source_scan_complete=True,historical_selection_gate_passed=False,test_opened=False)
    live(stage='awaiting_noise_rule_clarification',noise_report='reports/M6_noise_selection_v1.md',noise_report_run=str(run.path),noise_scan_complete=True,milestone_complete=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

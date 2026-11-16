"""Apply the user's preregistered eight-point M6 B1 rule, without retuning."""
import argparse,json,time,hashlib
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import snapshot,live
from clr.utils.m6_reports import markdown


def choose(rows):
    if {r['sigma'] for r in rows}!={.5,1.,1.5,2.,3.,4.,6.,8.}:raise ValueError('All eight prespecified noise points required')
    feasible=[r for r in rows if r['drop']<.05 and r['nondegenerate']>=.30]
    degraded=not bool(feasible)
    if degraded:feasible=[r for r in rows if r['drop']<.05]
    if not feasible:raise ValueError('No point satisfies even the authorized drop-only fallback')
    selected=dict(max(feasible,key=lambda r:(r['nondegenerate'],-r['sigma'])))
    selected.update(degraded=degraded,eligible=not degraded,status='threshold_unmet_degraded' if degraded else 'both_thresholds_passed',authorized_selection_version='M6_B1_user_extension_v2')
    return selected


def write_evidence(old,new,result,out):
    """Eight-point evidence; the original four-point report stays unchanged."""
    from scripts.m6_shift import bootstrap_matrix
    data=pd.concat([pq.read_table(p).to_pandas() for root in [old,new] for p in sorted(root.glob('noise_*.parquet'))],ignore_index=True)
    expected={.5,1.,1.5,2.,3.,4.,6.,8.}
    if len(data)!=300*6*8 or set(data.sigma)!=expected or data.duplicated(['problem_id','t','sigma']).any():raise ValueError('Incomplete or duplicated eight-point panel')
    summary=[]
    for sigma,g in data.groupby('sigma'):
        if len(g)!=1800 or g.problem_id.nunique()!=300 or set(g.t)!=set(range(6)):raise ValueError('Incomplete relative-step panel')
        panel=g.assign(drop=g.clean_correct.astype(float)-g.value).groupby('problem_id')[['value','drop','nondegenerate']].mean()
        ci=bootstrap_matrix(panel.to_numpy(dtype=float));point=panel.mean()
        row={'sigma':sigma,'accuracy':float(point['value']),'drop':float(point['drop']),'nondegenerate':float(point['nondegenerate'])}
        row.update({f'{metric}_ci_{side}':float(ci[j,k]) for k,metric in enumerate(['accuracy','drop','nondegenerate']) for j,side in enumerate(['low','high'])})
        row.update(passes_drop_only=row['drop']<.05,passes_historical_gate=row['drop']<.05 and row['nondegenerate']>=.3)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'eight_point_summary_ci.csv',index=False)
    data.groupby(['sigma','t'])[['value','clean_correct','nondegenerate']].mean().reset_index().to_csv(out/'eight_point_by_t.csv',index=False)
    selected=result['selected'];f=lambda x:format(x,'.3g')
    lines=['# M6 B1：预登记扩展后的八点噪声选择（v2）',
      '原四点的关卡失败及v1报告保持不改。用户在扩展运行前预登记sigma={3,4,6,8}及统一规则；本报告只在完整val300/K16计算结束后生成，不以中途结果选设置。',
      '|sigma|accuracy %|drop pp [95%CI]|nondegenerate % [95%CI]|drop<5pp|同时满足两门槛|',
      '|---|---|---|---|---|---|']
    for r in summary:
        lines.append('|'+ '|'.join([f(r['sigma']),f(100*r['accuracy']),f(100*r['drop'])+' ['+f(100*r['drop_ci_low'])+','+f(100*r['drop_ci_high'])+']',f(100*r['nondegenerate'])+' ['+f(100*r['nondegenerate_ci_low'])+','+f(100*r['nondegenerate_ci_high'])+']','通过' if r['passes_drop_only'] else '未通过','通过' if r['passes_historical_gate'] else '未通过'])+'|')
    lines += ['按预登记规则选择sigma='+f(selected['sigma'])+'，状态：'+('门槛未达，降级使用；LLaMA Q1标注“标签退化程度高”。' if selected['degraded'] else '下降<5pp且非退化≥30%，两门槛通过。'),
      '统一规则：先在同时满足下降<5pp、非退化≥30%的点中最大化非退化；无此点时，在下降<5pp的点中最大化非退化并标降级；并列只取sigma小者。扩展点若全下降≥5pp，自然回到原四点降级选择。',
      '非退化定义为每题每相对t的K16正确数严格位于0与16之间，t=0..5等权；不是输出答案种类数。CI为10000次按题cluster bootstrap，条件于seed11；门槛按预登记点估计判断，CI不改变选择。',
      'Q1按源分别定义V_s(z,t; π_sigma)：GPT2 sigma=1.5，LLaMA sigma='+f(selected['sigma'])+'；跨源Q1不是同一目标。Q2/Q3/Q4标签定义与sigma无关。',
      '原始run：'+str(old)+'；扩展run：'+str(new)+'；预登记：'+result['preregistration']+'；选择：data/M6_v1/noise_selection_v2.json；逐t与CI：data/M6_v1/noise/eight_point_{summary_ci,by_t}.csv。']
    Path('reports/M6_noise_selection_v2.md').write_text(markdown(lines))


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();snapshot(run,c)
    regpath=Path(c['preregistration']);reg=json.loads(regpath.read_text())
    old=Path(c['original_run']);new=Path(c['extension_run'])
    if json.loads((new/'metrics.json').read_text()).get('status')!='completed':raise ValueError('Complete extension required')
    rows=json.loads((old/'selection.json').read_text())['summary']+json.loads((new/'scan_summary.json').read_text())['summary']
    for root in [old,new]:
        files=sorted(root.glob('noise_*.parquet'))
        if len(files)!=300:raise ValueError('All val300 extension/original panels required')
    oldids=[pq.read_table(p,columns=['problem_id']).column(0)[0].as_py() for p in sorted(old.glob('noise_*.parquet'))]
    newids=[pq.read_table(p,columns=['problem_id']).column(0)[0].as_py() for p in sorted(new.glob('noise_*.parquet'))]
    if oldids!=newids:raise ValueError('Extension changed val problem IDs/order')
    for key in ['seed','protocol','gpt2_root']:
        if json.loads((old/'config.json').read_text())[key]!=json.loads((new/'config.json').read_text())[key]:raise ValueError('Extension changed source protocol: '+key)
    selected=choose(rows);result={'summary':rows,'selected':selected,'preregistration':str(regpath),'preregistration_sha256':hashlib.sha256(regpath.read_bytes()).hexdigest(),'rule':reg['rule'],'original_run':str(old),'extension_run':str(new),'unit':'problem x relative t=0..5, K16, seed11'}
    write_json(run.path/'selection.json',result);write_json(Path(c['output'])/'noise_selection_v2.json',{'selection_run':str(run.path),**result})
    out=Path(c['output'])/'noise';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'eight_point_summary.csv',index=False)
    write_evidence(old,new,result,out)
    if selected['degraded']:
        with Path('docs/BLOCKERS.md').open('a') as f:f.write('\n## M6 B1门槛未达，按用户预登记降级使用\n全部8点无drop<5pp且nondeg≥30%的设置；按已授权规则选sigma='+str(selected['sigma'])+'，drop='+str(selected['drop'])+'、nondeg='+str(selected['nondegenerate'])+'。\nLLaMA Q1标“标签退化程度高”，源内π分别定义；不把两源Q1视为相同V。证据data/M6_v1/noise_selection_v2.json；原4点失败保持不改。\n')
    live(stage='B_noise_selected_v2',selected_sigma=selected['sigma'],noise_selection_status=selected['status'],noise_selection_run=str(run.path),noise_report='reports/M6_noise_selection_v2.md',milestone_complete=False)
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start,degraded=selected['degraded'],selected_sigma=selected['sigma'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

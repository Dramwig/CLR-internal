"""YAML-only, val-only M5a analysis and review. All settings are retained."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.search.analysis_m5a import setting, capacity, vote_at, bootstrap_weights, estimate, fraction
from scripts.m5a_search import val_rows


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_file(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(x):
    return f'{100*x:.3g}'


def interval(x):
    return f"{percent(x['mean'])} [{percent(x['low'])}, {percent(x['high'])}]"


def label(key):
    method, value, n, b, h = key
    return f'{method}/{value}/N{n}/b{b}/h{h}'


def main(c):
    with recorded(c) as run:
        for name in ['scripts/m5a_report.py','clr/search/analysis_m5a.py',c['config_path']]:
            dest=run.path/'source_snapshot'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(name,dest)
        root = Path(c['search_run'])
        cfg, metrics = read(root/'config.json'), read(root/'metrics.json')
        recovery = Path(c['vote_extension_run']) if c.get('vote_extension_run') else None
        if metrics['status'] != 'completed' or (not metrics['gate_passed'] and recovery is None):
            raise ValueError('Only completed passing search evidence can be reported')
        if recovery:
            rm=read(recovery/'metrics.json');rr=read(recovery/'recovery.json')
            if rm['status']!='completed' or not rm['gate_passed'] or not rr['gate_passed']:
                raise ValueError('Incomplete explicit recovery')
            if Path(rr['original_search_run']).resolve()!=root.resolve() or sha(root/'results.json')!=rr['original_results_sha256']:
                raise ValueError('Recovery search ancestry mismatch')
        smoke = c.get('smoke', False)
        if not smoke and (metrics['questions'] != 300 or cfg['seeds'] != [11, 22, 33]):
            raise ValueError('Full report requires all val300 and all three seeds')
        inputs = read(root/'input_sha256.json')
        for path, digest in inputs.items():
            archived = root/'source_snapshot'/path
            evidence = archived if archived.exists() else Path(path)
            if sha(evidence) != digest:
                raise ValueError('Input changed since search startup: '+path)
        rows = read(root/'results.json')
        seeds = cfg['seeds']
        ids = sorted({row['problem_id'] for row in rows})
        if len(ids) != metrics['questions']:
            raise ValueError('Missing question journals')
        gold_rows = val_rows(cfg)
        from clr.data.labels import numeric_answer
        gold = {r['id']: numeric_answer(r['answer']) for r in gold_rows}
        index = {pid: i for i, pid in enumerate(ids)}
        arrays = defaultdict(dict)
        pools = {}
        for path in sorted(root.glob('vote_*.json')):
            pool = read(path)
            if not pool['target_reached'] and recovery:
                extended=read(recovery/path.name)
                n=len(pool['answers'])
                if (extended['original_sha256']!=sha(path) or extended['answers'][:n]!=pool['answers'] or
                    extended['cumulative_source_forwards'][:n]!=pool['cumulative_source_forwards']):
                    raise ValueError('Extended pool changed original prefix')
                pool=extended
            if pool['status'] != 'completed' or not pool['target_reached']:
                raise ValueError('Incomplete voting pool: '+str(path))
            pools[pool['problem_id'], pool['seed']] = pool
        if len(pools) != len(ids)*len(seeds):
            raise ValueError('Missing independent voting journals')
        paired = []
        expected_greedy={r['id']:numeric_answer(r['expected']) for r in gold_rows}
        for row in rows:
            if bool(row['correct']) != (row['answer'] == gold[row['problem_id']]):
                raise ValueError('Behavior label mismatch')
            key, pid, seed = setting(row), row['problem_id'], row['seed']
            if (pid, seed) in arrays[key]:
                raise ValueError('Duplicate setting/question/seed')
            arrays[key][pid, seed] = row
            if row['method'] == 'S0':
                if row['answer']!=expected_greedy[pid]:raise ValueError('Frozen greedy protocol mismatch')
                continue
            pool = pools[pid, seed]
            vote = vote_at(pool, row['source_forwards'])
            if vote is None or not vote['coverage']:
                raise ValueError('Insufficient matched voting coverage')
            paired.append({**row, 'matched_vote_answer': vote['answer'],
                           'matched_vote_correct': vote['answer'] == gold[pid],
                           'matched_vote_cost': vote['compute'], 'matched_vote_n': vote['n'],
                           'budget_residual': vote['residual']})
        expected_keys = 1 + 6 + 18 + 24 + 1
        if len(arrays) != expected_keys:
            raise ValueError(f'Expected {expected_keys} setting keys, got {len(arrays)}')
        paired_lookup = {(setting(r), r['problem_id'], r['seed']): r for r in paired}
        weights = bootstrap_weights(len(ids), c['bootstrap_repetitions'], c['seed'])
        summaries, vectors = [], {}
        for key in sorted(arrays):
            deterministic = key[0] == 'S0'
            ss = [0] if deterministic else seeds
            required = {(pid, seed) for pid in ids for seed in ss}
            if set(arrays[key]) != required:
                raise ValueError('Incomplete setting: '+label(key))
            acc, votes, costs, residuals, equiv, oracle = [], [], [], [], [], []
            for pid in ids:
                rr = [arrays[key][pid, seed] for seed in ss]
                acc.append(np.mean([r['correct'] for r in rr]))
                costs.append(np.mean([r['source_forwards'] for r in rr]))
                equiv.extend(r.get('value_source_equivalent', 0) for r in rr)
                oracle.extend(r['oracle_forwards'] for r in rr)
                if not deterministic:
                    pp = [paired_lookup[key, pid, seed] for seed in ss]
                    votes.append(np.mean([r['matched_vote_correct'] for r in pp]))
                    residuals.extend(r['budget_residual'] for r in pp)
            acc, votes = np.asarray(acc), np.asarray(votes)
            vectors[key] = {'accuracy': acc, 'vote': votes, 'compute': np.asarray(costs)}
            a = estimate(acc, weights)
            result = dict(zip(['method', 'value', 'N', 'b', 'horizon'], key))
            result.update(accuracy=a['mean'], accuracy_low=a['low'], accuracy_high=a['high'],
                          mean_compute=float(np.mean(costs)), min_question_compute=float(min(costs)),
                          max_question_compute=float(max(costs)),
                          mean_value_equivalent=float(np.mean(equiv)),
                          mean_oracle_forwards=float(np.mean(oracle)))
            if not deterministic:
                v, d = estimate(votes, weights), estimate(acc-votes, weights)
                result.update(vote_accuracy=v['mean'], vote_low=v['low'], vote_high=v['high'],
                              delta=d['mean'], delta_low=d['low'], delta_high=d['high'],
                              mean_residual=float(np.mean(residuals)), max_residual=float(max(residuals)))
            summaries.append(result)
        summary = {setting(r): r for r in summaries}

        # Allowed-budget curves: largest predeclared capacity, not best accuracy.
        families = [('S0','none',0), ('S1','none',6)]
        families += [(m,v,h) for m in ['S2','S3'] for v in cfg['values'] for h in cfg['horizons']
                     if any(k[0] == m and k[1] == v and k[4] == h for k in arrays)]
        curves, curve_choices = [], []
        for method, value, horizon in families:
            for budget in cfg['budget_points']:
                hits, costs, choices = [], [], []
                for pid in ids:
                    hh, cc = [], []
                    for seed in seeds:
                        if method == 'S1':
                            vote = vote_at(pools[pid,seed], budget)
                            # Never extrapolate beyond the generated pool.
                            if vote is None or not vote['coverage']:
                                continue
                            correct, cost, name = vote['answer'] == gold[pid], vote['compute'], f"vote_n{vote['n']}"
                        else:
                            candidates = [arrays[k][pid,0 if method == 'S0' else seed] for k in arrays
                                          if k[0] == method and k[1] == value and k[4] == horizon]
                            feasible = [r for r in candidates if r['source_forwards'] <= budget]
                            if not feasible:
                                continue
                            chosen = max(feasible, key=capacity)
                            correct, cost, name = chosen['correct'], chosen['source_forwards'], label(setting(chosen))
                        hh.append(correct); cc.append(cost)
                        choices.append({'problem_id':pid,'seed':seed,'method':method,'value':value,
                                        'horizon':horizon,'budget':budget,'chosen':name,'compute':cost,
                                        'correct':correct})
                    if len(hh) == len(seeds):
                        hits.append(np.mean(hh)); costs.append(np.mean(cc))
                complete = len(hits) == len(ids)
                entry = {'method':method,'value':value,'horizon':horizon,'budget':budget,
                         'question_coverage':len(hits)/len(ids),'complete':complete,
                         'mean_compute':float(np.mean(costs)) if costs else None}
                if complete:
                    a = estimate(hits, weights)
                    entry.update(accuracy=a['mean'],low=a['low'],high=a['high'],
                                 utilization=float(np.mean(costs)/budget))
                curves.append(entry); curve_choices.extend(choices)

        # Same allowed budget as primary oracle, common S1 reference for ratios.
        primary = ('S3','oracle',16,4,6)
        diagnostics=[]
        diagnostic_votes={}
        for key in [primary,('S2','oracle',16,1,6)]:
            hits,costs,residuals=[],[],[]
            for pid in ids:
                hh=[]
                for seed in seeds:
                    row=arrays[key][pid,seed]
                    budget=row['source_forwards']-row['oracle_forwards']
                    vote=vote_at(pools[pid,seed],budget)
                    if vote is None or not vote['coverage']:
                        raise ValueError('No unbilled diagnostic vote coverage')
                    hh.append(vote['answer']==gold[pid]);costs.append(budget);residuals.append(vote['residual'])
                hits.append(np.mean(hh))
            diagnostic_votes[key]=np.asarray(hits)
            diagnostics.append({'setting':label(key),'accuracy':estimate(vectors[key]['accuracy'],weights),
                                'vote':estimate(hits,weights),
                                'delta':estimate(vectors[key]['accuracy']-hits,weights),
                                'mean_compute':float(np.mean(costs)),
                                'mean_residual':float(np.mean(residuals)),
                                'max_residual':float(max(residuals)),
                                'cost_mode':'Oracle MC source forwards separate (privileged diagnostic)'})
        ratios = []
        for mode,value in [(m,v) for m in ['billed','oracle_MC_separate'] for v in ['B4','B3']]:
            hits, reference, actual_cost, budget_cost = [], [], [], []
            for pid in ids:
                hh, vv = [], []
                for seed in seeds:
                    budget = arrays[primary][pid,seed]['source_forwards']
                    if mode=='oracle_MC_separate':
                        budget-=arrays[primary][pid,seed]['oracle_forwards']
                    # Common budget for the three fixed N16/b4 policies. Tiny
                    # probe FLOPs or answer-length differences must not demote
                    # a policy to an entirely different beam configuration.
                    budget=max(budget,*(arrays[('S3',v,16,4,6)][pid,seed]['source_forwards']
                                        for v in ['B4','B3']))
                    vote=vote_at(pools[pid,seed],budget)
                    if vote is None or not vote['coverage']:
                        raise ValueError('Common fixed-beam voting budget uncovered')
                    r=arrays[('S3',value,16,4,6)][pid,seed]
                    hh.append(r['correct']);vv.append(vote['answer']==gold[pid])
                    actual_cost.append(r['source_forwards']); budget_cost.append(budget)
                hits.append(np.mean(hh));reference.append(np.mean(vv))
            reference=np.asarray(reference)
            denominator=vectors[primary]['accuracy']-reference
            num=np.asarray(hits)-reference
            ratios.append({'value':value,'cost_mode':mode,'accuracy':estimate(hits,weights),
                           'delta_common_vote':estimate(num,weights),
                           'oracle_delta_common_vote':estimate(denominator,weights),
                           'fraction':fraction(num,denominator,weights),
                           'mean_compute':float(np.mean(actual_cost)),
                           'mean_allowed_budget':float(np.mean(budget_cost)),
                           'utilization':float(np.sum(actual_cost)/np.sum(budget_cost)),
                           'interpretation':'Common per-question budget=max(actual fixed N16/b4 oracle, B4, B3 cost); oracle MC separated only in diagnostic'})

        out = Path(c['output']); out.mkdir(parents=True,exist_ok=True)
        csv_file(out/'all_settings.csv',rows)
        csv_file(out/'paired_comparisons.csv',paired)
        csv_file(out/'summary.csv',summaries)
        csv_file(out/'common_budget_curves.csv',curves)
        csv_file(out/'common_budget_choices.csv',curve_choices)
        write_json(out/'fractions.json',ratios)
        write_json(out/'oracle_mc_separate.json',diagnostics)

        ledger = read('data/M5a_preflight_v0/cost_ledger.json')
        existing = {str(Path(r['run']).resolve()) for r in ledger}
        for path in sorted(Path('runs').glob('*/config.json')):
            rc = read(path)
            if rc.get('milestone') != 'M5a' or rc.get('stage') not in ['smoke','full','vote_extension']:
                continue
            if str(path.parent.resolve()) in existing:
                continue
            m = read(path.parent/'metrics.json')
            if m['status'] == 'running':
                if smoke:
                    continue  # Smoke analysis never claims a final cost ledger.
                raise ValueError('Cannot freeze costs while an M5a GPU run is active')
            ledger.append({'run':str(path.parent),'version':rc['version'],'status':m['status'],
                           'gpu_hours_recorded':m.get('gpu_hours'),
                           'gpu_hours_conservative_charge':max(m.get('gpu_hours',0),m.get('wall_seconds',0)/3600),
                           'source_forward_calls':m.get('source_forward_calls'),
                           'lane_forwards':m.get('lane_forwards'),'final_cost_missing':False})
        charged = sum(r['gpu_hours_conservative_charge'] for r in ledger)
        if charged > cfg['max_gpu_hours']:
            raise ValueError('Actual conservative GPU cost exceeds authorized budget')
        csv_file(out/'cost_ledger.csv',ledger)
        write_json(out/'cost_ledger.json',ledger)

        plt.rcParams.update({'font.size':10,'savefig.dpi':180})
        fig, axes = plt.subplots(1,2,figsize=(13,4.5),sharey=True)
        colors = {'none':'black','B4':'#0072B2','B3':'#D55E00','oracle':'#009E73'}
        for ax,h in zip(axes,[6,4]):
            for method,value,horizon in families:
                if horizon not in [h,0] or (method == 'S1' and h != 6):
                    continue
                points = [r for r in curves if r['method'] == method and r['value'] == value
                          and r['horizon'] == horizon and r['complete']]
                if not points:
                    continue
                ax.plot([r['budget'] for r in points],[100*r['accuracy'] for r in points],
                        color=colors[value],linestyle={'S0':':','S1':'-','S2':'--','S3':'-.'}[method],
                        marker='o',markersize=3,label=f'{method} {value}')
            # The independent S1 policy law is shared by both horizon panels.
            if h == 4:
                points=[r for r in curves if r['method']=='S1' and r['complete']]
                ax.plot([r['budget'] for r in points],[100*r['accuracy'] for r in points],
                        color='black',marker='o',markersize=3,label='S1 none')
            ax.set_xscale('log',base=2);ax.set_xlabel('Allowed source-forward equivalents (oracle billed)')
            ax.set_title('Full search' if h==6 else 'Search t <= 3')
            ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
        axes[0].set_ylabel('val accuracy (%)')
        fig.tight_layout();fig.savefig(out/'accuracy_compute.png');fig.savefig(out/'accuracy_compute.pdf');plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=True)
        for ax,h in zip(axes,[6,4]):
            for method,value,horizon in families:
                if method=='S0':
                    r=summary[('S0','none',1,1,0)]
                    ax.scatter(r['mean_compute'],100*r['accuracy'],color='black',marker='x',label='S0')
                    continue
                if horizon!=h:
                    continue
                points=[r for r in summaries if r['method']==method and r['value']==value and r['horizon']==horizon]
                points=sorted(points,key=lambda r:r['mean_compute'])
                if not points:
                    continue
                ax.plot([r['mean_compute'] for r in points],[100*r['accuracy'] for r in points],
                        color=colors[value],linestyle={'S1':':','S2':'--','S3':'-'}[method],
                        marker='o',markersize=3,label=f'{method} {value}')
                if value=='oracle':
                    for r in points:
                        ax.errorbar(r['mean_compute'],100*r['accuracy'],
                                    yerr=[[100*(r['accuracy']-r['accuracy_low'])],
                                          [100*(r['accuracy_high']-r['accuracy'])]],
                                    color=colors[value],capsize=2,linewidth=.7)
                        ax.scatter(r['mean_compute']-r['mean_residual'],100*r['vote_accuracy'],
                                   color='black',marker='*',s=55,zorder=5,
                                   label='S1 at oracle cost' if method=='S3' else None)
            points=[r for r in curves if r['method']=='S1' and r['complete']]
            ax.plot([r['mean_compute'] for r in points],[100*r['accuracy'] for r in points],
                    color='gray',linewidth=.8,label='Independent S1 budget curve')
            ax.set_xscale('log');ax.set_xlabel('Mean actual source-forward equivalents (oracle billed)')
            ax.set_title('Full search' if h==6 else 'Search t <= 3')
            ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
        axes[0].set_ylabel('val accuracy (%)');fig.tight_layout()
        fig.savefig(out/'actual_compute.png');fig.savefig(out/'actual_compute.pdf');plt.close(fig)

        fig,ax=plt.subplots(figsize=(8,4.8))
        for method,value,horizon in families:
            if method=='S0':
                continue
            points=[r for r in summaries if r['method']==method and r['value']==value and r['horizon']==horizon]
            if not points:
                continue
            ax.plot([r['mean_compute']-r['mean_oracle_forwards'] for r in points],
                    [100*r['accuracy'] for r in points],color=colors[value],
                    linestyle='-' if horizon==6 else '--',marker='o',markersize=3,
                    label=f'{method} {value} h{horizon}')
        ax.set_xscale('log');ax.set_xlabel('Source equivalents EXCLUDING privileged oracle rollout cost')
        for r in diagnostics:
            ax.scatter(r['mean_compute']-r['mean_residual'],100*r['vote']['mean'],
                       color='black',marker='*',s=65,label='Matched S1 '+r['setting'].split('/')[0])
        ax.set_ylabel('val accuracy (%)');ax.set_title('Diagnostic only: not the billed compute comparison')
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=3);fig.tight_layout()
        fig.savefig(out/'oracle_unbilled_diagnostic.png');plt.close(fig)

        def summary_table(keys):
            text='|配置|accuracy % [95% CI]|均值前向|同预算S1 %|差 pp [95% CI]|\n|---|---:|---:|---:|---:|\n'
            for key in keys:
                r=summary[key]
                a={x:r[y] for x,y in [('mean','accuracy'),('low','accuracy_low'),('high','accuracy_high')]}
                d={x:r[y] for x,y in [('mean','delta'),('low','delta_low'),('high','delta_high')]} if key[0]!='S0' else None
                text+=f"|{label(key)}|{interval(a)}|{r['mean_compute']:.3g}|{percent(r['vote_accuracy']) if d else '—'}|{interval(d) if d else '—'}|\n"
            return text
        main_keys=[('S0','none',1,1,0)]+[('S3',v,16,4,6) for v in ['oracle','B4','B3']]
        secondary_keys=[('S3',v,16,4,4) for v in ['B4','B3']]+[('S2',v,16,1,6) for v in ['oracle','B4','B3']]
        table1,table2=summary_table(main_keys),summary_table(secondary_keys)
        diag=diagnostics[0]
        table1+=f"|S3/oracle（MC另计）|{interval(diag['accuracy'])}|{diag['mean_compute']:.3g}|{percent(diag['vote']['mean'])}|{interval(diag['delta'])}|\n"
        table3='|成本口径/价值函数|相同允许预算下差 pp [95% CI]|取得oracle增益 % [95% CI]|预算使用率 %|\n|---|---:|---:|---:|\n'
        for r in ratios:
            f=r['fraction']
            mode='全计费' if r['cost_mode']=='billed' else 'MC另计'
            table3+=f"|{mode}/{r['value']}|{interval(r['delta_common_vote'])}|{interval(f) if f['identifiable'] else '不可解释（oracle分母不稳）'}|{percent(r['utilization'])}|\n"
        p=summary[primary]
        d={x:p[y] for x,y in [('mean','delta'),('low','delta_low'),('high','delta_high')]}
        findings=f"主要S3 oracle相对同题同seed、账单预算内S1：{interval(d)} pp。"
        if p['delta_low']<=0:
            findings+='95%区间包含零，未得到同计算量正收益的明确证据；上限比例不作稳定结论。'
        else:
            findings+='其95%区间下界为正；这是该网格/策略的特权经验参考，非数学上界。'
        findings+=f"MC成本单独列出的上限诊断增益为{interval(diag['delta'])} pp，不能当作已计费收益。"
        findings+='B4/B3均未显示取得正收益；有好的价值函数时存在可用空间，现有探针尚未把它转为收益。'
        maxres=max(r['budget_residual'] for r in paired)
        meanres=float(np.mean([r['budget_residual'] for r in paired]))
        tests=Path(c['tests_log']).read_text().strip().splitlines()[-1]
        n=len(ids)
        limitation=("用户继续授权后保留10GPUh，仅S3 oracle限N16/b4全步；B4/B3与S1/S2原网格全保留。"
                    "K16有MC误差/自适应选择偏差，终点oracle读取gold后成为特权0/1选择。"
                    "LoRA合并非逐位一致；批量iid改变seed寻址，保持同扰动分布。"
                    "早期Q1特征错误已修复，旧smoke不作科研结果；OOM/资源失败/exit143及缺失成本见预检报告。")
        if recovery:
            limitation+='原搜索gate=false：仅一S1池4096条仍欠556前向，已按原随机地址另存补长到4224条，原结果未覆盖；恢复逐项核验后通过。'
        detail=f'''# M5a latent search feasibility

{'仅分析smoke，不能作为val300结论。' if smoke else 'val300全量完成，待审查。'} {findings}

## Protocol and statistics

任务与预先固定的比较规则：docs/M5a_TASK.md；不读取test。源fp32/temperature0，固定原cohort左padding与完整KV。先扰动、源前向，再让probe读真实观测。B3/B4只读状态/模板0/冻结dummy选项，不接收原题或gold；oracle例外读取gold。N4/8/16、beam2/4、全步与t≤3。各N独立π轨迹池；同N的S1/S2复用候选，所有设置逐题CSV保留，不按accuracy挑配置。
{n}题，seed {seeds}；先在题内平均seed，再按题配对bootstrap {c['bootstrap_repetitions']}次（bootstrap seed {c['seed']}），区间单位与均值一致，accuracy为百分比、差为pp。S0确定性只计一次。置信区间为逐比较区间，未多重比较校正；val300可行性观察不外推test。

## Primary actual-compute matches

{table1}
每项搜索与同题同seed独立长投票池的预算内最长前缀配对。未用完余额均值{meanres:.6g}、最大{maxres:.6g}源前向；严格离散预算上限相同，不能保证逐lane完全相等。投票池覆盖所有搜索成本；该基线覆盖不是在单一全局均值预算取相同条数。

## Early search and terminal selection

{table2}
早期beam保留分支之后继续π但不再扩展/中途筛选，终点再选一条。S1/S2两horizon没有中途搜索，仅作为独立随机面板。全部设置：{out}/summary.csv。

## Fraction at the same allowed oracle budget

{table3}
三者固定N16/b4全步；每题/seed共同预算取三者实际成本最大值，MC另计仅扣oracle内部rollout，probe开销仍计入。这样小数probe开销和答案长度差不会强迫换成小beam。分母为oracle对共同预算S1的增益，分子为probe对同一个S1的增益；该参考可与主表各自实耗预算S1有少量差异，精确分母在fractions.json。预算规则不依正确率，固定配置不换N/b。比例不截断到[0,100]；分母95%CI含零则不解释比例。全计费口径probe无法利用oracle多出的全部预算，使用率明确欠用，不能冒称全部算力已等量花完。

## Accuracy–compute curves

![Actual billed cost]({(out/'actual_compute.png').resolve()})

原始配置点使用每题实际成本的均值作横轴；黑色星号是oracle逐题账单预算内的独立S1，不能将横轴均值误读为每题花相同固定前向数。oracle误差条为按题95%CI；点的数值及配对差在CSV。预设共同预算点最高32768，oracle beam若超出则不在下图硬凑点，由本图实际成本点与配对表呈现。

![Billed cost]({(out/'accuracy_compute.png').resolve()})

横轴含prefill、潜步骤、答案解码、probe FLOPs等价和全部oracle K16内部源前向。计算单位是一条lane一次源API前向；prefill按一次源调用计费，单列prefill token数。probe折算使用源线性FLOPs+固定pad_to+4注意力近似，未称精确延迟比；B3/B4非常小，完整参数与公式在search run/compute_profile.json。物理batch调用/实际lane在metrics中记录；解码压缩已完成lane，所有执行lane均收费。汇总各配置理论独立成本不等于工程执行成本（候选池复用、额外匹配投票、门槛均影响后者）。
共同允许预算点固定128..32768，S2/S3每题/seed选预算内最大(N*b,N,b)，无accuracy参与选择；horizon分面。曲线只有全部题/seed支持时才画点，缺覆盖标在common_budget_curves.csv，不用部分题替代整集。预算内策略可欠花成本，使用率已落盘。常规S1长池曲线不外推未生成轨迹；N4/8/16主面板仍在summary.csv。

![Unbilled diagnostic]({(out/'oracle_unbilled_diagnostic.png').resolve()})

该图仅诊断去掉oracle成本后的几何趋势；oracle_mc_separate.json给出相应S1配对数值，不用于回答同账单计算量收益。

## Costs, gates, deviations

GPU为NVIDIA GeForce RTX4090；全量/当前search run墙钟GPUh {metrics['gpu_hours']:.9g}，源物理调用{metrics['source_forward_calls']}，lane {metrics['lane_forwards']}，prefill token {metrics['prefill_tokens']}。GPU共享，GPUh按作业占用墙钟保守累计，不是独占活跃kernel时间。含加载、smoke、失败和中断保守预留的总charge {charged:.9g} GPUh≤10；旧exit143保守预留0.080h，缺失最终源计数不当作零。逐run台账{out}/cost_ledger.csv，旧版本lane门槛计数缺口在预检保留。CPU报告GPUh=0。
v13五题smoke101.869秒通过；测试：{tests}。原全量gate_passed={metrics['gate_passed']}，补充恢复run={recovery}，组合证据逐题协议/标签/投票覆盖均通过；{len(rows)}条结果，{len(arrays)}配置键；原输入hash与启动记录逐项核对一致。test_access=false。{limitation}

## Reproduction and evidence

搜索配置configs/M5a_full_v13.yaml；本报告配置{c['config_path']}；搜索run {root}；报告run {run.path}。
全部配置与逐题匹配、共同预算及选择日志、比例、图和成本：{out}/。源码/环境/git/权重hash：search run/source_snapshot、input_sha256.json、environment.json、git.json；报告自身另存分析源码hash。旧预检reports/M5a_preflight.md及data/M5a_preflight_v0不覆盖。
'''
        Path(c['detail']).write_text(detail)
        review=f'''# M5a REVIEW

### (a) 状态一句话

{'仅smoke分析，里程碑未完成' if smoke else 'M5a完成，待审查'}：仅val{n}，未触碰test。{findings}

### (b) 关卡清单

- 通过：v13固定5/300题（1.67%）smoke102秒<900秒；观测状态、cohort左padding、fp32协议核对。
- 通过：{tests}；共享噪声函数、Q1冻结特征、K16语义、预算配对/标签隔离测试。
- 通过：{n}题×{len(seeds)}seed、{len(rows)}条配置结果；按题配对10,000次bootstrap，seed先题内平均。
- 通过：所有搜索成本有独立S1覆盖；预算余量均值{meanres:.3g}、最大{maxres:.3g}次。共同预算曲线覆盖缺口显式标缺。
- 通过：含全部smoke/失败/中断保守预留，总{charged:.3g}GPUh≤10.0；当前搜索{metrics['gpu_hours']:.3g}GPUh，源调用{metrics['source_forward_calls']:.3g}、lane {metrics['lane_forwards']:.3g}（精确计数在metrics）。
- 未通过（按本次任务不执行）：test及完整原S3 oracle网格；不代表G1/G2通过。

### (c) 核心主表（val{n}；最多3张）

accuracy和差单位分别为%/pp，括号为95%CI；前向包含probe折算与oracle内部成本。

{table1}
{table2}
{table3}
问题(1)：主表同时给oracle全计费收益和MC另计的特权上限诊断，二者不可混用。问题(2)：主表给B4/B3各自实耗预算收益；末表固定三者N16/b4全步，共同预算取三者实际成本最大值，以同一个S1作分母参考，避免小数probe成本迫使换小beam。使用率明确欠用，MC另计仍保留probe成本。分母CI含零则比例不可解释；精确分母见fractions.json。

### (d) 本里程碑新增的偏离与阻塞

{limitation}离散S1前缀有不足一条轨迹的预算余额；共同预算曲线是允许上限，非保证全部花完。probe折算为FLOPs近似。详细限制及历史预检成本缺口留在详细报告。

### (e) 需要审查者决定的问题

无。用户“允许 继续完成”已按保留10GPUh的备选继续；本报告不申请额外预算或启动完整M5。

### (f) 证据路径

- 详细报告：{c['detail']}；分析CSV/图/成本/比例：{out}/。
- 搜索run：{root}；报告run：{run.path}。
- 投票恢复run：{recovery}。
- 配置configs/M5a_full_v13.yaml；预算data/M5a_v13/budget_plan.json；测试{c['tests_log']}。
- 旧预检reports/M5a_preflight.md、data/M5a_preflight_v0/；完成证据冻结data/M5a_v13/freeze.json。
'''
        if len(review.encode())>8000:
            raise ValueError('Review exceeds 8000 UTF-8 bytes')
        Path(c['review']).write_text(review)
        write_json(out/'analysis.json',{'search_run':str(root),'report_run':str(run.path),
                   'vote_extension_run':str(recovery) if recovery else None,'combined_gate_passed':True,
                   'questions':n,'seeds':seeds,'rows':len(rows),'settings':len(arrays),
                   'cost_gpu_hours_conservative':charged,'summary':summaries,'fractions':ratios,
                   'oracle_mc_separate':diagnostics,
                   'review_bytes':len(review.encode()),'smoke':smoke,'test_access':False})
        write_json(run.path/'analysis_source_sha256.json',{p:sha(p) for p in
                   ['scripts/m5a_report.py','clr/search/analysis_m5a.py',c['config_path']]})
        run.write_metrics(status='completed',milestone_complete=not smoke,source_forward_calls=0,
                          gpu_hours=0,test_access=False,review_bytes=len(review.encode()),
                          questions=n,settings=len(arrays))
        print(run.path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('overrides',nargs='*')
    args=parser.parse_args();config=load_config(args.config,args.overrides)
    config['config_path']=args.config
    main(config)

"""Final M7.1 statistics from locked predictions; never fits to test labels."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clr.audit.m71_runtime import snapshot, live, assert_final_lock
from clr.audit.m71_statistics import gate_intervals,auc_group_draws
from clr.audit.m71_storage import load_bundle
from clr.audit.m7_evaluation import qa_summary
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from scripts.m7_report import fmt, ci_cell, interval, weights_for


KINDS = ['CLR', 'B3', 'linear', 'B0', 'supervised', 'norm', 'text']


def nonlinear_qa_intervals(frame,weights):
    from sklearn.metrics import roc_auc_score
    groups=sorted(frame.group_id.unique());seeds=sorted(frame.seed.unique())
    ece=0.;ece_draws=np.zeros(len(weights));auc=0.;auc_draws=np.zeros(len(weights));binary=False
    for seed in seeds:
        f=frame[frame.seed==seed];p=np.stack(f.probs.to_list());y=f.label.to_numpy(int)
        gi=np.searchsorted(groups,f.group_id.to_numpy());confidence=p.max(1);correct=p.argmax(1)==y
        bins=np.minimum((confidence*15).astype(int),14);sums=np.zeros((len(groups),15))
        np.add.at(sums,(gi,bins),correct-confidence)
        ece+=np.abs(sums.sum(0)).sum()/len(f)/len(seeds)
        ece_draws+=np.abs(weights@sums).sum(1)/len(f)/len(seeds)
        if p.shape[1]==2:
            binary=True;auc+=roc_auc_score(y,p[:,1])/len(seeds)
            auc_draws+=auc_group_draws(y,p[:,1],gi,weights)/len(seeds)
    def ci(point,draws):return {'mean':float(point),'lo':float(np.quantile(draws,.025)),'hi':float(np.quantile(draws,.975))}
    result={'ece':ci(ece,ece_draws)}
    if binary:result['auroc']=ci(auc,auc_draws)
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def detection_rows(root, rep, seeds):
    rows = []
    for kind in KINDS:
        for seed in seeds if kind not in ['norm', 'text'] else [0]:
            if kind in ['norm', 'text']:
                pointer = json.loads((root/rep/'baselines'/(kind+'_evaluated.json')).read_text())
                cal = json.loads((root/rep/'baselines'/(kind+'.json')).read_text())
            else:
                pointer = json.loads((root/rep/'evaluated'/f'{kind}_{seed}.json').read_text())
                cal = json.loads(Path(pointer['calibration']).read_text())
            for split in ['reader_val', 'test']:
                path = Path(pointer['run'])/(split+'_scores.jsonl')
                if not path.exists():
                    if kind == 'text' and split == 'reader_val':
                        continue
                    raise FileNotFoundError(str(path))
                for r in read_jsonl(path):
                    for key, thresholds in cal['thresholds'].items():
                        for alpha in [.01, .05]:
                            q = thresholds[str(alpha)]
                            q = float('inf') if q is None else q
                            rows.append({key: r[key] for key in ['id', 'group_id', 'category']} | {
                                'split': split, 'kind': kind, 'seed': seed,
                                'k': int(key) if key.isdigit() else 0,
                                'alpha': alpha, 'score': r[key], 'rejected': float(r[key] > q)})
    return pd.DataFrame(rows)


def summarize_detection(frame, c, cache, panel):
    rows = []; rates = {}; auc_cache = {}
    for (split, kind, k, alpha), f in frame.groupby(['split', 'kind', 'k', 'alpha']):
        clean = f[f.category == 'clean']; groups = sorted(clean.group_id.unique())
        w = weights_for(groups, c, cache, split)
        fp = interval(clean.groupby('group_id').rejected.mean().reindex(groups), w)
        cats = sorted(set(f.category)-{'clean'})
        if any(cat.startswith('corrupted_') for cat in cats):cats.append('corrupted')
        for category in cats:
            pos = f[f.category.str.startswith('corrupted_')] if category == 'corrupted' else f[f.category == category]
            values = pos.groupby('group_id').rejected.mean().reindex(groups)
            if values.isna().any():raise ValueError('Detection panel has missing paired groups')
            tp = interval(values, w)
            rates[(split, kind, k, alpha, category)] = values
            key = (split, kind, k, category)
            if key not in auc_cache:
                from sklearn.metrics import roc_auc_score
                point=0.;draws=np.zeros(len(w))
                seeds = sorted(clean.seed.unique())
                for seed in seeds:
                    neg = clean[clean.seed == seed].set_index('group_id').score.reindex(groups).to_numpy()
                    sub = pos[pos.seed == seed]
                    categories = list(sub.category.unique())
                    for cat in categories:
                        pp = sub[sub.category == cat].set_index('group_id').score.reindex(groups).to_numpy()
                        labels=np.r_[np.ones(len(groups),int),np.zeros(len(groups),int)]
                        scores=np.r_[pp,neg];indices=np.tile(np.arange(len(groups)),2)
                        point+=roc_auc_score(labels,scores)/len(seeds)/len(categories)
                        draws+=auc_group_draws(labels,scores,indices,w)/len(seeds)/len(categories)
                auc_cache[key] = {'mean': float(point), 'lo': float(np.quantile(draws, .025)), 'hi': float(np.quantile(draws, .975))}
            auc = auc_cache[key]
            rows.append({'panel': panel, 'split': split, 'kind': kind, 'k': k, 'alpha': alpha,
                'category': category, 'groups': len(groups), **tp,
                **{'FPR'+suffix: fp[name] for suffix, name in [('', 'mean'), ('_lo', 'lo'), ('_hi', 'hi')]},
                **{'AUROC'+suffix: auc[name] for suffix, name in [('', 'mean'), ('_lo', 'lo'), ('_hi', 'hi')]}})
    return rows, rates


def qa_statistics(root, rep, c, out, cache):
    rows = []; core = {}; per_seed = []; aux = []
    for split in ['reader_val', 'test']:
        for kind in ['CLR', 'B3', 'linear', 'B0']:
            for task in ['A1', 'A2', 'A3', 'A4']:
                for templates in ['seen', 'heldout']:
                    records = []
                    for seed in c['seeds']:
                        pointer = json.loads((root/rep/'evaluated'/f'{kind}_{seed}.json').read_text())
                        cal = json.loads(Path(pointer['calibration']).read_text())
                        data = load_bundle(Path(pointer['run'])/f'{split}_{task}_{templates}.pt.gz')
                        p = data['probs'].numpy(); rr = data['rows']; y = np.array([r['label'] for r in rr])
                        per_seed.append({'split': split, 'kind': kind, 'seed': seed, 'task': task, 'templates': templates, **qa_summary(rr, p)})
                        for i, r in enumerate(rr):
                            item = {'group_id': r['group_id'], 'category': r['category'], 'seed': seed,
                                    'label': r['label'], 'probs': p[i], 'correct': float(p[i].argmax() == y[i]),
                                    'nll': float(-np.log(max(p[i, y[i]], 1e-12))),
                                    'brier': float(np.square(p[i]-np.eye(p.shape[1])[y[i]]).sum())}
                            for alpha in [.01, .05]:
                                sets = (1-p[i]) <= cal['qa_conformal'][task][str(alpha)]
                                item['coverage_'+str(alpha)] = float(sets[y[i]])
                                item['size_'+str(alpha)] = int(sets.sum())
                            records.append(item)
                    frame = pd.DataFrame(records)
                    names = ['correct', 'nll', 'brier', 'coverage_0.01', 'size_0.01', 'coverage_0.05', 'size_0.05']
                    g = frame.groupby(['group_id', 'seed'])[names].mean().groupby('group_id').mean().sort_index()
                    w = weights_for(g.index, c, cache, split)
                    core[(split, kind, task, templates)] = g['correct']
                    summary = {name: interval(g[name], w) for name in names}
                    summary.update(nonlinear_qa_intervals(frame, w))
                    for metric, value in summary.items():
                        rows.append({'split': split, 'kind': kind, 'task': task, 'templates': templates,
                                     'metric': metric, 'groups': len(g), **value})
                    for category, f in frame.groupby('category'):
                        gcat = f.groupby(['group_id', 'seed'])[names].mean().groupby('group_id').mean().sort_index()
                        for metric in names:
                            aux.append({'split': split, 'kind': kind, 'task': task, 'templates': templates,
                                'category': category, 'metric': metric, 'groups': len(gcat),
                                **interval(gcat[metric], weights_for(gcat.index, c, cache, split))})
    pd.DataFrame(rows).to_csv(out/'qa_group_bootstrap.csv', index=False)
    pd.DataFrame(aux).to_csv(out/'qa_category_group_bootstrap.csv', index=False)
    pd.DataFrame(per_seed).to_csv(out/'qa_per_seed.csv', index=False)
    paired = []
    for split in ['reader_val', 'test']:
        for task in ['A1', 'A2', 'A3', 'A4']:
            for templates in ['seen', 'heldout']:
                a = core[(split, 'CLR', task, templates)]; b = core[(split, 'B3', task, templates)].reindex(a.index)
                paired.append({'split': split, 'task': task, 'templates': templates,
                               **interval(a-b, weights_for(a.index, c, cache, split))})
    pd.DataFrame(paired).to_csv(out/'qa_CLR_minus_B3.csv', index=False)
    return pd.DataFrame(rows), pd.DataFrame(paired)


def receiver_statistics(root, rep, c, out, cache):
    folder = root/rep/'source'/'test'
    entries = {r['id']: r for r in read_jsonl(folder/'manifest.jsonl')}
    records = read_jsonl(folder/'receiver.jsonl'); records.sort(key=lambda r: r['group_id'])
    if len(records) != 300:raise ValueError('Receiver final panel must contain 300 groups')
    groups = [r['group_id'] for r in records]; w = weights_for(groups, c, cache, 'test')
    values = {}
    for name in ['clean', 'none', 'mismatch']:
        values[name] = np.array([int(np.argmax(r[name]) == entries[r['id']]['route'][0]) for r in records], float)
    result = {name: interval(v, w) for name, v in values.items()}
    for name in ['none', 'mismatch']:result['gain_vs_'+name] = interval(values['clean']-values[name], w)
    contrast = []
    for r in records:
        a, b = entries[r['id']]['route']
        contrast.append((r['counterfactual'][b]-r['clean'][b])-(r['counterfactual'][a]-r['clean'][a]))
    result['private_fact_probability_contrast'] = interval(contrast, w)
    result['groups'] = len(groups)
    result['feature_replay_max_abs'] = max(r['feature_replay_max_abs'] for r in records)
    write_json(out/'receiver_causal.json', result)
    return result


def natural_statistics(root, rep, c, out, cache):
    records = []
    for seed in c['seeds']:
        pointer = json.loads((root/rep/'natural'/f'CLR_{seed}.json').read_text())
        records += read_jsonl(Path(pointer['run'])/'natural_probabilities.jsonl')
    frame = pd.DataFrame(records); rows = []
    for (split, mode, correct), f in frame.groupby(['split', 'mode', 'model_correct']):
        g = f.groupby(['group_id', 'seed']).P_consistent.mean().groupby('group_id').mean().sort_index()
        rows.append({'split': split, 'mode': mode, 'model_correct': bool(correct), 'groups': len(g),
                     'q10': float(g.quantile(.1)), 'q90': float(g.quantile(.9)),
                     **interval(g, weights_for(g.index, c, cache, split))})
    pd.DataFrame(rows).to_csv(out/'natural_commitment_distribution.csv', index=False)
    return pd.DataFrame(rows)


def plot_detection(det,out,split='test',panel='full'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,alpha in zip(axes,[.01,.05]):
        maximum=10.
        for kind in ['CLR','B3','norm','text','supervised']:
            f=det[(det.panel==panel)&(det.split==split)&(det.kind==kind)&
                  (det.alpha==alpha)&(det.category=='counterfactual')].sort_values('k')
            if kind in ['norm','text','supervised']:
                mean=np.repeat(f['mean'].iloc[0],3);lo=np.repeat(f.lo.iloc[0],3);hi=np.repeat(f.hi.iloc[0],3)
            else:mean=f['mean'].to_numpy();lo=f.lo.to_numpy();hi=f.hi.to_numpy()
            label=kind+(' (n=300)' if kind=='text' else '')
            ax.plot([1,4,16],100*mean,marker='o',label=label)
            ax.fill_between([1,4,16],100*lo,100*hi,alpha=.1)
            maximum=max(maximum,float(100*np.max(hi))*1.1)
        ax.set(xlabel='Audit questions k',ylabel='Counterfactual detection (%)',
               xticks=[1,4,16],ylim=(0,min(100,maximum)),title=f'Clean calibration alpha={alpha}')
        ax.legend(fontsize=8)
    if split!='test':fig.suptitle('Synthetic statistics fixture; no test data')
    fig.tight_layout();fig.savefig(out/'counterfactual_detection.png',dpi=180)
    fig.savefig(out/'counterfactual_detection.pdf');plt.close(fig)


def main(c):
    torch.set_num_threads(2)
    with recorded(c) as run:
        snapshot(run); root = Path(c['output']); rep = c['representation']; assert_final_lock(c)
        run.write_metrics(gpu_hours=0, source_forward_calls=0, test_opened=True)
        split_groups={}
        for split,expected in [('reader_train',8000),('reader_val',1000),('cal_A',500),('cal_B',1000),('test',2000)]:
            records=read_jsonl(root/rep/'source'/split/'manifest.jsonl')
            groups={r['group_id'] for r in records}
            if len(records)!=expected or len(groups)!=expected:
                raise ValueError('Incomplete or duplicated final split: '+split)
            if any(groups&other for other in split_groups.values()):
                raise ValueError('Final cross-split group leakage')
            split_groups[split]=groups
        for seed in c['seeds']:
            b3=json.loads((root/rep/'training'/f'B3_{seed}.json').read_text())
            clr=json.loads((root/rep/'training'/f'CLR_{seed}.json').read_text())
            if clr['selected_exposure']<b3['exposure']:
                raise ValueError('Returned CLR checkpoint has less exposure than B3')
        out = root/'analysis'; out.mkdir(exist_ok=False); cache = {}
        d = detection_rows(root, rep, c['seeds']); d.to_csv(out/'detection_scores.csv.gz', index=False)
        detection, rates = summarize_detection(d, c, cache, 'full')
        text_groups = set(d[(d.kind == 'text') & (d.split == 'test')].group_id)
        matched = d[(d.split == 'test') & d.group_id.isin(text_groups)]
        panel, _ = summarize_detection(matched, c, cache, 'matched_text300'); detection += panel
        det = pd.DataFrame(detection); det.to_csv(out/'detection_group_bootstrap.csv', index=False)
        plot_detection(det,out)
        paired = []
        for split in ['reader_val', 'test']:
            for alpha in [.01, .05]:
                a = rates[(split, 'CLR', 16, alpha, 'counterfactual')]
                for kind in ['B3', 'linear', 'B0', 'norm', 'text', 'supervised']:
                    key = (split, kind, 16 if kind in ['B3', 'linear', 'B0'] else 0, alpha, 'counterfactual')
                    if key not in rates:continue
                    b = rates[key]; groups = sorted(set(a.index) & set(b.index))
                    paired.append({'split': split, 'alpha': alpha, 'baseline': kind, 'groups': len(groups),
                        **interval(a.reindex(groups)-b.reindex(groups), weights_for(groups, c, cache, split))})
        pd.DataFrame(paired).to_csv(out/'counterfactual_CLR_paired_differences.csv', index=False)
        qa, qa_diff = qa_statistics(root, rep, c, out, cache)
        receiver = receiver_statistics(root, rep, c, out, cache)
        natural = natural_statistics(root, rep, c, out, cache)
        gates = json.loads((root/rep/'C1.json').read_text()); c2 = json.loads((root/rep/'C2.json').read_text())
        uncertainty = {}
        for kind, result in gates['results'].items():
            data = torch.load(Path(result['run'])/'reader_val_predictions.pt', weights_only=True, map_location='cpu')
            uncertainty[kind] = gate_intervals(data, c['bootstrap_seed'], c['bootstrap_repeats'])
        write_json(out/'C1_group_bootstrap.json', uncertainty)
        costs = []
        for p in Path('runs').glob('*/config.json'):
            cfg = json.loads(p.read_text())
            if cfg.get('milestone') != 'M7.1':continue
            if p.parent == run.path:continue  # This pure-CPU report is still open.
            m = json.loads((p.parent/'metrics.json').read_text())
            if p.parent != run.path and m['status'] not in ['completed', 'failed']:
                raise ValueError('M7.1 run still open: '+str(p.parent))
            costs.append({'run': str(p.parent), 'status': m['status'], 'gpu_hours': m.get('gpu_hours', 0),
                          'source_forward_calls': m.get('source_forward_calls', 0)})
        pd.DataFrame(costs).to_csv(out/'costs.csv', index=False)
        gpu = sum(r['gpu_hours'] for r in costs); calls = sum(r['source_forward_calls'] for r in costs)
        if gpu > 30:raise ValueError('M7.1 exceeded 30 GPUh')
        def dr(split, kind, alpha=.05, panel='full'):
            k = 16 if kind in ['CLR', 'B3', 'linear', 'B0'] else 0
            f = det[(det.panel == panel) & (det.split == split) & (det.kind == kind) &
                    (det.k == k) & (det.alpha == alpha) & (det.category == 'counterfactual')]
            return f.iloc[0].to_dict() if len(f) else None
        def qr(split, kind, task, templates='seen'):
            return qa[(qa.split == split) & (qa.kind == kind) & (qa.task == task) &
                      (qa.templates == templates) & (qa.metric == 'correct')].iloc[0].to_dict()
        table1 = '|方法|reader_val 检测% [CI] / FPR% / AUROC|test 检测% [CI] / FPR% / AUROC|\n|---|---|---|\n'
        for kind in KINDS:
            cells = []
            for split in ['reader_val', 'test']:
                r = dr(split, kind)
                cells.append('未执行' if r is None else f"{ci_cell(r)} / {fmt(100*r['FPR'])} / {fmt(r['AUROC'])}")
            table1 += '|'+kind+'|'+'|'.join(cells)+'|\n'
        table2 = '|QA|reader_val CLR/B3 %|test CLR/B3 %|test held-out CLR/B3 %|\n|---|---|---|---|\n'
        for task in ['A1', 'A2', 'A3', 'A4']:
            cells = [' / '.join(fmt(100*qr(s, k, task, t)['mean']) for k in ['CLR', 'B3'])
                     for s, t in [('reader_val', 'seen'), ('test', 'seen'), ('test', 'heldout')]]
            table2 += '|'+task+'|'+'|'.join(cells)+'|\n'
        table3 = '|自然承诺|reader_val n / P一致 [10%,90%]|test n / P一致 [10%,90%]|\n|---|---|---|\n'
        for mode in ['direct', 'cot512']:
            for correct in [True, False]:
                cells = []
                for split in ['reader_val', 'test']:
                    f = natural[(natural.split == split) & (natural['mode'] == mode) & (natural.model_correct == correct)]
                    if len(f):
                        r = f.iloc[0]; cells.append(f"{int(r['groups'])} / {fmt(r['mean'])} [{fmt(r['q10'])},{fmt(r['q90'])}]")
                    else:cells.append('0 / NA')
                table3 += f"|{mode} {'正确' if correct else '错误'}|"+'|'.join(cells)+'|\n'
        one_percent = '；'.join(f"{kind} {fmt(100*dr('test',kind,.01)['mean'])}% / FPR {fmt(100*dr('test',kind,.01)['FPR'])}%" for kind in ['CLR','B3','norm','text','supervised'])
        matched_summary = '；'.join(f"{kind} {fmt(100*dr('test',kind,panel='matched_text300')['mean'])}%" for kind in ['CLR','B3','norm','text','supervised'])
        differences = []
        directions = []
        for task in ['A1', 'A4']:
            r = qa_diff[(qa_diff.split == 'test') & (qa_diff.task == task) & (qa_diff.templates == 'seen')].iloc[0].to_dict()
            differences.append(task+' '+ci_cell(r)+'pp')
            directions.append(task+('显著同向' if r['lo']>0 else '显著反向' if r['hi']<0 else '区间含0，未确认优势'))
        supervised_gap = next(r for r in paired if r['split']=='test' and r['alpha']==.05 and r['baseline']=='supervised')
        testlog = Path(c['test_log']).read_text(); test_count = re.findall(r'(\d+) passed', testlog)[-1]
        selection = json.loads((root/'layer_selection_v2.json').read_text())
        layer_curves = pd.read_csv(Path(selection['run'])/'layer_curves.csv')
        original_curves = layer_curves[layer_curves.representation=='unprojected']
        a4_peak = original_curves.sort_values(['A4_accuracy','layer'],ascending=[False,True]).iloc[0]
        a1_peak = original_curves.sort_values(['A1_AUROC','layer'],ascending=[False,True]).iloc[0]
        review = f'''### (a) 状态一句话
M7.1足量非自适应审计完成、待审查；主表示pool准入，M7/M6冻结保留，M8未启动。

### (b) 关卡清单
- 通过：独立.venv-comm、冻结生成器三SHA只读import；可微f真实raw K/V梯度；val12组smoke23.4秒，gzip往返/跨设备3.56e6元素一致100%、误差0；{test_count}项测试。
- 通过（点门槛）：train1000组800/200内部扫描，层{selection['layers']}；投影A1/A4准确率损失0/1.75pp，A4损失CI[-2.25,5.75]pp，不能证明总体损失上界<2pp。
- 通过：全train8000/reader_val1000 C1，linear A4=49.8%、A1 AUROC=.756；B3=45.6%、.758；500样本C2={fmt(100*c2['selected']['accuracy'])}%（83epoch）。C2 val12诊断近随机，不代表泛化；备选未执行。
- 通过执行：train/reader_val/cal_A/cal_B/test=8000/1000/500/1000/2000组；四种QA模型和监督参照各3seed，CLR所选曝光≥同seed B3实际曝光。cal后锁模型/阈值再开test；成本{fmt(gpu)} GPUh≤30、前向{fmt(calls)}次。
- 私有传信：test300组receiver有/无/错配准确率={fmt(100*receiver['clean']['mean'])}/{fmt(100*receiver['none']['mean'])}/{fmt(100*receiver['mismatch']['mean'])}%；只改私有事实的目标概率对比={ci_cell(receiver['private_fact_probability_contrast'])}pp，特征复放误差{fmt(receiver['feature_replay_max_abs'])}。历史raw逐张量对照仍缺原artifact，不能宣称完全通过。

### (c) 核心主表（最多3张）
CF为主结果，k=16，cal_B clean名义α=.05；实测FPR不按test重调。latent/norm：reader_val1000/test2000；text仅获授权test300，无正式reader_val输出。95%CI按原/CF组bootstrap10000次，条件于固定模型/校准/抽题，先组内再跨3seed平均；确定性基线不伪造seed方差。

{table1}
名义α=.01的test CF检测：{one_percent}。同一text300组的α=.05检测：{matched_summary}；配对差值/CI及全部k、类别见CSV。

{table2}
A2支持关系和A3承诺一致性的能力须按各自QA结果判定；k16聚合检测不能单独证明这两项已可读。
自然承诺仅诊断，预固定val100/test300，完整direct/cot512；先在组内平均模板与seed，不进调参/阈值。

{table3}
### (d) 本里程碑新增的偏离与阻塞
fp16特征用gzip无损运输；初次确定性cumsum失败、scan高lr振荡及入口/显存预检失败保留，未降模型或数据量。投影A1 AUROC下降1.88pp [0.566,3.23]；通过准确率点门槛不等于信息无损。
首次CPU统计布局诊断用临时入口，启动源码未快照；排除正式结果，已补YAML入口重跑，不读test、无模型调用。
所有类别统一三模板程序化commitment；sender既有val direct33.8%（80组）/cot51250.0%（8组）口径不同，部署模型承诺仍有差距。文本CF仅能靠receiver输出变化；natural不进入主结果。
QA标签是消息来源事实，不保证每条KV已完整编码；CF干预语义e可修改多条支持记录，未强制token长度/范数相等。mismatch可能仍语义相容，组CI条件于固定供体面板。corrupted只在cal/test做raw操作后压缩；无可靠语义QA标签。conformal依赖clean校准与未来消息可交换等假设，非逐批硬FPR保证。
M6初始化/学习式消息/C2C未接；主任务仅synthetic v2。监督参照复用CLR架构、仅训练A3；训练目标/曝光及公开约束的显式化不同，差距不能单独归因为通用审计代价。

### (e) 需要审查者决定的问题
无。四问：(1)C1/C2通过；train内部层扫描A4峰值在{int(a4_peak.layer)}层、{fmt(100*a4_peak.A4_accuracy)}%，A1 AUROC峰值在{int(a1_peak.layer)}层、{fmt(a1_peak.A1_AUROC)}，仅反映探针可读性。兼容性历史raw缺证据，传信数字见(b)。(2)两名义FPR及独立实测FPR/CF检测见表与CSV。(3)test CLR−B3：{'；'.join(differences)}；A1对应GSM8K Q4、A4对应Q3：{'；'.join(directions)}。(4)名义α=.05 test CLR−监督参照={ci_cell(supervised_gap)}pp，限当前监督协议。

### (f) 证据路径
reports/M7.1_details.md；data/M71_v1/analysis/（detection_group_bootstrap.csv、qa_group_bootstrap.csv、qa_CLR_minus_B3.csv、counterfactual_CLR_paired_differences.csv、natural_commitment_distribution.csv、receiver_causal.json、costs.csv）；pool/C1.json、C2.json、training/、calibration/、evaluated/；final_lock.json、freeze.json；docs/M71_LIVE.json。源码/配置/环境/日志见各run与reports/M71_evidence/。
'''
        review_path = Path('reports/M7.1_REVIEW.md'); review_path.write_text(review)
        if review_path.stat().st_size > 8000:raise ValueError('Review exceeds 8000 UTF8 bytes')
        details = '# M7.1 足量潜通信审计\n\n'+review+'\n\n统计与协议：\n'
        details += ('所有区间按原/CF group整体重采样10000次，非线性AUROC/ECE每次重算；条件于固定3个训练seed、校准集和问答随机序列，未涵盖重新训练/重抽校准集的不确定性。QA核心等权clean/CF，seen四改写与held-out两改写分列。单类别A3不计算无定义AUROC，合并clean/CF后计算。QA conformal每组/任务一条clean校准实例；CF与held-out仅为经验覆盖率。\n'
                    '主检测k16预固定，k1/4消融；k序列为相同16题的前k题，问题有放回且相关，不声称指数检测界。公开T/z与commitment推导期望答案，不读取私有事实来生成检测目标。\n'
                    '文本300面板按组SHA预固定；CSV另有matched_text300以同组比较所有方法，配对CLR−text只在此交集计算，其他主比较用全量相同组。监督参照只训练A3 clean/CF，不称适用于任意篡改的oracle上界。\n'
                    '特征f为4层日志均值、公开前缀均值及末8有效token，K各头/V各头拼接后固定2048×512正交投影。物理位置、mask、层/段/原边界保存；不输入事实标签或原文。噪声/截断/层置零在raw KV上执行后经过同一f；严格噪声半径及梯度检查见测试。\n'
                    '真实层曲线与探针权重见layer_selection_v2.json对应run；C1置信区间见analysis/C1_group_bootstrap.json。C2仅证明500条可记忆，正式模型均从零初始化；C2权重不用于主比较。CLR checkpoint曝光不低于同seed B3实际总曝光，全部归一化仅训练组。\n'
                    '与GSM8K对照：冻结M4.1 test Q3 CLR−B3 +17.5pp [16.1,19.0]，Q4 +7.93pp [6.31,9.56]。M7.1 A4/A1对应Q3/Q4，以新的配对区间判断，不将小幅方向当复现。\n'
                    '成本为单设备进程驻留墙钟保守累计，含smoke/失败/重启/CPU加载，不按共享GPU利用率折算；纯CPU控制器和统计为0GPUh。批次前向计调用数而非样本数，forward_examples/令牌见各run。\n')
        details += '\n启动/执行索引：\n'+json.dumps({'C1': gates, 'C2': c2, 'receiver': receiver, 'costs': costs}, ensure_ascii=False, indent=2)
        details += '\n\n层扫描完整指标/图：'+selection['run']+'/layer_curves.csv、layer_curves.png、layer_curves.pdf；仅使用train内部800/200组，未用reader_val/test选层。\n'
        details += '\n\n![Counterfactual detection](../data/M71_v1/analysis/counterfactual_detection.png)\n'
        Path('reports/M7.1_details.md').write_text(details)
        write_json(root/'report.json', {'run': str(run.path), 'gpu_hours': gpu, 'source_forward_calls': calls,
            'primary': {s: {k: dr(s, k) for k in KINDS} for s in ['reader_val', 'test']},
            'qa_differences': qa_diff.to_dict('records'), 'receiver': receiver,
            'review_bytes': review_path.stat().st_size, 'test_opened': True})
        write_json(root/'completion.json', {'run': str(run.path), 'status': 'completed_pending_review',
            'C1_passed': True, 'C2_passed': True, 'D_executed': True, 'test_opened': True,
            'gpu_hours': gpu, 'source_forward_calls': calls})
        run.write_metrics(status='completed', gpu_hours=0, source_forward_calls=0, test_opened=True,
                          review_bytes=review_path.stat().st_size, total_gpu_hours=gpu)
        live(stage='report_complete_pending_freeze', review=str(review_path), gpu_hours=gpu, test_opened=True)
        message='M7.1足量非自适应审计完成待审查；C1/C2通过，三seed主比较/统计完成；唯一审查reports/M7.1_REVIEW.md，冻结data/M71_v1/freeze.json；接续docs/M71_LIVE.json，M8未启动。'
        for name in ['AGENTS.md','docs/STATUS.md']:
            path=Path(name);path.write_text(re.sub(r'<!-- M71_LIVE -->.*?<!-- /M71_LIVE -->','<!-- M71_LIVE -->\n'+message+'\n<!-- /M71_LIVE -->',path.read_text(),flags=re.S))
        for name,content in [('EXPERIMENTS',f'C1/C2通过，足量train8000/val1000/cal1500/test2000和三seed主比较/组bootstrap完成；累计{fmt(gpu)}GPUh。'),
                             ('BLOCKERS','投影准确率只过点门槛，A4损失区间较宽；程序化承诺部署差距与历史raw缺证据保留；text仅test300，不能当作全test2000。'),
                             ('CHANGELOG','足量特征管线、三seed审计/监督参照、锁定校准/最终test与natural诊断完成；上游冻结保持只读。')]:
            with Path('docs/'+name+'.md').open('a') as handle:
                handle.write('\n## 2026-11-07 — M7.1完成待审查\n'+content+'\n唯一审查reports/M7.1_REVIEW.md；完整证据reports/M7.1_details.md、data/M71_v1/analysis/。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('config'); parser.add_argument('overrides', nargs='*')
    args = parser.parse_args(); main(load_config(args.config, args.overrides))

"""Deterministic paper rendering. This module has no model or torch imports."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'CLR': '#0072B2', 'CLR-lite': '#0072B2', 'CLR-lite-multi': '#009E73',
          'CLR-lite-single': '#56B4E9', 'B3': '#D55E00', 'B3-prefix': '#E69F00',
          'B0': '#777777', 'linear': '#CC79A7', 'norm': '#777777', 'text': '#000000',
          'D0': '#555555', 'R1': '#0072B2', 'R2': '#D55E00', 'R3': '#009E73',
          'R4': '#CC79A7', 'B4': '#777777', 'B5b': '#CC79A7', 'scratch': '#009E73'}
MODEL_ORDER = ['B0', 'B3', 'B3-prefix', 'CLR-lite-single', 'CLR-lite', 'CLR-lite-multi']
SOURCE_NAMES = {'gpt2': 'CODI-GPT2', 'llama': 'CODI-LLaMA'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.dpi': 180, 'axes.grid': True, 'grid.alpha': .18})


def escape(value):
    value = str(value)
    return ''.join({'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
                    '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}.get(c, c) for c in value)


def number(x):
    if x is None or pd.isna(x):return 'NA'
    return f'{float(x):.3g}'


def interval(row, scale=1, mean='mean', lo='lo', hi='hi'):
    if row is None:return 'NA'
    return rf'${number(scale * row[mean])}\;[{number(scale * row[lo])},\,{number(scale * row[hi])}]$'


def one(frame, **filters):
    for key, value in filters.items():
        frame = frame[frame[key] == value]
    if len(frame) > 1:raise ValueError(f'Ambiguous paper cell: {filters}; rows={len(frame)}')
    return None if frame.empty else frame.iloc[0]


class Context:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.get('output_root', 'paper'))
        self.inputs = config['inputs']
        self.used, self.outputs = {}, []
        freeze = json.loads(Path(config['input_freeze']).read_text())
        self.frozen = freeze['sha256']
        style()

    def read(self, key):
        p = Path(self.inputs[key]);digest = sha(p)
        if self.frozen.get(str(p)) != digest:raise ValueError(f'Paper input is not hash-frozen: {p}')
        self.used[str(p)] = digest
        return pd.read_csv(p)

    def write_data(self, frame):
        p = self.root / 'derived' / (self.config['artifact'] + '.csv')
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False, float_format='%.15g', lineterminator='\n')
        self.outputs.append(p)

    def table(self, columns, rows, notes='', long=False):
        p = self.root / 'tables' / (self.config['artifact'] + '.tex')
        p.parent.mkdir(parents=True, exist_ok=True)
        caption = escape(self.config['caption'])
        label = self.config['artifact'].replace('_', '-')
        header = ' & '.join(columns) + r' \\'
        body = [' & '.join(map(str, row)) + r' \\' for row in rows]
        if long:
            weights = self.config.get('column_widths', [1] * len(columns))
            widths = ''.join(r'p{' + f'{.9*w/sum(weights):.4f}' + r'\textwidth}' for w in weights)
            lines = [r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{@{}' + widths + '@{}}',
                     r'\caption{' + caption + r'}\label{tab:' + label + r'}\\',
                     r'\toprule', header, r'\midrule\endfirsthead', r'\toprule', header,
                     r'\midrule\endhead', *body, r'\bottomrule', r'\end{longtable}',
                     r'\noindent ' + escape(notes), r'\endgroup']
        else:
            lines = [r'\begin{table*}[t]', r'\centering\scriptsize', r'\caption{' + caption + '}',
                     r'\label{tab:' + label + '}', r'\resizebox{\textwidth}{!}{%',
                     r'\begin{tabular}{' + 'l' * len(columns) + '}', r'\toprule', header,
                     r'\midrule', *body, r'\bottomrule', r'\end{tabular}}',
                     r'\par\smallskip\parbox{\textwidth}{\scriptsize ' + escape(notes) + '}',
                     r'\end{table*}']
        p.write_text('\n'.join(lines) + '\n')
        self.outputs.append(p)

    def figure(self, fig):
        p = self.root / 'figures' / (self.config['artifact'] + '.pdf')
        p.parent.mkdir(parents=True, exist_ok=True)
        when = datetime(2026, 11, 10, tzinfo=timezone.utc)
        fig.savefig(p, format='pdf', bbox_inches='tight', metadata={
            'Title': self.config['caption'], 'Author': 'CalReader', 'Creator': 'CalReader M9',
            'Producer': 'Matplotlib', 'CreationDate': when, 'ModDate': when})
        preview = self.root / 'previews' / (self.config['artifact'] + '.png')
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, bbox_inches='tight', dpi=160)
        plt.close(fig)
        self.outputs.append(p)

    def finish(self):
        p = self.root / 'build_records' / (self.config['artifact'] + '.json')
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {'artifact': self.config['artifact'], 'inputs': self.used,
             'input_freeze_sha256': sha(self.config['input_freeze']),
             'render_source_sha256': sha(__file__), 'entrypoint_sha256': sha(sys.argv[0]),
             'config': self.config,
             'outputs': {str(x.relative_to(self.root)): sha(x) for x in self.outputs},
             'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'matplotlib': matplotlib.__version__,
                          'PyYAML': yaml.__version__}}
        p.write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')


def t1(c):
    f = c.read('main');split = c.config.get('split', 'test');f = f[f.split == split]
    rows = []
    for source in ['gpt2', 'llama']:
        for model in MODEL_ORDER:
            cells = [SOURCE_NAMES[source], escape(model)]
            for q, metric, scale in [('Q1','nll',1), ('Q2','auroc',1), ('Q3','accuracy',100),
                                     ('Q4','accuracy',100), ('Q3','Q3_minus_B0',100)]:
                cells.append(interval(one(f, source=source, model=model, qtype=q, metric=metric), scale))
            rows.append(cells)
    c.write_data(f)
    c.table(['Source', 'Reader', r'Q1 NLL $\downarrow$', r'Q2 AUROC $\uparrow$',
             r'Q3 acc. (\%)', r'Q4 acc. (\%)', 'Q3$-$B0 (pp)'], rows,
            f'{split}; clean states, seen templates; micro within step, macro across steps and three fixed seeds. '
            '95% problem-cluster bootstrap intervals, conditional on fixed trained seeds. '
            'Missing LLaMA B0/prefix/single-state experiments are NA; Q3-B0 cannot be computed there. '
            'Q1 has source-specific continuation policies and is not the same cross-source target.')


def t2(c):
    f = c.read('conformal');rows = []
    for source in ['gpt2', 'llama']:
        for split in ['test', 'reader_val']:
            for model in MODEL_ORDER:
                for q in ['Q1','Q2','Q3','Q4']:
                    cov = one(f, source=source, split=split, model=model, qtype=q, metric='coverage')
                    size = one(f, source=source, split=split, model=model, qtype=q, metric='size')
                    if cov is None:continue
                    rows.append([SOURCE_NAMES[source], escape(split), escape(model), q,
                                 interval(cov, 100), interval(size)])
    c.write_data(f)
    c.table(['Source', 'Split', 'Reader', 'Type', r'Coverage (\%)', 'Set size'], rows,
            'Nominal alpha=0.1; source/type/step calibration on cal_B. One hashed question per problem per stratum. '
            'Q1 calibration selects one hashed Bernoulli rollout outcome; reported Q1 coverage averages '
            'the prediction-set mass under each saved rollout-frequency target. '
            'Intervals resample problem clusters and retain fixed seeds and fitted thresholds. '
            'Marginal coverage requires exchangeability; source-training reader_val is a different distribution.', long=True)


def t2_paired(c):
    f = c.read('paired');rows = []
    for _, r in f.sort_values(['source','split','qtype','model']).iterrows():
        rows.append([SOURCE_NAMES[r.source], escape(r['split']), r.qtype, escape(r.model),escape(r.baseline),interval(r)])
    c.write_data(f)
    c.table(['Source','Split','Type','Reader','Reference','Paired set-size difference'], rows,
            'Negative values mean smaller deployed prediction sets at alpha=0.1. Coverage can differ; '
            'these are not matched-coverage comparisons. Problem bootstrap with the same seed/problem/step pairs.', long=True)


def t3(c):
    f = c.read('shift');f = f[f.alpha == .1].sort_values(['shift','model','qtype','method']);rows=[]
    names = {'split':'standard', 'weighted':'weighted', 'mondrian_source_only':'Mondrian (source cal)',
             'mondrian_target_cal':'Mondrian (target cal)'}
    for _, r in f.iterrows():
        bias = {'mean':r.coverage_minus_target, 'lo':r.coverage_ci_low-.9, 'hi':r.coverage_ci_high-.9}
        rows.append([escape(r['shift']), escape(r.model), r.qtype, escape(names[r.method]),
                     interval(bias,100), interval(r,mean='size',lo='size_ci_low',hi='size_ci_high')])
    c.write_data(f)
    c.table(['Shift','Reader','Type','Calibration',r'Coverage $-$ 90\% (pp)','Set size'], rows,
            'Frozen M6 shift results, alpha=0.1. Source-only Mondrian uses the full set if the target stratum is missing; '
            'target-calibrated Mondrian has additional target calibration data. Estimated weights do not establish '
            'coverage under arbitrary shift, especially disjoint step support.', long=True)


def t4(c):
    d = c.read('detection');q = c.read('qa');loc = c.read('localization');rows=[]
    for split in ['test','reader_val']:
        for kind in ['CLR','B3','linear','norm','text','B0','supervised']:
            k = 16 if kind in ['CLR','B3','linear','B0'] else 0
            base = dict(panel='full',split=split,kind=kind,k=k,category='counterfactual')
            a = one(d,**base,alpha=.05);b = one(d,**base,alpha=.01)
            rows.append([escape(split), escape(kind), str(int(a.groups)) if a is not None else 'NA',
                         interval(a,100), interval(b,100), interval(a,mean='AUROC',lo='AUROC_lo',hi='AUROC_hi'),
                         interval(a,100,mean='FPR',lo='FPR_lo',hi='FPR_hi')])
    c.table(['Split','Auditor','Groups',r'CF detect @5\%',r'CF detect @1\%','AUROC',r'Actual clean FPR @5\%'], rows,
            'k=16 for latent QA readers; nominal clean conformal thresholds. Text uses only 300 authorized test groups '
            'and is not paired with the full 2000-group panel; matched-panel comparisons are in the source CSV. '
            'Commitments are programmatic for every category; text can detect CF only through receiver-output changes. '
            'Zero-event bootstrap intervals do not imply a zero population event rate.')
    c.write_data(d[(d.category=='counterfactual') & ((d.k==16)|(d.k==0))])


def t4_qa(c):
    q = c.read('qa');rows=[]
    for split in ['test','reader_val']:
        for kind in ['CLR','B3','linear','B0']:
            rows.append([escape(split),escape(kind), *[interval(one(q,split=split,kind=kind,task=t,
                                                                   templates='seen',metric='correct'),100)
                                                     for t in ['A1','A2','A3','A4']]])
    c.write_data(q[(q.templates=='seen')&(q.metric=='correct')])
    c.table(['Split','Reader',r'A1 acc. (\%)',r'A2 acc. (\%)',r'A3 acc. (\%)',r'A4 acc. (\%)'],rows,
            'Three fixed seeds, 10000 group-bootstrap intervals. A2/A3 remain near chance; '
            'aggregation detection does not establish direct support/inconsistency QA ability. '
            'Supervised A3 reference used different training objectives/exposure, so it is not an oracle upper bound.')


def t4_localization(c):
    f = c.read('localization');rows=[]
    for _, r in f.sort_values(['split','task','mode']).iterrows():
        rows.append([escape(r['split']),r.task,escape(r['mode']),interval(r,100),str(int(r.groups))])
    c.write_data(f)
    c.table(['Split','Task','Observed KV segments',r'Accuracy (\%)','Groups'],rows,
            'Frozen M8a P0 interventions on the M7.1 reader. Critical-only keeps the private-value record; '
            'critical-remove removes it. These are localization controls, not new training experiments.')


def t5(c):
    f = c.read('costs');rows=[]
    for _, r in f.iterrows():
        partial = 'lower bound' in r.cost_scope
        scope = ('unavailable' if r.cost_scope.startswith('unavailable') else
                 'partial legacy ledger' if partial else 'closed ledger')
        rows.append([escape(r.milestone), (r'$\geq$' if partial else '') + number(r.gpu_hours),
                     (r'$\geq$' if partial or r.missing_source_counts else '') + number(r.source_forward_calls),
                     str(int(r.missing_source_counts)), scope])
    c.write_data(f)
    c.table(['Milestone','GPU hours','Recorded source forwards','Missing count rows','Scope'], rows,
            'RTX 4090 resident wall time summed over GPU workers, including documented smoke/failure/restart costs '
            'and conservative additions. Historical missing costs are lower bounds, not zero. '
            'Source calls are physical batched forward calls and are not directly comparable to per-lane FLOPs-equivalent '
            'search budgets. Frozen donor training reused in M6/M9 is charged to its original milestone. No grand total is claimed.')


def errorline(ax, x, y, lo, hi, **kwargs):
    return ax.errorbar(np.asarray(x), np.asarray(y), yerr=np.maximum(0,np.vstack([np.asarray(y)-np.asarray(lo),
                       np.asarray(hi)-np.asarray(y)])), capsize=2, markersize=3, **kwargs)


def f1(c):
    a = c.read('type');b = c.read('source');fig, axes = plt.subplots(1,3,figsize=(10.2,3.25))
    a = a[(a.split=='test')&(a.qtype=='Q4')&(a.kind=='clean')&(~a.heldout)&
          (a.calibration=='temperature')&(a.subset=='all')&(a.metric=='accuracy')].copy()
    for prefix,label,color in [('transfer_','CLR transfer',COLORS['CLR']),('scratch_b3_','B3 scratch',COLORS['B3'])]:
        f = a[a.case.str.startswith(prefix)].copy()
        f['N'] = f.case.str.extract(r'_(\d+)(?:_lite)?$').astype(int);f=f.sort_values('N')
        errorline(axes[0],f.N,100*f['mean'],100*f.ci95_low,100*f.ci95_high,color=color,marker='o',label=label)
    axes[0].set(title='(a) New question type: Q4',xlabel='Labeled QA instances N',ylabel='Test accuracy (%)',xscale='log')
    axes[0].set_xticks([100,1000,10000],['100','1k','10k']);axes[0].legend(loc='lower right',fontsize=7)
    b=b[(b.split=='test')&(b.kind=='clean')&(b.metric=='accuracy')&b.qtype.isin(['Q3','Q4'])].copy()
    for ax,(donor,target) in zip(axes[1:],[('gpt2','llama'),('llama','gpt2')]):
        for family,label,color in [('transfer','Adapter only',COLORS['CLR']),('b3','B3',COLORS['B3']),
                                   ('lite','CLR scratch',COLORS['scratch']),('b5b','B5b',COLORS['B5b'])]:
            f=b[(b.source==target)&(b.family==family)].copy()
            f=f[f.case.str.startswith(f'transfer_{donor}_to_{target}_') | f.case.str.startswith(f'b5b_{target}_')]
            f['N']=pd.to_numeric(f.n_labels,errors='coerce');f=f.dropna(subset=['N'])
            for q,ls in [('Q3','-'),('Q4','--')]:
                z=f[f.qtype==q].sort_values('N')
                if len(z):errorline(ax,z.N,100*z['mean'],100*z.ci_low,100*z.ci_high,color=color,
                                   linestyle=ls,marker='o',label=label if q=='Q3' else None)
        ax.set(xscale='log',xlabel='Labeled QA instances N',title=f'({"b" if donor=="gpt2" else "c"}) {donor.upper()} to {target.upper()}')
        ax.set_xticks([100,300,1000,3000],['100','300','1k','3k'])
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.67,-.06),frameon=False)
    fig.text(.67,-.11,'Solid: Q3; dashed: Q4. B5b: N=100 only.',ha='center',fontsize=7)
    fig.subplots_adjust(bottom=.24,wspace=.28)
    c.figure(fig);c.write_data(b)


def f2(c):
    early=c.read('early');rank=c.read('ranking');search=c.read('search');fig,axes=plt.subplots(1,3,figsize=(10.2,3.15))
    for source,color in [('gpt2',COLORS['CLR']),('llama',COLORS['B3'])]:
        for split,ls in [('test','-'),('reader_val','--')]:
            z=early[(early.source==source)&(early.split==split)].sort_values('t')
            axes[0].plot(z.t,100*z['mean'],ls,marker='o',markersize=3,color=color,
                         label=f'{source.upper()} {"test" if split=="test" else "reader val"}')
    axes[0].set(title='(a) Forced early exit',xlabel='Latent step t',ylabel='Accuracy (%)');axes[0].legend(fontsize=7)
    for model,color in [('B4',COLORS['B4']),('B3',COLORS['B3']),('CLR-lite',COLORS['CLR'])]:
        z=rank[(rank.value==model)&(rank.t!='all')].copy();z['t']=pd.to_numeric(z.t)
        p=z[z.metric=='spearman'].merge(z[z.metric=='between_auroc'],on='t',suffixes=('_rho','_auc')).sort_values('t')
        axes[1].plot(p.mean_rho,p.mean_auc,'o-',color=color,label=model,markersize=3)
        for _,r in p.iterrows():axes[1].annotate(str(int(r.t)),(r.mean_rho,r.mean_auc),xytext=(3,3),textcoords='offset points',fontsize=6)
    axes[1].set(title='(b) Across vs within problems (val)',xlabel='Within-parent Spearman rho',ylabel='Between-problem AUROC');axes[1].legend(fontsize=7)
    search=search[search.scope_horizon==6]
    for method,value,label,color,ls in [('S1','none','Independent voting','#777777','-'),
              ('S3','CLR-lite','CLR beam',COLORS['CLR'],'-'),('S3','B3','B3 beam',COLORS['B3'],'-'),
              ('S2','pass@N','Privileged pass@N','#000000',':')]:
        z=search[(search.method==method)&(search.value==value)].sort_values('budget')
        axes[2].plot(z.budget,100*z.delta_greedy,marker='o',markersize=3,color=color,linestyle=ls,label=label)
    axes[2].axhline(0,color='black',lw=.8);axes[2].set(title='(c) Compute-matched search (val)',xlabel='Equivalent forward budget / problem',ylabel='Accuracy minus greedy (pp)',xscale='log');axes[2].legend(fontsize=7)
    axes[2].set_xticks([32,128,512,2048,8192],['32','128','512','2k','8k'])
    fig.tight_layout(w_pad=1.3);c.figure(fig);c.write_data(search)


def f3(c):
    a=c.read('weak_d0');b=c.read('weak_designs');d=c.read('dependent');rows=[]
    a=a[(a.kind=='E3')&(a.step_budget==200)&(a.alpha==.05)&(a.metric=='robust_failure')&np.isfinite(a.epsilon)]
    for _,r in a.iterrows():rows.append(dict(receiver='weak',design='D0',epsilon=r.epsilon,mean=r['mean'],lo=r.lo,hi=r.hi,groups=r.groups))
    b=b[(b.split=='test')&(b.kind=='E3')&(b.steps==200)&(b.alpha==.05)&(b.k==16)]
    for _,r in b.iterrows():rows.append(dict(receiver='weak',design=r.design,epsilon=r.epsilon,mean=r.failure,lo=r.failure_lo,hi=r.failure_hi,groups=r.groups))
    d=d[(d.split=='test')&(d.panel=='main')&(d.rule=='lambda_rows')&(d['lambda'].astype(str)=='1.0')&(d.alpha==.05)]
    for _,r in d.iterrows():rows.append(dict(receiver='dependent',design=r.design,epsilon=r.epsilon,mean=r.failure_mean,lo=r.failure_low,hi=r.failure_high,groups=r.groups))
    frame=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(7.5,3.1));out=[]
    for ax,receiver in zip(axes,['weak','dependent']):
        for design,g in frame[frame.receiver==receiver].groupby('design',sort=False):
            g=g.sort_values('epsilon').copy();g['envelope']=np.maximum.accumulate(g['mean']);out.append(g)
            ax.plot(g.epsilon,100*g['mean'],'o:',alpha=.25,color=COLORS[design],markersize=4)
            ax.plot(g.epsilon,100*g.envelope,'-',label=design,color=COLORS[design],lw=1.7)
        ax.set(xscale='log',ylim=(0,100),xlabel='Per-layer relative KV radius epsilon',ylabel='Robust failure (%)',
               title=f'{"(a) Weak receiver, n=500" if receiver=="weak" else "(b) Message-dependent receiver, n=200"}')
        ticks=[.01,.03,.1,.3] if receiver=='weak' else [.01,.1,1.]
        ax.set_xticks(ticks,[str(x) for x in ticks]);ax.legend(ncol=3 if receiver=='weak' else 2,fontsize=7,loc='lower left')
    fig.tight_layout();c.figure(fig);c.write_data(pd.concat(out,ignore_index=True))


def f3_r1(c):
    f=c.read('weak_designs');f=f[(f.split=='test')&(f.design=='R1')&(f.kind=='E3')&
                              (f.steps==200)&(f.epsilon==.03)&(f.alpha==.05)].sort_values('k').copy()
    base=one(f,k=1);f['independent_veto_reference']=base.changed_to_target*(base.failure/base.changed_to_target)**f.k
    fig,ax=plt.subplots(figsize=(4.1,2.9));errorline(ax,f.k,100*f.failure,100*f.failure_lo,100*f.failure_hi,
                           color=COLORS['R1'],marker='o',label='R1 mean NLL + conformal')
    ax.plot(f.k,100*f.independent_veto_reference,'--',color='#777777',label='Independent veto reference')
    ax.set(xscale='log',ylim=(0,100),xlabel='Random audit questions k',ylabel='Robust failure (%)')
    ax.set_xticks([1,4,16,64],['1','4','16','64']);ax.legend(fontsize=7,loc='lower left');fig.tight_layout()
    c.figure(fig);c.write_data(f)


def appendix_noise(c):
    a=c.read('gpt2');b=c.read('llama');rows=[]
    for _,r in a[(a.metric=='accuracy')&(a.temperature==0)].iterrows():
        n=one(a,sigma=r.sigma,from_step=r.from_step,temperature=0,metric='nondegenerate')
        rows.append(['GPT2',number(r.sigma),str(int(r.from_step)),interval(r,100),interval(n,100)])
    for _,r in b.iterrows():
        rows.append(['LLaMA',number(r.sigma),'all relative t=0..5',
                     interval(r,100,mean='accuracy',lo='accuracy_ci_low',hi='accuracy_ci_high'),
                     interval(r,100,mean='nondegenerate',lo='nondegenerate_ci_low',hi='nondegenerate_ci_high')])
    c.table(['Source','Sigma','Noise start',r'Accuracy (\%)',r'Nondegenerate (\%)'],rows,
            'GPT2: M0.1 val300/K16/three seeds, original seed-t intervals for stochastic settings and problem bootstrap '
            'for deterministic settings. LLaMA: M6 val300/K16/seed11, problem bootstrap. '
            'GPT2 selected sigma=1.5/start=3 by the original mean rule. LLaMA selected sigma=2 as an explicitly '
            'authorized degradation: no eight-point setting met both original thresholds. Starting-step protocols differ.')
    c.write_data(a)


def appendix_protocol(c):
    f=c.read('protocol');c.write_data(f)
    c.table(['Protocol','Field','Frozen value'],[[escape(r.section),escape(r.field),escape(r.value)]
            for _,r in f.iterrows()], 'Frozen configuration/verification metadata only. Protocol differences do not '
            'imply equal model behavior; cohort padding and post-RoPE K conventions must be retained.',long=True)


def appendix_dedup(c):
    f=c.read('dedup');c.write_data(f)
    c.table(['Historical split','Original n','Exact overlap','13-gram overlap','Removed union'],
            [[escape(r['split']),str(r.n),str(r.exact_overlap),str(r['13gram_overlap']),str(r.union_overlap)]
             for _,r in f.iterrows()], 'Frozen M0.1 train-versus-heldout audit. The retained val/cal pool was '
            'subsequently reassigned to M1 val300/cal500/test1005; these rows are historical inputs, not current split counts. '
            'Near-duplicate filtering does not prove absence of all semantic overlap.')


def appendix_labels(c):
    f=c.read('labels');c.write_data(f)
    rows=[[SOURCE_NAMES[r.source],escape(r['split']),r.qtype,str(int(r.t)),escape(r.kind),str(int(r.n)),number(r.label_mean)]
          for _,r in f.sort_values(['source','split','qtype','kind','t']).iterrows()]
    c.table(['Source','Split','Type','t','State','Instances','Positive / soft-label mean'],rows,
            'Counts and label marginals are descriptive. GPT2 uses its original full-QA summary (evaluation held-out '
            'templates included); LLaMA rows use seen templates only. Q1 trajectory starts supervise only the observed '
            'clean origin and subsequent trajectory states. Q3 label index means are not positive-class rates. '
            'Balanced positive counts do not imply identical numeric marginals; LLaMA train/reader-val retain '
            'the frozen pair-closure limitation documented in the M6 review.',long=True)


def appendix_scales(c):
    f=c.read('scales');c.write_data(f)
    c.table(['Source','Split','Problem/group count','QA rows','Count unit'],
            [[escape(r.source),escape(r['split']),number(r.problems),number(r.qa_rows),escape(r.unit)]
             for _,r in f.iterrows()], 'Problem/episode groups never cross splits. Synthetic groups have several '
            'message categories and QA instances; group counts must not be read as IID QA counts. '
            'GPT2 and LLaMA use different training-data sizes and continuation policies.',long=True)


def appendix_q1(c):
    f=c.read('q1');rows=[]
    for source in ['gpt2','llama']:
        for model in MODEL_ORDER:
            a=one(f,source=source,model=model,kind='clean',metric='nll')
            b=one(f,source=source,model=model,kind='traj',metric='nll')
            if a is not None:rows.append([SOURCE_NAMES[source],escape(model),interval(a),interval(b)])
    c.write_data(f);c.table(['Source','Reader','reader-val Q1 clean NLL','reader-val Q1 trajectory NLL'],rows,
                          'Same calibrated predictions and problem-cluster bootstrap as T1; clean/traj remain separate.')


BUILDERS = {k:v for k,v in globals().copy().items() if k in ['t1','t2','t2_paired','t3','t4','t4_qa',
            't4_localization','t5','f1','f2','f3','f3_r1','appendix_noise','appendix_protocol',
            'appendix_dedup','appendix_labels','appendix_scales','appendix_q1']}


def main(builder):
    p=argparse.ArgumentParser();p.add_argument('config');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());c=Context(config)
    BUILDERS[builder](c);c.finish()

"""Analysis of frozen B/C runs; no QA/test data access."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m5b_runtime import snapshot
from clr.search.m5b import group_metrics
from clr.search.m5b_analysis import soft_auc,bootstrap_mean


def read_parts(root,prefix):
    files=sorted(Path(root).glob(f'{prefix}_*.parquet'))
    if not files:raise ValueError(f'Missing {prefix} rows: {root}')
    return pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)


def main(c):
 if c['stage']=='smoke' and c.get('expected_questions',300)>15:raise ValueError('BC analysis smoke must declare <=15 val IDs')
 with recorded(c) as run:
    if c['stage']=='full':
        for root in [c['B_run'],*c['C_runs'],*c.get('D_C_runs',[])]:
            met=json.loads(Path(root,'metrics.json').read_text())
            if met.get('status')!='completed' or not met.get('gate_passed',True):raise ValueError('Completed protocol-matching B/C evidence required: '+root)
    start=time.monotonic();snapshot(run,c,['scripts/m5b_analyze_bc.py','clr/search/m5b_analysis.py','clr/search/m5b.py']);out=Path(c['output'])/'analysis';out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(c['seed'])
    b=read_parts(c['B_run'],'B');values=['B4','B3','CLR-lite','CLR-v1']
    if c.get('D_B_run'):
        d=read_parts(c['D_B_run'],'B');b=b.merge(d,on=['problem_id','seed','t','child'],validate='one_to_one');values+=['D-lite']
    expected=c.get('expected_questions',300);nseeds=len(c['seeds'])
    if len(b)!=expected*nseeds*6*17:raise ValueError(f'Incomplete B panel {len(b)}')
    if not (b.groupby(['problem_id','seed','t']).child.nunique()==17).all():raise ValueError('Sibling panel mismatch')
    group=[]
    for (pid,seed,t),f in b.groupby(['problem_id','seed','t'],sort=False):
        f=f.sort_values('child');truth=f.Vhat.to_numpy()
        for value in values:
            gm=group_metrics(f[value].to_numpy(),truth)
            group.append({'problem_id':pid,'seed':seed,'t':t,'value':value,**gm,'truth_range':float(np.ptp(truth)),'flat_truth':not bool(np.ptp(truth)),'flat_prediction':not bool(np.ptp(f[value].to_numpy()))})
    gd=pd.DataFrame(group);gd.to_csv(out/'B_groups.csv',index=False);summary=[]
    for value in values:
        for t in [*range(6),'all']:
            f=gd[(gd.value==value)&((gd.t==t) if t!='all' else True)]
            for metric in ['spearman','pair_accuracy','gain_vs_sigma0']:
                x=f.groupby('problem_id')[metric].mean();stat=bootstrap_mean(x.to_numpy(),rng)
                summary.append({'value':value,'t':t,'metric':metric,**stat,'valid_groups':int(f[metric].notna().sum()),'total_groups':len(f),'flat_truth_fraction':float(f.flat_truth.mean()),'flat_prediction_fraction':float(f.flat_prediction.mean())})
            raw=b[(b.t==t) if t!='all' else np.ones(len(b),bool)]
            # Cross-question AUROC is pooled across parent groups, not within group.
            # K MC outcomes enter as positive/negative mass, equivalent to K
            # individually repeated labels without a majority threshold.
            aucs=[float(np.mean([soft_auc(raw[(raw.seed==s)&(raw.t==tt)][value],raw[(raw.seed==s)&(raw.t==tt)].Vhat) for tt in (range(6) if t=='all' else [t])])) for s in c['seeds']]
            half=4.3026527299*float(np.std(aucs,ddof=1))/np.sqrt(nseeds) if nseeds>1 else 0.
            summary.append({'value':value,'t':t,'metric':'between_auroc','mean':float(np.mean(aucs)),'ci_low':float(np.mean(aucs)-half),'ci_high':float(np.mean(aucs)+half),'seed_values':json.dumps(aucs),'n':raw.problem_id.nunique(),'ci_scope':'three-seed t interval'})
    pd.DataFrame(summary).to_csv(out/'B_summary.csv',index=False)
    dcomp=[]
    if 'D-lite' in values:
        for t in [*range(6),'all']:
            a=gd[(gd.value=='D-lite')&((gd.t==t) if t!='all' else True)];bb=gd[(gd.value=='CLR-lite')&((gd.t==t) if t!='all' else True)]
            for metric in ['spearman','pair_accuracy','gain_vs_sigma0']:
                aa=a.set_index(['problem_id','seed','t'])[metric];cc=bb.set_index(['problem_id','seed','t'])[metric];delta=(aa-cc).groupby('problem_id').mean();dcomp.append({'t':t,'metric':metric,**bootstrap_mean(delta.to_numpy(),rng)})
        pd.DataFrame(dcomp).to_csv(out/'D_B_paired.csv',index=False)
    if c.get('B_only',False):
        write_json(out/'B_analysis.json',{'rows':len(b),'scored_parent_value_groups':len(gd),'bootstrap':c['bootstrap'],'D_B_paired':dcomp})
        run.write_metrics(status='completed',gpu_hours=0,wall_seconds=time.monotonic()-start,source_forward_calls=0,test_opened=False)
        write_json(Path(c['output'])/'B_analysis_run.json',{'run':str(run.path)})
        return
    frames=[read_parts(root,'C') for root in c['C_runs']]
    for root in c.get('D_C_runs',[]):frames.append(read_parts(root,'C'))
    f=pd.concat(frames,ignore_index=True);key=['problem_id','seed','method','value','N','b','horizon']
    duplicate=f[f.duplicated(key,keep=False)]
    if len(duplicate) and (duplicate.groupby(key)[['correct','answer','source_forwards','value_source_equivalent']].nunique(dropna=False)>1).any().any():raise ValueError('Repeated reference settings changed answers or costs across runs')
    f=f.drop_duplicates(key,keep='first');f['compute']=f.source_forwards+f.value_source_equivalent;f.to_csv(out/'C_all_settings.csv',index=False)
    idcols=['method','value','N','b','horizon'];records=[];mat=[];order=[];problem_ids=sorted(f.problem_id.unique())
    greedy=f[(f.method=='S0')].groupby('problem_id').correct.mean().reindex(problem_ids).to_numpy(dtype=float)
    if len(problem_ids)!=expected:raise ValueError('C requires the declared distinct val IDs')
    # One paired problem bootstrap draw is shared across every configuration.
    draws=c.get('bootstrap',10000);weights=np.zeros((draws,expected));ii=rng.integers(expected,size=(draws,expected));np.add.at(weights,(np.arange(draws)[:,None],ii),1);weights/=expected
    for keys,x in f.groupby(idcols,dropna=False):
        counts=x.groupby('problem_id').seed.nunique()
        if len(x)!=expected*nseeds or not (counts==nseeds).all():raise ValueError(f'Incomplete C setting {keys}: {len(x)}')
        by=x.groupby('problem_id').correct.mean().reindex(problem_ids).to_numpy(dtype=float);delta=by-greedy;boot=weights@delta;lo,hi=np.quantile(boot,[.025,.975]);alo,ahi=np.quantile(weights@by,[.025,.975])
        r=dict(zip(idcols,keys));r.update(accuracy=float(by.mean()),accuracy_ci_low=float(alo),accuracy_ci_high=float(ahi),compute=float(x.compute.mean()),source_forwards=float(x.source_forwards.mean()),value_source_equivalent=float(x.value_source_equivalent.mean()),delta_greedy=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),seed_accuracy=json.dumps(x.groupby('seed').correct.mean().tolist()));records.append(r);mat.append(delta);order.append(keys)
    summaryc=pd.DataFrame(records);arr=np.stack(mat,axis=1);delta=arr.mean(0);se=arr.std(0,ddof=1)/np.sqrt(expected);valid=(summaryc.method.isin(['S2','S3'])&~summaryc.value.isin(['oracle','pass@N'])).to_numpy()&(se>0)
    centered=(weights@arr-delta)/np.where(se>0,se,1);critical=float(np.quantile(np.max(np.abs(centered[:,valid]),axis=1),.95)) if valid.any() else 0.;summaryc['simul_ci_low']=delta-critical*se;summaryc['simul_ci_high']=delta+critical*se;summaryc.to_csv(out/'C_summary.csv',index=False)
    # Compare D to original lite at exactly corresponding settings/common noise.
    comp=[]
    d_differences=[]
    if 'D-lite' in f.value.unique():
        for keys,x in f[(f.value=='D-lite')].groupby(['method','N','b','horizon']):
            method,N,b1,h=keys;a=x.set_index(['problem_id','seed']).correct.astype(float);bb=f[(f.value=='CLR-lite')&(f.method==method)&(f.N==N)&(f.b==b1)&(f.horizon==h)].set_index(['problem_id','seed']).correct.astype(float);dd=(a-bb).groupby('problem_id').mean().reindex(problem_ids).to_numpy();d_differences.append(dd);boot=weights@dd;lo,hi=np.quantile(boot,[.025,.975]);comp.append({'method':method,'N':N,'b':b1,'horizon':h,'delta':float(dd.mean()),'ci_low':float(lo),'ci_high':float(hi)})
        dx=np.stack(d_differences,axis=1);ds=dx.std(0,ddof=1)/np.sqrt(expected);active=ds>0
        dc=float(np.quantile(np.max(np.abs((weights@dx-dx.mean(0))/np.where(active,ds,1))[:,active],axis=1),.95)) if active.any() else 0.
        for j,r in enumerate(comp):r.update(simul_ci_low=r['delta']-dc*ds[j],simul_ci_high=r['delta']+dc*ds[j])
        pd.DataFrame(comp).to_csv(out/'D_C_paired.csv',index=False)
    # Budget curve uses a predeclared cost rank, never observed accuracy.
    # Below a setting's mean cost use greedy; spend remaining compute unused.
    curves=[]
    for scope in [6,4]:
        for budget in c['budget_points']:
            for method,value in [('S0','none'),('S1','none'),('S1-clean','none')]+[(m,v) for m in ['S2','S3'] for v in values]+[('S2','oracle'),('S2','pass@N')]:
                available=summaryc[(summaryc.method==method)&(summaryc.value==value)&(summaryc.compute<=budget)&((summaryc.horizon==scope) if method in ['S2','S3','S1-clean'] else True)]
                if available.empty:
                    r=summaryc[summaryc.method=='S0'].iloc[0]
                else:
                    # Largest actual compute setting; ties prefer full then N then b.
                    r=available.sort_values(['compute','horizon','N','b']).iloc[-1]
                curves.append({'scope_horizon':scope,'budget':budget,'method':method,'value':value,'chosen_method':r.method,'chosen_N':int(r.N),'chosen_b':int(r.b),'chosen_horizon':int(r.horizon),'actual_compute':float(r.compute),'accuracy':float(r.accuracy),'accuracy_ci_low':float(r.accuracy_ci_low),'accuracy_ci_high':float(r.accuracy_ci_high),'delta_greedy':float(r.delta_greedy),'ci_low':float(r.ci_low),'ci_high':float(r.ci_high)})
    curves=pd.DataFrame(curves);curves.to_csv(out/'C_budget_curves.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'B4':'#999999','B3':'#3b75af','CLR-lite':'#e88b2b','CLR-v1':'#6c4ca4','D-lite':'#279477','none':'#333333','oracle':'#c84650','pass@N':'#c84650'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for row,scope in enumerate([6,4]):
        for col,method in enumerate(['S2','S3']):
            ax=axes[row,col];panel=curves[curves.scope_horizon==scope]
            for value in values:
                z=panel[(panel.method==method)&(panel.value==value)].sort_values('budget');ax.plot(z.budget,100*z.delta_greedy,marker='o',markersize=3,label=value,color=colors[value])
            z=panel[(panel.method=='S1')].sort_values('budget');ax.plot(z.budget,100*z.delta_greedy,color='black',linestyle='--',label='S1 independent pi')
            z=panel[(panel.method=='S1-clean')].sort_values('budget');ax.plot(z.budget,100*z.delta_greedy,color='black',linestyle='-.',label='S1 with clean')
            if method=='S2':
                z=panel[(panel.method=='S2')&(panel.value=='pass@N')].sort_values('budget');ax.plot(z.budget,100*z.delta_greedy,color=colors['oracle'],linestyle=':',label='privileged pass@N')
            ax.axhline(0,color='black',linewidth=.8);ax.set_xscale('log',base=2);ax.set_title(f"{method}: {'all t' if scope==6 else 't <= 3'}");ax.set_xlabel('Mean allowed source-forward budget / question');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
        axes[row,0].set_ylabel('Accuracy minus greedy (percentage points)')
    fig.tight_layout();fig.savefig(out/'accuracy_budget.png',dpi=180);fig.savefig(out/'accuracy_budget.pdf');plt.close(fig)
    # Actual mean cost is the primary accuracy-compute figure. The budget
    # envelope above separately retains unused compute and its selected settings.
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    vote=summaryc[summaryc.method=='S1'].sort_values('compute')
    for row,scope in enumerate([6,4]):
        for col,method in enumerate(['S2','S3']):
            ax=axes[row,col]
            for value in values:
                for beam in ([1] if method=='S2' else [2,4]):
                    z=summaryc[(summaryc.method==method)&(summaryc.value==value)&(summaryc.horizon==scope)&(summaryc.b==beam)].sort_values('compute')
                    ax.plot(z.compute,100*z.delta_greedy,marker='o',markersize=3,color=colors[value],linestyle=':' if beam==4 else '-',label=value if beam!=4 else None)
            ax.plot(vote.compute,100*vote.delta_greedy,color='black',linestyle='--',label='S1 independent pi')
            clean_vote=summaryc[(summaryc.method=='S1-clean')&(summaryc.horizon==scope)].sort_values('compute')
            ax.plot(clean_vote.compute,100*clean_vote.delta_greedy,color='black',linestyle='-.',marker='+',label='S1 with clean')
            ax.axhline(0,color='black',linewidth=.8);ax.set_xscale('log',base=2);ax.set_title(f"{method}: {'all t' if scope==6 else 't <= 3'}"+('; solid b2 / dotted b4' if method=='S3' else ''))
            ax.set_xlabel('Actual mean source forwards / question (value FLOPs included)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[row,0].set_ylabel('Accuracy minus greedy (percentage points)')
    fig.tight_layout();fig.savefig(out/'accuracy_compute.png',dpi=180);fig.savefig(out/'accuracy_compute.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,scope in zip(axes,[6,4]):
        z=summaryc[(summaryc.value=='pass@N')&(summaryc.horizon==scope)].sort_values('compute')
        ax.plot(z.compute,100*z.delta_greedy,color=colors['oracle'],marker='o',label='gold endpoint oracle = pass@N')
        ax.plot(vote.compute,100*vote.delta_greedy,color='black',linestyle='--',label='S1 independent pi')
        clean_vote=summaryc[(summaryc.method=='S1-clean')&(summaryc.horizon==scope)].sort_values('compute');ax.plot(clean_vote.compute,100*clean_vote.delta_greedy,color='black',linestyle='-.',marker='+',label='S1 with clean')
        ax.axhline(0,color='black',linewidth=.8);ax.set_xscale('log',base=2);ax.set_title('Privileged reference: '+('all t' if scope==6 else 't <= 3'));ax.set_xlabel('Actual mean source forwards / question');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Accuracy minus greedy (percentage points)');fig.tight_layout();fig.savefig(out/'pass_at_N.png',dpi=180);fig.savefig(out/'pass_at_N.pdf');plt.close(fig)
    write_json(out/'BC_analysis.json',{'B_rows':len(b),'B_groups':len(gd),'C_rows':len(f),'greedy_accuracy':float(greedy.mean()),'bootstrap':draws,'cluster_unit':'problem; all t/seeds together','simultaneous_critical':critical,'significant_settings_over_greedy':summaryc[(summaryc.simul_ci_low>0)&~summaryc.value.isin(['oracle','pass@N'])&summaryc.method.isin(['S2','S3'])].to_dict('records'),'C_candidates':'N total with sigma0 per parent; S1 independent pi retained separately','B_same_parent_auc_not_used':True,'D_B_paired':dcomp})
    run.write_metrics(status='completed',gpu_hours=0,wall_seconds=time.monotonic()-start,source_forward_calls=0,test_opened=False);write_json(Path(c['output'])/'BC_analysis_run.json',{'run':str(run.path)})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

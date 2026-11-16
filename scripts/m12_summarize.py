"""Read-only provenance-aware paired problem-cluster summaries (10k draws)."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts.m9_export import Export
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.calib.m4_evaluation import unique_indices
from clr.calib.m12_statistics import metric_summary
from clr.calib.m4_statistics import WeightedAUC

METRICS={'Q1':'nll','Q2':'auroc','Q3':'accuracy','Q4':'accuracy'}

def registry(c):
    m41=json.loads(Path('data/M41_v1/evaluation_suite.json').read_text())
    single=json.loads(Path('data/M5b_v1/completion.json').read_text())['A']['runs']
    multi=json.loads(Path('data/M9_v1/evaluation_index.json').read_text())
    regs=[]
    for s in [11,22,33]:
        for variant,key in [('CLR-lite','main_lite'),('CLR-v1','main_v1')]:regs.append(dict(variant=variant,model=variant,seed=s,run=m41[f'{key}_{s}'],source='gpt2',native=False,reused=True,origin='M4.1'))
        regs.append(dict(variant='CLR-lite-single',model='CLR-lite-single',seed=s,run=single[f'CLR-lite-single_{s}'],source='gpt2',native=False,reused=True,origin='M5b'))
        per=json.loads(Path(f'data/M10_v1/evaluated/gpt2_CLR-per-type_{s}_full.json').read_text())['run']
        regs.append(dict(variant='CLR-per-type',model='CLR-per-type',seed=s,run=per,source='gpt2',native=True,reused=True,origin='M10'))
        regs.append(dict(variant='CLR-lite-multi',model='CLR-lite-multi',seed=s,run=multi[f'multi_lite_{s}']['gpt2'],source='gpt2',native=True,reused=True,origin='M9'))
    if c.get('mode')=='source_val_smoke':return regs
    evals=json.loads(Path(c['output'],'evaluation_complete.json').read_text())
    for key,run in evals.items():
        a,s=key.split('_');regs.append(dict(variant=a,model=a,seed=int(s),run=run,source='gpt2',native=False,reused=False,origin='M12'))
    return regs

def auc_pair(frame,baseline,draws=10000):
    """One shared cluster draw for all steps, seeds and paired models."""
    ids=sorted(frame.problem_id.unique());lookup={p:i for i,p in enumerate(ids)};panels=[]
    if ids!=sorted(baseline.problem_id.unique()):raise ValueError('AUROC problem support mismatch')
    for seed in [11,22,33]:
        for t in sorted(frame.t.unique()):
            pair=[]
            for f in [frame,baseline]:
                g=f[(f.seed==seed)&(f.t==t)].sort_values('problem_id')
                if g.empty:raise ValueError('Missing AUROC panel')
                ix=np.array([lookup[p] for p in g.problem_id]);pair.append((WeightedAUC(g.label.to_numpy(),g.score.to_numpy()),ix))
            ga=frame[(frame.seed==seed)&(frame.t==t)].sort_values('problem_id');gb=baseline[(baseline.seed==seed)&(baseline.t==t)].sort_values('problem_id')
            if ga.problem_id.tolist()!=gb.problem_id.tolist() or not np.array_equal(ga.label,gb.label):raise ValueError('AUROC outcomes misaligned')
            panels.append(pair)
    def calculate(w):
        a=np.mean([p[0][0](w[:,p[0][1]]) for p in panels],axis=0)
        b=np.mean([p[1][0](w[:,p[1][1]]) for p in panels],axis=0)
        return np.stack([a,a-b],1)
    point=calculate(np.ones((1,len(ids))))[0];rng=np.random.default_rng(11);boot=[]
    for j in range(0,draws,100):
        n=min(100,draws-j);ii=rng.integers(len(ids),size=(n,len(ids)));w=np.zeros((n,len(ids)),dtype=np.int32)
        for k in range(n):w[k]=np.bincount(ii[k],minlength=len(ids))
        boot.append(calculate(w))
    ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
    seeds=[]
    for s in [11,22,33]:
        vals=[float(p[0][0](np.ones((1,len(ids)))[:,p[0][1]])[0]) for i,p in enumerate(panels) if i//len(frame.t.unique())==[11,22,33].index(s)];seeds.append(float(np.mean(vals)))
    return point,ci,seeds

def summarize(c):
    root=Path(c['output']);ex=Export(c);regs=registry(c);points=[];aucs=[];cps=[];q1=[]
    write_json(root/'source_registry.json',regs)
    for r in regs:
        for split in ['reader_val','test']:
            for q in ['Q1','Q2','Q3','Q4']:
                raw,p,y,f=ex.predict_rows(r,q,split);co={k:r[k] for k in ['variant','seed','reused','origin','run']};co.update(split=split,qtype=q)
                g=f.groupby(['problem_id','t'])[['accuracy','nll']].mean();g['instances']=f.groupby(['problem_id','t']).size();points.append(g.reset_index().assign(**co))
                if q=='Q2':aucs.append(f.assign(**co))
                if q=='Q1' and split=='reader_val':
                    q1.append(g.reset_index().assign(**co,kind='clean'))
                    rr,pp,yy,ff=ex.predict_rows(r,q,split,'traj')
                    if rr:
                        gg=ff.groupby(['problem_id','t'])[['nll','accuracy']].mean();gg['instances']=ff.groupby(['problem_id','t']).size();q1.append(gg.reset_index().assign(**co,kind='traj'))
                if q not in ['Q3','Q4']:continue
                normalized=raw if r['native'] else ex.metadata_rows(raw,q,split)
                conf=ex.js(Path(r['run'])/'conformal.json')
                for cp in conf:
                    if cp['split']!=split or cp['qtype']!=q or cp['heldout'] or cp['alpha']!=.1 or cp.get('calibration','temperature')!='temperature' or cp.get('subset','all')!='all':continue
                    t=cp['t'];thr=np.inf if cp['threshold'] is None else cp['threshold'];jj=unique_indices(normalized,[i for i,a in enumerate(normalized) if a['state_ref']['t']==t],q,r['seed']);sets=1-p[jj]<=thr
                    if abs(float(sets.sum(1).mean())-cp['size'])>1e-7:raise ValueError('Conformal size replay mismatch')
                    for i,ss in zip(jj,sets):cps.append({**co,'problem_id':raw[i]['problem_id'],'t':t,'instances':1,'size':float(ss.sum()),'coverage':float((ss*y[i]).sum())})
        print('collected',r['variant'],r['seed'],flush=True)
    frame=pd.concat(points,ignore_index=True);af=pd.concat(aucs,ignore_index=True);cpf=pd.DataFrame(cps)
    frame.to_parquet(root/'problem_metrics.parquet',index=False);af.to_parquet(root/'auroc_observations.parquet',index=False);cpf.to_parquet(root/'conformal_observations.parquet',index=False)
    result=[]
    for (variant,split,q),g in frame.groupby(['variant','split','qtype']):
        b=frame[(frame.variant=='CLR-lite')&(frame.split==split)&(frame.qtype==q)];metric=METRICS[q]
        co={'variant':variant,'split':split,'qtype':q,'metric':metric,'reused':bool(g.reused.iloc[0]),'source_milestone':g.origin.iloc[0],'source_runs':'|'.join(sorted(g.run.unique())),'seeds':'11|22|33','bootstrap_draws':10000,'bootstrap_unit':'problem','aggregation':'clean/seen; micro within step, macro over t and fixed seeds','paired_reference':'CLR-lite','n_problems':g.problem_id.nunique()}
        if metric=='auroc':
            a=af[(af.variant==variant)&(af.split==split)];base=af[(af.variant=='CLR-lite')&(af.split==split)]
            point,ci,seed_values=auc_pair(a,base)
            row={**co,'mean':point[0],'ci_low':ci[0,0],'ci_high':ci[1,0],'paired_delta':point[1],'paired_ci_low':ci[0,1],'paired_ci_high':ci[1,1],**{f'seed_{s}':v for s,v in zip([11,22,33],seed_values)}}
        else:
            d=g.merge(b,on=['seed','problem_id','t'],suffixes=('_a','_b'),validate='one_to_one')
            if len(d)!=len(g) or len(d)!=len(b) or not np.array_equal(d.instances_a,d.instances_b):raise ValueError('Paired sample/count support mismatch')
            d['value']=d[metric+'_a'];d['delta']=d[metric+'_a']-d[metric+'_b'];d['instances']=d.instances_a
            e=metric_summary(d,['value','delta']);seed_values={}
            for s in [11,22,33]:
                f=d[d.seed==s]
                seed_values[s]=float((f.assign(weighted=f.value*f.instances).groupby('t').weighted.sum()/f.groupby('t').instances.sum()).mean())
            row={**co,'mean':e[0]['mean'],'ci_low':e[0]['lo'],'ci_high':e[0]['hi'],'paired_delta':e[1]['mean'],'paired_ci_low':e[1]['lo'],'paired_ci_high':e[1]['hi'],**{f'seed_{s}':v for s,v in seed_values.items()}}
        if q in ['Q3','Q4']:
            cc=cpf[(cpf.variant==variant)&(cpf.split==split)&(cpf.qtype==q)];e=metric_summary(cc,['size','coverage'])
            row.update(conformal_alpha=.1,conformal_size=e[0]['mean'],conformal_size_ci_low=e[0]['lo'],conformal_size_ci_high=e[0]['hi'],conformal_coverage=e[1]['mean'],conformal_coverage_ci_low=e[1]['lo'],conformal_coverage_ci_high=e[1]['hi'],conformal_source='cal_B Mondrian (qtype,t), one instance/problem/t')
            cb=cpf[(cpf.variant=='CLR-lite')&(cpf.split==split)&(cpf.qtype==q)]
            cd=cc.merge(cb,on=['seed','problem_id','t'],suffixes=('_a','_b'),validate='one_to_one')
            if len(cd)!=len(cc) or len(cd)!=len(cb):raise ValueError('Paired conformal support mismatch')
            cd['instances']=1;cd['delta']=cd.size_a-cd.size_b
            ce=metric_summary(cd,['delta'])[0]
            row.update(conformal_size_paired_delta=ce['mean'],conformal_size_paired_ci_low=ce['lo'],conformal_size_paired_ci_high=ce['hi'])
        row['status']='completed'
        result.append(row);print('summarized',variant,split,q,flush=True)
    lock=json.loads((root/'training_complete.json').read_text())
    for a in lock['dropped']:
        for split in ['reader_val','test']:
            for q,metric in METRICS.items():
                result.append({'variant':a,'split':split,'qtype':q,'metric':metric,'reused':False,'source_milestone':'M12','status':'not_completed_budget_or_deadline','mean':np.nan,'ci_low':np.nan,'ci_high':np.nan,'paired_delta':np.nan,'paired_ci_low':np.nan,'paired_ci_high':np.nan,'paired_reference':'CLR-lite','seeds':'11|22|33'})
    pd.DataFrame(result).to_csv(root/'summary.csv',index=False,float_format='%.15g')
    q1frame=pd.concat(q1,ignore_index=True);out=[]
    for key,g in q1frame.groupby(['variant','kind']):
        for e in metric_summary(g,['nll']):out.append(dict(zip(['variant','kind'],key),split='reader_val',**e))
    pd.DataFrame(out).to_csv(root/'reader_val_q1_clean_traj.csv',index=False)
    write_json(root/'statistics_provenance.json',{'source_sha256':ex.used,'no_training_or_recalibration_for_reused':True,'draws':10000,'delta_direction':'variant minus CLR-lite','CI':'percentile problem cluster bootstrap; 3 fixed seeds averaged in each draw; conditional on original calibration sets'})
    write_json(root/'statistics_complete.json',{'rows':len(result),'variants':sorted(frame.variant.unique()),'status':'completed'})
def main(c):
    from clr.utils.records import recorded
    from clr.utils.m12_runtime import digest
    import tarfile
    with recorded({**c,'stage':'CPU_statistics'}) as run:
        source=[Path(__file__),Path('scripts/m9_export.py'),Path('clr/calib/m12_statistics.py'),Path('clr/paper/statistics.py'),Path('scripts/m6_report.py'),Path('clr/calib/m4_statistics.py')]
        write_json(run.path/'startup_code_sha256.json',{str(p):digest(p) for p in source})
        with tarfile.open(run.path/'startup_source.tar.gz','w:gz') as tar:
            for p in source:tar.add(p)
        summarize(c)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=True)
        write_json(Path(c['output'])/'statistics_run.json',{'run':str(run.path)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

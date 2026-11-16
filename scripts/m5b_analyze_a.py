"""M5b-A metrics, paired confound contrasts and exact empirical matched coverage."""
import argparse,json,time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m5b_runtime import snapshot
from clr.calib.m4 import apply
from clr.calib.m4_evaluation import unique_indices,target_array
from clr.calib.m41_statistics import paired
from clr.data.m4.reader import rows_for
from clr.search.m5b_analysis import MatchedCoverage


def main(c):
 if c['stage']!='full':raise ValueError('A test analysis is final-only; smoke uses val evaluators and numerical unit tests')
 with recorded(c) as run:
    start=time.monotonic();snapshot(run,c,['scripts/m5b_analyze_a.py','clr/search/m5b_analysis.py','clr/calib/m41_statistics.py']);root=Path(c['output']);out=root/'analysis';out.mkdir(exist_ok=True)
    complete=json.loads((root/'A_evaluation_complete.json').read_text())['runs']
    if len(complete)!=6:raise ValueError('A final outputs incomplete; analysis cannot read test')
    refs=json.loads(Path('data/M41_v1/evaluation_complete.json').read_text())['runs'];suite={**complete,**{f'{case}_{s}':refs[f'{old}_{s}'] for case,old in [('B3','B3'),('CLR-lite','main_lite'),('CLR-v1','main_v1')] for s in c['seeds']}}
    stats=[];cp=[]
    for key,path in suite.items():
        case,seed=key.rsplit('_',1)
        stats.extend({**r,'case':case,'seed':int(seed)} for r in json.loads(Path(path,'statistics.json').read_text()))
        cp.extend({**r,'case':case,'seed':int(seed)} for r in json.loads(Path(path,'conformal.json').read_text()))
    df=pd.DataFrame(stats);cf=pd.DataFrame(cp);df.to_csv(out/'A_metrics.csv',index=False);cf.to_csv(out/'A_conformal.csv',index=False);primary=df[(df.kind=='clean')&(~df.heldout)&(df.calibration=='temperature')&(df.subset=='all')];ag=[]
    for (case,split,q),f in primary.groupby(['case','split','qtype']):
        for metric in ['nll','accuracy','auroc','brier','mae','ece15']:
            if metric not in f:continue
            x=f.groupby('seed')[metric].mean().to_numpy()
            if np.isnan(x).all():continue
            half=4.3026527299*np.std(x,ddof=1)/np.sqrt(3);ag.append({'case':case,'split':split,'qtype':q,'metric':metric,'mean':float(np.mean(x)),'ci_low':float(np.mean(x)-half),'ci_high':float(np.mean(x)+half),'seed_values':json.dumps(x.tolist())})
    pd.DataFrame(ag).to_csv(out/'A_summary.csv',index=False)
    bykind=[]
    for (case,split,q,kind),f in df[(~df.heldout)&(df.calibration=='temperature')&(df.subset=='all')].groupby(['case','split','qtype','kind']):
        for metric in ['nll','accuracy','auroc','brier','mae','ece15']:
            if metric not in f:continue
            x=f.groupby('seed')[metric].mean().to_numpy()
            if np.isnan(x).all():continue
            half=4.3026527299*np.std(x,ddof=1)/np.sqrt(len(x));bykind.append({'case':case,'split':split,'qtype':q,'kind':kind,'metric':metric,'mean':float(np.mean(x)),'ci_low':float(np.mean(x)-half),'ci_high':float(np.mean(x)+half),'seed_values':json.dumps(x.tolist())})
    pd.DataFrame(bykind).to_csv(out/'A_summary_by_kind.csv',index=False)
    rng=np.random.default_rng(c['seed']);contrasts=[('CLR-lite','B3'),('B3-prefix','B3'),('CLR-lite','B3-prefix'),('CLR-lite-single','B3'),('CLR-lite','CLR-lite-single'),('CLR-v1','B3'),('CLR-v1','B3-prefix')];pairs=[]
    for split in ['reader_val','test']:
        for a,b in contrasts:
            for q,metric in [('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy')]:
                pairs.append(paired(suite,a,b,c['seeds'],split,q,metric,rng,c['bootstrap']))
                pd.DataFrame(pairs).to_csv(out/'A_paired.csv',index=False)
    # Per qtype empirical 90%-coverage set sizes; one fixed question/pid/t.
    matched=[];mc_pairs=[];actual_cp_pairs=[]
    for split in ['reader_val','test']:
        for q in ['Q1','Q2','Q3','Q4']:
            qa=rows_for(c,split,'final',qtypes=[q]);pids=sorted({r['problem_id'] for r in qa});n=len(pids);idmap={p:i for i,p in enumerate(pids)};curves={case:[] for case in ['B3','B3-prefix','CLR-lite','CLR-lite-single','CLR-v1']};ordinary={case:[] for case in curves}
            for seed in c['seeds']:
                # Same selection of one question per pid/t for every model.
                indices={t:unique_indices(qa,[i for i,r in enumerate(qa) if r['state_ref']['kind']=='clean' and r['state_ref']['t']==t],q,seed) for t in range(7)}
                for case in curves:
                    path=Path(suite[f'{case}_{seed}']);frame=pd.read_parquet(path/f'{q}_{split}_False.parquet')
                    if len(frame)!=len(qa) or frame.problem_id.tolist()!=[r['problem_id'] for r in qa]:raise ValueError('A CP row alignment mismatch')
                    cal=json.loads((path/'calibration.json').read_text())[q]['temperature'];p=softmax(apply(np.stack(frame.logits),cal),axis=1);target=target_array(qa,q);th=json.loads((path/'thresholds.json').read_text())[q]['temperature']
                    for t,ii in indices.items():
                        pp=p[ii];yy=target[ii];mapping=np.array([idmap[qa[i]['problem_id']] for i in ii]);curves[case].append((MatchedCoverage(pp,yy),mapping));threshold=th[f'{t}:0.1'];sets=1-pp<=(np.inf if threshold is None else threshold);ordinary[case].append((mapping,(sets*yy).sum(1),sets.sum(1)))
            sizes={case:float(np.mean([curve(np.ones((1,len(mapping))))[0] for curve,mapping in xs])) for case,xs in curves.items()}
            for case,size in sizes.items():matched.append({'case':case,'split':split,'qtype':q,'coverage':.9,'size':size,'mode':'empirical randomized exact-coverage curve; not cal_B guarantee'})
            # Paired problem bootstrap, recomputing the matching threshold within
            # each draw. All t and all three model seeds stay in one cluster.
            boot={case:[] for case in curves};actual={case:{'coverage':[],'size':[]} for case in curves}
            for begin in range(0,c['bootstrap'],100):
                count=min(100,c['bootstrap']-begin);ii=rng.integers(n,size=(count,n));w=np.zeros((count,n));np.add.at(w,(np.arange(count)[:,None],ii),1)
                for case,xs in curves.items():
                    boot[case].extend(np.mean([curve(w[:,mapping]) for curve,mapping in xs],axis=0))
                    for metric,col in [('coverage',1),('size',2)]:actual[case][metric].extend(np.mean([(w[:,x[0]]@x[col])/w[:,x[0]].sum(1) for x in ordinary[case]],axis=0))
            for a,b in contrasts:
                dd=np.asarray(boot[a])-np.asarray(boot[b]);lo,hi=np.quantile(dd,[.025,.975]);mc_pairs.append({'a':a,'b':b,'split':split,'qtype':q,'metric':'matched_size','coverage':.9,'delta':sizes[a]-sizes[b],'ci_low':float(lo),'ci_high':float(hi),'bootstrap':c['bootstrap'],'threshold_reestimated_each_draw':True})
                for metric in ['coverage','size']:
                    dd=np.asarray(actual[a][metric])-np.asarray(actual[b][metric]);lo,hi=np.quantile(dd,[.025,.975]);point=np.mean([x[1 if metric=='coverage' else 2].mean() for x in ordinary[a]])-np.mean([x[1 if metric=='coverage' else 2].mean() for x in ordinary[b]]);actual_cp_pairs.append({'a':a,'b':b,'split':split,'qtype':q,'metric':metric,'delta':float(point),'ci_low':float(lo),'ci_high':float(hi),'alpha':.1,'bootstrap':c['bootstrap']})
            pd.DataFrame(matched).to_csv(out/'A_matched_coverage.csv',index=False);pd.DataFrame(mc_pairs).to_csv(out/'A_matched_coverage_paired.csv',index=False);pd.DataFrame(actual_cp_pairs).to_csv(out/'A_conformal_paired.csv',index=False);print(split,q,'matched coverage done',flush=True)
    fractions=[]
    for split in ['reader_val','test']:
        for q in ['Q3','Q4']:
            d={f"{r['a']}-{r['b']}":r['delta'] for r in pairs if r['split']==split and r['qtype']==q};gap=d['CLR-lite-B3']
            fractions.append({'split':split,'qtype':q,'original_gap':gap,'prefix_effect_in_B3':d['B3-prefix-B3'],'prefix_effect_in_CLR':d['CLR-lite-CLR-lite-single'],'residual_prefix_access_matched':d['CLR-lite-B3-prefix'],'residual_current_only':d['CLR-lite-single-B3'],'fraction_prefix_access_matched':d['CLR-lite-B3-prefix']/gap,'fraction_current_only':d['CLR-lite-single-B3']/gap,'interpretation':'two controlled contrasts, not additive causal percentages; B3 prefix is mean summary, CLR ordered sequence'})
    pd.DataFrame(fractions).to_csv(out/'A_gap_decomposition.csv',index=False);write_json(out/'A_analysis.json',{'fractions':fractions,'cases':list(curves),'bootstrap':c['bootstrap'],'matched_coverage':'exact empirical 90% with randomized ties, threshold reestimated per bootstrap; descriptive not deployment/conformal guarantee','test_scope':'pre-specified A only; invalid early exposure retained separately'})
    run.write_metrics(status='completed',gpu_hours=0,wall_seconds=time.monotonic()-start,source_forward_calls=0,test_opened=True,test_scope='A analysis');write_json(root/'A_analysis_run.json',{'run':str(run.path)})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

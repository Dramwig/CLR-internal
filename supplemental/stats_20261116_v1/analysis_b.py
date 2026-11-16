"""Final-only three-seed LLaMA paired analysis; no model calls."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np,pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.calib.m4_evaluation import unique_indices
from supplemental.stats_20261116_v1.analysis_a import Inputs
from supplemental.stats_20261116_v1.statistics import linear,paired,auc,direction

def main(c):
    root=Path(c['output']);assert json.loads((root/'b_completed.json').read_text())['status']=='completed'
    inp=Inputs(c);mf=inp.csv('data/M10_v1/paper_inputs/main_problem_metrics.csv');mf=mf[(mf.source=='llama')&mf.model.isin(['CLR-lite','CLR-lite-single','CLR-per-type'])]
    cpf=inp.csv('data/M10_v1/paper_inputs/conformal_problem.csv');cpf=cpf[(cpf.source=='llama')&cpf.model.isin(['CLR-lite','CLR-lite-single','CLR-per-type'])&cpf.qtype.isin(['Q3','Q4'])]
    af=pd.concat([inp.cache('llama',m,'aucs') for m in ['CLR-lite','CLR-lite-single','CLR-per-type']],ignore_index=True)
    points=[];aucs=[];cps=[];q1extra=[]
    for ab in ['A1','A2','A4']:
        for seed in [11,22,33]:
            ref=json.loads(inp.remember(root/f'evaluated_{ab}_{seed}.json').read_text());r=dict(source='llama',model=ab,seed=seed,run=ref['run'],native=False)
            for split in ['reader_val','test']:
                for q in ['Q1','Q2','Q3','Q4']:
                    raw,p,y,f=inp.ex.predict_rows(r,q,split);inp.used.update(inp.ex.used);common=dict(source='llama',model=ab,seed=seed,split=split,qtype=q)
                    g=f.groupby(['problem_id','t'])[['accuracy','nll']].mean();g['instances']=f.groupby(['problem_id','t']).size();points.append(g.reset_index().assign(**common))
                    if q=='Q2':aucs.append(f.assign(**common))
                    if q=='Q1' and split=='reader_val':
                        rr,pp,yy,ff=inp.ex.predict_rows(r,q,split,'traj');gg=ff.groupby(['problem_id','t'])[['nll','accuracy']].mean();gg['instances']=ff.groupby(['problem_id','t']).size();q1extra.append(gg.reset_index().assign(**common,kind='traj'));q1extra.append(g.reset_index().assign(**common,kind='clean'))
                    if q not in ['Q3','Q4']:continue
                    normalized=[{**a,'state_ref':{'t':a['t'],'traj_id':a['traj_id'],'kind':a['kind']}} for a in raw]
                    stored=json.loads(inp.remember(Path(r['run'])/'conformal.json').read_text())
                    for co in stored:
                        if co['split']!=split or co['qtype']!=q or co['heldout'] or co['alpha']!=.1 or co.get('calibration','temperature')!='temperature':continue
                        t=co['t'];thr=np.inf if co['threshold'] is None else co['threshold'];ii=unique_indices(normalized,[i for i,a in enumerate(normalized) if a['t']==t],q,seed);sets=1-p[ii]<=thr
                        assert abs(sets.sum(1).mean()-co['size'])<1e-7
                        assert abs((sets*y[ii]).sum(1).mean()-co['coverage'])<1e-7
                        for i,ss in zip(ii,sets):cps.append(dict(**common,problem_id=raw[i]['problem_id'],t=t,instances=1,size=float(ss.sum()),coverage=float((ss*y[i]).sum())))
            print('Collected B',ab,seed,flush=True)
    pm=pd.concat([mf,*points],ignore_index=True);af=pd.concat([af,*aucs],ignore_index=True);cp=pd.concat([cpf,pd.DataFrame(cps)],ignore_index=True)
    pm.to_parquet(root/'b_problem_metrics.parquet',index=False);af.to_parquet(root/'b_auroc_observations.parquet',index=False);cp.to_parquet(root/'b_conformal_observations.parquet',index=False)
    result=[]
    for split in ['reader_val','test']:
        for model in ['CLR-lite','A1','A2','A4','CLR-per-type','CLR-lite-single']:
            for q,metric in [('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy'),('Q3','size')]:
                f=af if metric=='auroc' else cp if metric=='size' else pm
                a=f[(f.model==model)&(f.split==split)&(f.qtype==q)];b=f[(f.model=='CLR-lite')&(f.split==split)&(f.qtype==q)]
                r=auc(a,b) if metric=='auroc' else paired(a,b,metric)
                result.append(dict(source='llama',variant=model,split=split,qtype=q,metric=metric,**r,classification=direction(r['delta'],metric),reused=model in ['CLR-lite','CLR-per-type','CLR-lite-single']))
                write_json(root/'b_results.json',result)
            print('Summarized B',split,model,flush=True)
    qq=pd.concat(q1extra,ignore_index=True);out=[]
    for key,g in qq.groupby(['model','kind']):out.append(dict(zip(['model','kind'],key),**linear(g,['nll'])[0]))
    write_json(root/'b_reader_val_q1_clean_traj.json',out);inp.used.update(inp.ex.used);write_json(root/'input_sha256_b.json',inp.used)
    write_json(root/'b_analysis_completed.json',{'status':'completed','rows':len(result),'gpu_hours':0})

if __name__=='__main__':main(load_config(sys.argv[1]))

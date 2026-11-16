"""Read-only A1-A4 from frozen predictions and hash-validated derivations."""
import sys
sys.dont_write_bytecode=True
import json,pickle
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from scipy.special import softmax
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from scripts.m9_export import Export
from supplemental.stats_20261116_v1.runtime import digest
from supplemental.stats_20261116_v1.statistics import linear,paired,auc,direction

class Inputs:
    def __init__(self,c):self.root=Path(c['output']);self.used={};self.ex=Export(c)
    def remember(self,p):
        p=Path(p)
        if str(p) not in self.used:self.used[str(p)]=digest(p)
        return p
    def csv(self,p):return pd.read_csv(self.remember(p))
    def pq(self,p):return pq.read_table(self.remember(p)).to_pandas()
    def cache(self,source,model,part):
        out=[]
        for seed in [11,22,33]:
            p=self.remember(f'data/M10_v1/paper_cache/{source}_{model}_{seed}.pkl')
            with p.open('rb') as f:d=pickle.load(f)
            for name,h in d['sources'].items():
                self.remember(name)
                if self.used[name]!=h:raise ValueError('Cached prediction input SHA mismatch '+name)
            out.extend(d[part])
        return pd.concat(out,ignore_index=True)
    def predict(self,r,q,split):
        raw,p,y,f=self.ex.predict_rows(r,q,split)
        self.used.update(self.ex.used)
        f=f.assign(seed=r['seed'],source=r['source'],model=r['model'],split=split,qtype=q)
        g=f.groupby(['seed','source','model','split','qtype','problem_id','t'])[['nll','accuracy']].mean();g['instances']=f.groupby(['seed','source','model','split','qtype','problem_id','t']).size()
        return g.reset_index(),f

def main(c):
    root=Path(c['output']);assert json.loads((root/'reproduction_passed.json').read_text())['passed']
    inp=Inputs(c);mf=inp.csv('data/M10_v1/paper_inputs/main_problem_metrics.csv');main=inp.csv('data/M10_v1/paper_inputs/main.csv')
    pm=inp.pq('data/M12_ablation/problem_metrics.parquet');af=inp.pq('data/M12_ablation/auroc_observations.parquet');cp=inp.pq('data/M12_ablation/conformal_observations.parquet')
    # Validate original prediction/calibration hashes behind these frozen derivations.
    prov=json.loads(inp.remember('data/M12_ablation/statistics_provenance.json').read_text())
    for p,h in prov['source_sha256'].items():
        inp.remember(p)
        if inp.used[p]!=h:raise ValueError('Frozen M12 input mismatch '+p)
    variants=[] if (root/'a1.json').exists() and len(json.loads((root/'a1.json').read_text()))==45 else ['CLR-lite','A1','A2','A3','A4','CLR-v1','CLR-lite-single','A5','CLR-per-type']
    a1=[]
    for v in variants:
        for q,metric in [('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy'),('Q3','size')]:
            f=af if q=='Q2' else cp if metric=='size' else pm
            a=f[(f.variant==v)&(f.split=='test')&(f.qtype==q)];b=f[(f.variant=='CLR-lite')&(f.split=='test')&(f.qtype==q)]
            result=auc(a,b) if metric=='auroc' else paired(a,b,metric)
            row=dict(variant=v,qtype=q,metric=metric,**result,classification=direction(result['delta'],metric));a1.append(row)
            write_json(root/'a1.json',a1);print('A1',v,q,metric,flush=True)
    # Paired accuracy/set-size equivalence uses exactly the original selected queries.
    vp=inp.csv('data/M10b_v1/paper_inputs/B5c_problem_metrics.csv');vc=inp.csv('data/M10b_v1/paper_inputs/B5c_conformal_problem.csv')
    a2=[]
    for q in ['Q3','Q4']:
        for metric in ['accuracy','size']:
            a=(pm if metric=='accuracy' else cp);a=a[(a.variant=='CLR-lite')&(a.split=='test')&(a.qtype==q)]
            b=(vp if metric=='accuracy' else vc);b=b[(b.split=='test')&(b.qtype==q)]
            r=paired(a,b,metric);scale=100 if metric=='accuracy' else 1.;tol=c['equivalence_accuracy_pp'] if metric=='accuracy' else c['equivalence_set_size']
            d={k:scale*v for k,v in r['delta'].items()};r['delta']=d
            a2.append(dict(qtype=q,metric=metric,**r,tolerance=tol,equivalent=bool(d['lo90']>-tol and d['hi90']<tol),minimum_margin_infimum=max(abs(d['lo90']),abs(d['hi90']))))
    write_json(root/'a2.json',a2)
    # Validation panel for the three CLR designs, with original calibration retained.
    a3=[];conformal=inp.csv('data/M10_v1/paper_inputs/conformal.csv')
    for s in ['gpt2','llama']:
        for model in ['CLR-lite-single','CLR-lite','CLR-lite-multi']:
            row={'source':s,'model':model}
            for q,metric in [('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy')]:
                f=main[(main.source==s)&(main.model==model)&(main.split=='reader_val')&(main.qtype==q)&(main.metric==metric)];assert len(f)==1
                row[q]={k:float(f.iloc[0][k]) for k in ['mean','lo','hi']}
            f=conformal[(conformal.source==s)&(conformal.model==model)&(conformal.split=='reader_val')&(conformal.qtype=='Q3')&(conformal.metric=='size')];assert len(f)==1
            row['size']={k:float(f.iloc[0][k]) for k in ['mean','lo','hi']};a3.append(row)
    write_json(root/'a3_validation.json',a3)
    # Every requested Q1/Q2 specialist is included; select observed strongest per metric.
    a4=json.loads((root/'a4.json').read_text()) if (root/'a4.json').exists() else []
    m41=json.loads(inp.remember('data/M4_v0/evaluation_suite.json').read_text())
    q1diag=[]
    for s in ['gpt2','llama']:
        clf=mf[(mf.source==s)&(mf.model=='CLR-lite')&(mf.split=='test')&(mf.qtype=='Q1')]
        clauc=inp.cache(s,'CLR-lite','aucs');clauc=clauc[clauc.split=='test']
        for model in ['B3','B3-prefix','B6','B4','HSRM','CLUE']:
            if sum(r['source']==s and r['model']==model for r in a4)==2:continue
            if model in ['B3','B3-prefix','B6']:
                f=mf[(mf.source==s)&(mf.model==model)&(mf.split=='test')&(mf.qtype=='Q1')]
                ac=inp.cache(s,model,'aucs');ac=ac[ac.split=='test']
            elif model in ['HSRM','CLUE']:
                f=inp.csv(f'data/M10e_v1/problem_metrics_{s}_{model}.csv');f=f[(f.split=='test')&(f.kind=='clean')&(f.qtype=='Q1')]
                ref=json.loads(inp.remember(f'data/M10e_v1/final_{s}_{model}.json').read_text());acs=[]
                for p in sorted(Path(ref['run']).glob(f'{model}_Q2_*_test.parquet')):
                    raw=pq.read_table(inp.remember(p)).to_pylist();raw=[r for r in raw if r['state_ref']['kind']=='clean'];pp=softmax(np.array([r['calibrated_logits'] for r in raw],float),axis=1);seed=0 if '_None_' in p.name else int(p.stem.split('_')[2]);acs.append(pd.DataFrame({'seed':seed,'problem_id':[r['problem_id'] for r in raw],'t':[r['state_ref']['t'] for r in raw],'score':pp[:,1],'label':[r['label'] for r in raw]}))
                ac=pd.concat(acs,ignore_index=True)
            else:
                if s=='gpt2':
                    fs=[];acs=[]
                    for seed in [11,22,33]:
                        r=dict(source=s,model=model,seed=seed,run=m41[f'B4_{seed}'],native=False)
                        ff,_=inp.predict(r,'Q1','test');_,aa=inp.predict(r,'Q2','test');fs.append(ff);acs.append(aa)
                    f=pd.concat(fs,ignore_index=True);ac=pd.concat(acs,ignore_index=True)
                else:
                    f=inp.csv('data/M10d_v1/problem_metrics.csv');f=f[(f.split=='test')&(f.qtype=='Q1')&(f.kind=='clean')]
                    ref=json.loads(inp.remember('data/M10d_v1/final.json').read_text());acs=[]
                    for p in sorted(Path(ref['run']).glob('*Q2*test.parquet')):
                        raw=pq.read_table(inp.remember(p)).to_pylist();raw=[r for r in raw if r['state_ref']['kind']=='clean'];pp=softmax(np.array([np.array(r['logits'],float)/r['temperature'] for r in raw]),axis=1);seed=int(p.stem.split('_')[2]);acs.append(pd.DataFrame({'seed':seed,'problem_id':[r['problem_id'] for r in raw],'t':[r['state_ref']['t'] for r in raw],'score':pp[:,1],'label':[r['label'] for r in raw]}))
                    ac=pd.concat(acs,ignore_index=True)
            for q,metric,r in [('Q1','nll',paired(clf,f,'nll')),('Q2','auroc',auc(clauc,ac))]:
                a4.append(dict(source=s,model=model,qtype=q,metric=metric,**r,classification=direction(r['delta'],metric)))
            write_json(root/'a4.json',a4);print('A4',s,model,flush=True)
    write_json(root/'input_sha256_a.json',inp.used)
    write_json(root/'a_completed.json',{'status':'completed','gpu_hours':0,'new_test_inference':False})

if __name__=='__main__':main(load_config(sys.argv[1]))

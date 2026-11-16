"""Compare every supplied Table 1/2 value to frozen prediction-derived cells."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from supplemental.stats_20261116_v1.analysis_a import Inputs

EXPECTED={
 'B0':[[.696,.508,35.9,49.8],[.692,.509,39.1,50.1]],
 'B2':[[.553,.836,53.4,50.1],[.615,.764,54.,50.1]],
 'B3':[[.523,.846,58.6,68.3],[.612,.851,58.9,62.7]],
 'B3-prefix':[[.515,.857,61.8,68.4],[.606,.852,59.3,60.3]],
 'B6':[[.574,.853,65.4,72.],[.632,.824,57.8,66.6]],
 'B4':[[.507,.845,None,None],[.551,.806,None,None]],
 'HSRM':[[.559,.834,None,None],[.625,.834,None,None]],
 'CLUE':[[.648,.575,None,None],[.545,.793,None,None]],
 'B5c':[[None,None,76.8,76.3],[None]*4],
 'CLR-lite-single':[[.564,.819,73.8,74.8],[.627,.813,74.7,75.5]],
 'CLR-lite':[[.548,.830,76.1,76.2],[.615,.821,72.1,74.2]],
 'CLR-lite-multi':[[.558,.831,76.2,76.4],[.622,.798,75.,76.5]],
}
T2={'CLR-lite':[.548,.830,76.1,76.2,1.49],'A1':[.568,.822,75.1,75.4,1.54],'A2':[.584,.819,72.9,74.3,1.58],'A3':[.549,.834,75.,75.4,1.49],'A4':[.557,.838,74.,76.7,1.54],'CLR-v1':[.564,.830,76.2,75.,1.52],'CLR-lite-single':[.564,.819,73.8,74.8,1.61],'A5':[.575,.829,75.9,76.6,1.51],'CLR-per-type':[.557,.823,76.7,73.7,1.48]}

def main(c):
    inp=Inputs(c);root=Path(c['output']);main=inp.csv('data/M10_v1/paper_inputs/main.csv');e=inp.csv('data/M10e_v1/summary.csv');b4=inp.csv('data/M10d_v1/B4_summary.csv');vp=inp.csv('data/M10b_v1/paper_inputs/B5c_summary.csv');out=[]
    suite=json.loads(inp.remember('data/M4_v0/evaluation_suite.json').read_text())
    for model,values in EXPECTED.items():
        for i,s in enumerate(['gpt2','llama']):
            for j,(q,metric) in enumerate([('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy')]):
                expected=values[i][j]
                if expected is None:continue
                if model in ['B0','B3','B3-prefix','B6','CLR-lite-single','CLR-lite','CLR-lite-multi']:f=main
                elif model in ['HSRM','CLUE'] or (model=='B2' and s=='llama'):f=e
                elif model=='B5c':f=vp
                elif model=='B4' and s=='llama':f=b4
                else:
                    vals=[]
                    for seed in [11,22,33]:
                        r=dict(source=s,model=model,seed=seed,run=suite[f'{model}_{seed}'] if model!='B2' else 'runs/20261103T045628621237Z_2964c31f2153_11a348',native=False)
                        # Original saved per-step metrics; no bootstrap needed for a display audit.
                        rs=json.loads(inp.remember(Path(r['run'])/'statistics.json').read_text())
                        use=[x.get('calibrated',x)[metric] for x in rs if x['split']=='test' and x['qtype']==q and not x['heldout'] and x.get('kind','clean')=='clean' and x.get('calibration','temperature')=='temperature' and x.get('subset','all')=='all' and (model!='B2' or x.get('baseline')=='B2' and x.get('seed')==seed)]
                        if not use:raise ValueError((model,q,list(rs[0])))
                        vals.append(sum(use)/len(use))
                    actual=sum(vals)/len(vals);f=None
                if f is not None:
                    ff=f[(f.source==s)&(f.model==model)&(f.split=='test')&(f.qtype==q)&(f.metric==metric)]
                    if 'kind' in ff:ff=ff[ff.kind=='clean']
                    assert len(ff)==1,(s,model,q,len(ff));actual=float(ff.iloc[0]['mean'])
                val=actual*(100 if j>=2 else 1);precision=1 if j>=2 else 3
                out.append(dict(table=1,source=s,model=model,qtype=q,actual=val,expected=expected,display_match=round(val,precision)==expected))
    t=inp.csv('data/M12_ablation/summary.csv')
    for v,values in T2.items():
        for j,(q,metric) in enumerate([('Q1','nll'),('Q2','auroc'),('Q3','accuracy'),('Q4','accuracy'),('Q3','size')]):
            r=t[(t.variant==v)&(t.split=='test')&(t.qtype==q)].iloc[0];actual=float(r['conformal_size'] if metric=='size' else r['mean']);val=actual*(100 if metric=='accuracy' else 1);precision=1 if metric=='accuracy' else 2 if metric=='size' else 3
            out.append(dict(table=2,source='gpt2',model=v,qtype=q,metric=metric,actual=val,expected=values[j],display_match=round(val,precision)==values[j]))
    pd.DataFrame(out).to_csv(root/'table_number_audit.csv',index=False)
    write_json(root/'table_number_audit.json',{'cells':len(out),'differences':[r for r in out if not r['display_match']],'input_sha256':inp.used})
    # Original matched-exposure per-type receipt, not an inference from test scores.
    checks=[];target=json.loads(inp.remember('data/M10_v1/exposure_targets.json').read_text())['llama']
    for seed in [11,22,33]:
        ref=json.loads(inp.remember(f'data/M10_v1/trained/llama_CLR-per-type_{seed}_full.json').read_text());tr=json.loads(inp.remember(ref['training_manifest']).read_text())
        for q,x in tr['exposure'].items():
            assert x['selected']['epoch']==8
            assert x['total_by_type'][q]==target[str(seed)]['exposure'][q]
            p=ref['checkpoints'][q]['path'];assert inp.remember(p) and inp.used[p]==ref['checkpoints'][q]['sha256']
            checks.append(dict(seed=seed,qtype=q,exposure=x['total_by_type'][q],epochs=8,passed=True))
    write_json(root/'b_reused_per_type_audit.json',{'passed':True,'checks':checks,'input_sha256':inp.used})
    print('Number audit',len(out),'differences',[r for r in out if not r['display_match']],flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]))

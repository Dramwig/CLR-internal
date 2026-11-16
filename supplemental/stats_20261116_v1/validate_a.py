"""Independent checks against frozen CI/point exports, including nonlinear AUROC."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import pandas as pd
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    root=Path(c['output']);a=json.loads((root/'a1.json').read_text());f=pd.read_csv('data/M12_ablation/summary.csv');errors=[]
    for r in a:
        old=f[(f.variant==r['variant'])&(f.split=='test')&(f.qtype==r['qtype'])].iloc[0]
        fields=['conformal_size','conformal_size_paired_delta','conformal_size_paired_ci_low','conformal_size_paired_ci_high'] if r['metric']=='size' else ['mean','paired_delta','paired_ci_low','paired_ci_high']
        values=[r['value']['mean'],r['delta']['mean'],r['delta']['lo'],r['delta']['hi']]
        error=max(abs(float(old[k])-v) for k,v in zip(fields,values));assert error<1e-10,(r['variant'],r['metric'],error);errors.append(error)
    main=pd.read_csv('data/M10_v1/paper_inputs/main.csv');special=pd.read_csv('data/M10e_v1/summary.csv');b4=pd.read_csv('data/M10d_v1/B4_summary.csv')
    for r in json.loads((root/'a4.json').read_text()):
        if r['model'] in ['B3','B3-prefix','B6']:source=main
        elif r['model'] in ['HSRM','CLUE']:source=special
        elif r['source']=='llama':source=b4
        else:continue
        s=source[(source.source==r['source'])&(source.model==r['model'])&(source.split=='test')&(source.qtype==r['qtype'])&(source.metric==r['metric'])]
        if 'kind' in s:s=s[s.kind=='clean']
        assert len(s)==1
        error=abs(float(s.iloc[0]['mean'])-r['baseline']['mean']);assert error<1e-10,(r,error);errors.append(error)
    cp=pd.read_csv('data/M10c_v1/paper_inputs/conformal_paired.csv')
    for r in json.loads((root/'a2.json').read_text()):
        if r['metric']!='size':continue
        x=cp[(cp.split=='test')&(cp.qtype==r['qtype'])&(cp.metric=='size_difference')].iloc[0]
        assert abs(r['delta']['mean']+x['mean'])<1e-12
        assert abs(r['delta']['lo']+x['hi'])<1e-12
        assert abs(r['delta']['hi']+x['lo'])<1e-12
    audit=json.loads((root/'table_number_audit.json').read_text());assert audit['cells']==123 and not audit['differences']
    write_json(root/'a_validation.json',{'passed':True,'checked_M12_intervals':len(a),'checked_point_interval_numbers':len(errors),'maximum_absolute_error':max(errors),'conformal_sign_replay':True,'paper_cells_matched':123})
    print((root/'a_validation.json').read_text(),flush=True)

if __name__=='__main__':main(load_config(sys.argv[1]))

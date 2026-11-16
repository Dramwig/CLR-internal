"""Validate all reused schemas and conformal identities using reader_val only."""
import argparse,json
from pathlib import Path
import numpy as np
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from scripts.m12_summarize import registry
from scripts.m9_export import Export
from clr.calib.m4_evaluation import unique_indices

def main(c):
 with recorded({**c,'mode':'source_val_smoke','stage':'CPU_source_schema'}) as run:
    ex=Export(c);checks=[]
    for r in registry({**c,'mode':'source_val_smoke'}):
        if r['seed']!=11:continue
        for q in ['Q1','Q2','Q3','Q4']:
            raw,p,y,f=ex.predict_rows(r,q,'reader_val')
            if not len(f) or not np.isfinite(p).all():raise ValueError('Invalid source schema')
            if q in ['Q3','Q4']:
                normalized=raw if r['native'] else ex.metadata_rows(raw,q,'reader_val')
                for cp in ex.js(Path(r['run'])/'conformal.json'):
                    if cp['split']!='reader_val' or cp['qtype']!=q or cp['heldout'] or cp['alpha']!=.1 or cp.get('calibration','temperature')!='temperature' or cp.get('subset','all')!='all':continue
                    t=cp['t'];jj=unique_indices(normalized,[i for i,a in enumerate(normalized) if a['state_ref']['t']==t],q,r['seed']);threshold=cp['threshold'] if cp['threshold'] is not None else np.inf;size=(1-p[jj]<=threshold).sum(1).mean()
                    if abs(size-cp['size'])>1e-7:raise ValueError('Reused conformal replay differs')
            checks.append({'variant':r['variant'],'qtype':q,'rows':len(f),'schema_ok':True});print(checks[-1],flush=True)
    write_json(run.path/'checks.json',checks);run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False)
    write_json(Path(c['output'])/'source_val_smoke.json',{'run':str(run.path),'passed':True,'checks':len(checks),'test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

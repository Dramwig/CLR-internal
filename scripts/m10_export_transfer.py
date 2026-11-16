"""Learning curves and paired initialization differences; no model calls."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from scripts.m10_export_audit import group_summary
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 root=Path(c['output']);ref=json.loads((root/'evaluated/transfer_full.json').read_text());run=Path(ref['run']);train=json.loads(Path(ref['training']).read_text());sources={};allframes=[];result=[];paired=[]
 for key,cp in train['checkpoints'].items():
  sel=cp['selected'];n=sel['N'];nvalue=192000 if n is None else n
  for split in ['reader_val','test']:
   for task in ['A1','A4']:
    path=run/f'{key}_{split}_{task}.parquet';sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();f=pq.read_table(path).to_pandas();f=f.assign(seed=sel['seed'],initialization=sel['initialization'],N='full' if n is None else str(n),N_value=nvalue,split=split,task=task);allframes.append(f)
 frame=pd.concat(allframes,ignore_index=True)
 for keys,f in frame.groupby(['initialization','N','N_value','split','task']):
  common=dict(zip(['initialization','N','N_value','split','task'],keys))
  for metric in ['accuracy','nll','coverage','set_size']:result.append({**common,'metric':metric,**group_summary(f,metric)})
 for keys,f in frame.groupby(['N','N_value','split','task']):
  # Aggregate within group first, then pair exactly the same trained seed.
  a=f[f.initialization=='M9'].groupby(['group_id','seed']).accuracy.mean();b=f[f.initialization=='scratch'].groupby(['group_id','seed']).accuracy.mean();d=(a-b).rename('delta').reset_index();paired.append(dict(zip(['N','N_value','split','task'],keys),**group_summary(d,'delta')))
 dest=root/'paper_inputs';pd.DataFrame(result).to_csv(dest/'cross_use_transfer.csv',index=False,float_format='%.15g');pd.DataFrame(paired).to_csv(dest/'cross_use_transfer_paired.csv',index=False,float_format='%.15g');write_json(root/'transfer_export_sources.json',{'sha256':sources})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

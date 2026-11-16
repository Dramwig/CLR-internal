"""CPU-only M10 GSM summaries, preserving the frozen M9 estimands/caches."""
import argparse,json,shutil
from pathlib import Path
import pandas as pd
from scripts.m9_export import Export
from clr.paper.statistics import metric_summary
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

class M10Export(Export):
 def registry(self,include_new):
  root=self.root;self.root=Path('data/M9_v1')
  try:old=super().registry(True)
  finally:self.root=root
  added=[]
  for p in sorted(Path(self.c['output'],'evaluated').glob('*_full.json')):
   r=self.js(p)
   if 'training' not in r or not str(r['training']).endswith('_full.json') or 'transfer' in p.name:continue
   t=self.js(r['training']);c=t['config']
   if 'source' not in c or 'kind' not in c:continue
   added.append(dict(source=c['source'],model=c['kind'],seed=c['seed'],run=r['run'],native=True))
  return old+added
 def registry_old(self):return super().registry(True)

def main(c):
 dest=Path(c['output'])/'paper_cache';dest.mkdir(exist_ok=True,parents=True)
 # Copy the frozen serialized derivation caches; each source SHA is verified
 # again by Export before reuse. This invokes no model or new experiment.
 for p in Path('data/M9_v1/paper_cache').glob('*.pkl'):
  q=dest/p.name
  if not q.exists():shutil.copyfile(p,q)
 ex=M10Export(c);ex.main_tables(True)
 frame=pd.read_csv(ex.dest/'main_problem_metrics.csv');paired=[]
 for source in ['gpt2','llama']:
  for split in ['reader_val','test']:
   for qtype in ['Q1','Q2','Q3','Q4']:
    f=frame[(frame.source==source)&(frame.split==split)&(frame.qtype==qtype)]
    for model,baseline in [('CLR-lite','B6'),('CLR-lite','B3-prefix'),('CLR-lite','CLR-lite-single'),('CLR-lite','CLR-per-type')]:
     a=f[f.model==model];b=f[f.model==baseline]
     if a.empty or b.empty:continue
     d=a.merge(b,on=['problem_id','t','seed'],suffixes=('_a','_b'),validate='one_to_one');d['instances']=d.instances_a
     for metric in ['accuracy','nll']:
      d['delta']=d[metric+'_a']-d[metric+'_b'];e=metric_summary(d,['delta'])[0];e['metric']=metric+'_difference';paired.append(dict(source=source,split=split,qtype=qtype,model=model,baseline=baseline,**e))
 ex.save('M10_gsm_paired',paired);write_json(ex.root/'gsm_export_sources.json',{'sha256':ex.used})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

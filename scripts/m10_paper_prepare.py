"""Freeze M10 derived CSVs and generate complete paper build configurations."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import yaml
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m10_runtime import ledger

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(c):
 root=Path(c['output']);dest=root/'paper_inputs';dest.mkdir(exist_ok=True);f=pd.read_csv('data/M9_v1/paper_inputs/costs.csv');costs=ledger();f=pd.concat([f,pd.DataFrame([dict(milestone='M10',gpu_hours=sum(r['gpu_hours'] for r in costs),source_forward_calls=sum(r['source_forward_calls'] or 0 for r in costs),missing_source_counts=sum(r['source_forward_calls'] is None for r in costs),cost_scope='closed M10 resident wall ledger incl smoke, failures, restarts and final inference',evidence='data/M10_v1/cost_ledger.json')])],ignore_index=True);f.to_csv(dest/'costs.csv',index=False,float_format='%.15g');write_json(root/'cost_ledger.json',costs)
 sources={}
 for p in root.glob('*export_sources.json'):sources.update(json.loads(p.read_text())['sha256'])
 write_json(dest/'provenance.json',{'source_sha256':sources,'protocol':'CPU-only exports of final-locked predictions; 10000 group bootstrap; fixed trained seeds'})
 freeze=dest/'freeze.json'
 if freeze.exists():raise FileExistsError('Paper inputs already frozen')
 write_json(freeze,{'version':'M10_paper_inputs_v1','sha256':{str(p):sha(p) for p in sorted(dest.glob('*.csv'))+[dest/'provenance.json']}})
 old=yaml.safe_load(Path('configs/M9/paper.yaml').read_text());artifacts=[];folder=Path('configs/M10/paper');folder.mkdir(exist_ok=True)
 replacements={'t1_gsm8k':'m10_t1_gsm8k','t1_reader_val':'m10_t1_reader_val','t4_audit':'m10_t4_audit','t4_qa':'m10_t4_qa'}
 for item in old['artifacts']:
  cfg=yaml.safe_load(Path(item['config']).read_text());name=cfg['artifact'];script=item['script']
  if name in replacements:
   script=f'scripts/paper/make_{replacements[name]}.py';cfg['input_freeze']=str(freeze);cfg['inputs']={k:str(dest/Path(p).name) for k,p in cfg['inputs'].items()}
   # Localization is unchanged frozen evidence, so T4 no longer lists unused inputs.
   if name=='t4_audit':cfg['inputs']={'detection':str(dest/'communication_detection.csv')}
  elif name in ['t2_conformal','t2_paired_sizes','appendix_q1_clean_traj','t5_cost']:
   cfg['input_freeze']=str(freeze);cfg['inputs']={k:str(dest/Path(p).name) for k,p in cfg['inputs'].items()}
  path=folder/(name+'.yaml');path.write_text(yaml.safe_dump(cfg,sort_keys=False));artifacts.append({'script':script,'config':str(path)})
 for name,builder,input_key,file,caption in [('t4_hiddenbench','t4_hiddenbench','hiddenbench','hiddenbench.csv','T4 supplement: exploratory HiddenBench mismatch audit'),('appendix_per_type','appendix_per_type','main','main.csv','Separate-type versus multitype CLR with matched exposure'),('f4_cross_use_transfer','f4_cross_use_transfer','transfer','cross_use_transfer.csv','M9 inference-body initialization for communication auditing')]:
  cfg=dict(artifact=name,caption=caption,input_freeze=str(freeze),inputs={input_key:str(dest/file)},output_root='paper');p=folder/(name+'.yaml');p.write_text(yaml.safe_dump(cfg,sort_keys=False));artifacts.append(dict(script=f'scripts/paper/make_{builder}.py',config=str(p)))
 for key,output in [('paper_build.yaml','paper'),('paper_repro.yaml',str(root/'repro_build'))]:
  (Path('configs/M10')/key).write_text(yaml.safe_dump(dict(milestone='M10',output_root=output,isolated=True,artifacts=artifacts),sort_keys=False))
 print(len(artifacts),'artifacts configured')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

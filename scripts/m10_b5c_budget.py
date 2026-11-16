"""Budget admission from val-only measured full-model training throughput."""
import argparse,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 targets=json.loads(Path(c['exposure_targets']).read_text())['gpt2']['11']['counts'];trials=[]
 for path in c['smoke_refs']:
  r=json.loads(Path(path).read_text());run=Path(r['run']);h=json.loads((run/'history.json').read_text())[0]
  rates={q:h['exposure'][q]/h['task_seconds'][q] for q in ['Q3','Q4']};hours=sum(targets[q]/rates[q] for q in rates)*3/3600
  trials.append({'ref':path,'run':str(run),'batch':r['config']['batch'],'samples_per_second':rates,'GPT2_one_epoch_three_seed_train_GPUh':hours})
 best=min(trials,key=lambda r:r['GPT2_one_epoch_three_seed_train_GPUh'])
 write_json(c['output'],{'passed':best['GPT2_one_epoch_three_seed_train_GPUh']<=c['section_budget'],'trials':trials,'fastest_measured_GPT2_train_GPUh':best['GPT2_one_epoch_three_seed_train_GPUh'],'section_budget_GPUh':c['section_budget'],'scope':'One pass over the same full Q3/Q4 training rows for three seeds; omits LLaMA, audit, calibration, final inference and loading, so these are additional costs. Estimate is conditional on measured hardware throughput, not a theoretical lower bound.','decision':'No full B5c run admitted; no unapproved reduction in data or seed count. No B5c test predictions.','test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

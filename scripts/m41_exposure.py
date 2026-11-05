"""Reconstruct exposure exactly from frozen configs/history (not test results)."""
import argparse,json,math
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
 with recorded(c) as run:
    suite=json.loads(Path(c['m4_suite']).read_text());selection=json.loads(Path(c['m3_baseline_run'],'selection.json').read_text());rows=[]
    for seed in c['seeds']:
        p=Path(suite[f'main_{seed}']);config=json.loads((p/'config.json').read_text());metrics=json.loads((p/'metrics.json').read_text());best=json.loads((p/'best.json').read_text());counts=json.loads((p/'training_manifest.json').read_text())['train_counts']
        per_epoch=sum(math.ceil(config['samples_per_type']/config['micro_batch']) for q in config['qtypes']);steps=math.ceil(per_epoch/config['gradient_accumulation'])
        for q in ['Q1','Q2','Q3','Q4']:
            b3=next(r for r in selection if r['baseline']=='B3' and r['qtype']==q and r['seed']==seed)
            rows.append({'seed':seed,'qtype':q,'dataset_rows':counts[q],'clr_best_examples':best['epoch']*config['samples_per_type'],'clr_total_examples':metrics['epochs']*config['samples_per_type'],'clr_joint_best_optimizer_steps':best['epoch']*steps,'clr_joint_total_optimizer_steps':metrics['epochs']*steps,'b3_best_examples':b3['epoch']*counts[q],'b3_best_optimizer_steps':b3['epoch']*math.ceil(counts[q]/4096),'b3_total_search_examples':16*counts[q],'b3_total_search_optimizer_steps':16*math.ceil(counts[q]/4096),'b3_selected_lr':b3['lr'],'b3_selected_epoch':b3['epoch']})
    write_json(run.path/'exposure.json',rows);write_json(Path(c['output'])/'exposure.json',rows)
    run.write_metrics(status='completed',gpu_hours=0.,source_forward_calls=0,test_opened=False,rows=len(rows));write_json(Path(c['output'])/'exposure_run.json',{'run':str(run.path)})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

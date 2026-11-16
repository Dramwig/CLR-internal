"""Typed QA on saved val features; no receiver/source rerun."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m8b_reader import SourceBank,new_model,rng_seed,make_row
from clr.audit.m71_reader import Serializer,Bank
from clr.audit.m8b_runtime import Cost,snapshot
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.config import load_config
from scripts.m8b_train import evaluate

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output']);serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True));results=[]
            for design in c['designs']:
                if design=='R1':continue
                bank=SourceBank(root/'source'/'reader_val',design)
                for seed in c['reader_seeds']:
                    ref=json.loads((root/c['training_folder']/f'{design}_{seed}.json').read_text());model=new_model(seed,design,c['device'],ref['checkpoint']).eval().requires_grad_(False)
                    if design=='R2':bank.refresh(rng_seed(seed,'QA'),c['device'])
                    rows=[]
                    for mid,entry in bank.entry.items():
                        v=bank.values[mid];rows.extend([make_row(entry,v,'A1',4,(v+j)%4) for j in range(4)]+[make_row(entry,v,'A4',4,None,rng_seed(mid,'options'))])
                    for task in ['A1','A4']:
                        rr=[r for r in rows if r['task']==task]
                        for start in range(0,len(rr),c['batch_size']):
                            cost.check();rs=rr[start:start+c['batch_size']]
                            with torch.autocast('cuda',dtype=torch.bfloat16):ls=model(serializer.batch(rs,c['device']),bank.batch(rs,c['device']),task)
                            for r,p in zip(rs,ls.argmax(-1).tolist()):
                                for repeat in range(3 if task=='A1' and r['label']==1 else 1):results.append({'group_id':r['group_id'],'design':design,'seed':seed,'task':task,'correct':p==r['label'],'split':'reader_val','category':'clean' if r['message_id']==r['episode_id'] else 'counterfactual'})
                    del model
                del bank
            write_json(root/'reader_val_QA200_allneg.json',{'run':str(run.path),'rows':results,'groups':200});cost.update(status='completed')
        except Exception:cost.update(status='failed');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

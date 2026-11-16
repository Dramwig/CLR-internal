"""Matched 200-group frozen-reader QA using existing source features."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from clr.audit.m71_reader import Bank,Serializer
from clr.audit.m8b_reader import new_model,make_row,rng_seed
from clr.audit.m8b_runtime import Cost,snapshot
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.config import load_config

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output']);groups={json.loads(l)['group_id'] for l in (root/'source/reader_val/manifest.jsonl').read_text().splitlines()};bank=Bank('data/M71_v1','reader_val',templates=(4,))
            rows=[]
            for e in bank.entries:
                if e['group_id'] not in groups:continue
                value=bank.source[e['id']]['e'];rows.extend([make_row(e,value,'A1',4,(value+j)%4) for j in range(4)]+[make_row(e,value,'A4',4,None,rng_seed(e['id'],'options'))])
            serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True));results=[]
            for seed in c['reader_seeds']:
                model=new_model(seed,'R1',c['device']).eval().requires_grad_(False)
                for task in ['A1','A4']:
                    rr=[r for r in rows if r['task']==task]
                    for start in range(0,len(rr),c['batch_size']):
                        cost.check();rs=rr[start:start+c['batch_size']]
                        with torch.autocast('cuda',dtype=torch.bfloat16):ls=model(serializer.batch(rs,c['device']),bank.batch(rs,c['device']),task)
                        for r,p in zip(rs,ls.argmax(-1).tolist()):
                            for repeat in range(3 if task=='A1' and r['label']==1 else 1):results.append({'group_id':r['group_id'],'design':'R1','seed':seed,'task':task,'correct':p==r['label'],'split':'reader_val','category':'clean'})
                del model
            write_json(root/'reader_val_QA200_R1_allneg.json',{'run':str(run.path),'rows':results,'groups':200});cost.update(status='completed')
        except Exception:cost.update(status='failed');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

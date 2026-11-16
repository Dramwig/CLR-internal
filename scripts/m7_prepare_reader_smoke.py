import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        out=Path(c['output'])
        sources={'reader_train':Path(c['train_source'])/'source/reader_train/manifest.jsonl',
                 'reader_val':Path(c['val_source'])/'source/reader_val/manifest.jsonl'}
        for split,p in sources.items():
            n=4 if split=='reader_train' else 2
            rows=p.read_text().splitlines()[:n]
            if len(rows)!=n:raise ValueError('Smoke source panel not yet available')
            d=out/'source'/split;d.mkdir(parents=True,exist_ok=False)
            (d/'manifest.jsonl').write_text('\n'.join(rows)+'\n')
            write_json(d/'completed.json',{'source':str(p),'episodes':n,'val_proxy':split=='reader_val'})
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

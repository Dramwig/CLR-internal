"""Audit exact serialized lengths on train/reader_val, never test."""
import argparse,json
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.models.clr_v0 import Serializer

def main(c):
 with recorded(c) as run:
    templates=load_config(c['templates']);s=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates);out={}
    for split in ['reader_train','reader_val']:
        maxima={};count=0
        file=pq.ParquetFile(Path(c['qa_root'])/f'{split}.parquet')
        for batch in file.iter_batches(batch_size=65536,columns=['qtype','template_id','options','is_heldout_template']):
            for r in batch.to_pylist():
                if r['is_heldout_template']:continue
                q=r['qtype'];text=templates[q][r['template_id']]
                if q=='Q4':text=text.format(v=r['options'][0])
                n=2+len(s.tokens(text))+sum(2+len(s.tokens(o)) for o in r['options']);count+=1
                if q not in maxima or n>maxima[q]['length']:
                    assert len(s.encode(r)['ids'])==n;maxima[q]={'length':n,'example':r}
        out[split]={'rows':count,'maxima':maxima};write_json(run.path/'lengths.json',out)
    maximum=max(v['length'] for data in out.values() for v in data['maxima'].values())
    write_json(Path(c['output'])/'length_audit.json',{'run':str(run.path),'max_length':maximum,'splits':out,'test_opened':False});run.write_metrics(status='completed',gpu_hours=0.,source_forward_calls=0,max_length=maximum,test_opened=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

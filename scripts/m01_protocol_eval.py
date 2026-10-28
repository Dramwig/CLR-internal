import argparse,json,time,signal,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.stats_v2 import bootstrap_mean
from clr.sources.codi_v2 import load
from clr.data.labels import numeric_answer
import torch

def deadline(*args):raise TimeoutError('M0.1 smoke exceeded 900s')
def main(c):
 with recorded(c) as run:
    t=time.perf_counter();source=None
    if c['mode']=='smoke':signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
    try:
        torch.set_num_threads(c['threads']);seed_everything(11)
        protocol=load_config(c['protocol']);source=load(protocol['source'])
        rows=json.loads(Path(c['data']).read_text());all_rows=[]
        for start in range(0,len(rows),128):
            cohort=rows[start:start+128]
            width=max(len(source.tokenizer.encode(r['question'].strip().replace('  ',' '))) for r in cohort)
            for row in cohort:all_rows.append({**row,'pad_to':width if c['padding']=='official_cohort' else len(source.tokenizer.encode(row['question'].strip().replace('  ',' '))),'cohort':start//128})
        if c['mode']=='smoke':
            if c['split']=='test' or c['limit']>len(rows)*.05:raise ValueError('Invalid smoke subset')
            rows=all_rows[:c['limit']]
        else:
            proof=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if proof['status']!='completed':raise RuntimeError('Successful smoke required')
            if proof.get('protocol_sha256')!=hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest():raise RuntimeError('Smoke protocol differs from full protocol')
            rows=all_rows
        write_json(run.path/'contexts.json',[{k:r[k] for k in ['id','pad_to','cohort']} for r in all_rows])
        write_json(run.path/'protocol.json',protocol)
        results={};ref={}
        if c.get('reference'):
            ref={r['problem_id']:r['official_answer'] for r in json.loads(Path(c['reference']).read_text())}
        for bs in c['batch_sizes']:
            outputs=[]
            # Preserve canonical cohort and context width. Never derive padding from this microbatch.
            groups={}
            for row in rows:groups.setdefault((row['cohort'],row['pad_to']),[]).append(row)
            for group in groups.values():
                for start in range(0,len(group),bs):
                    batch=group[start:start+bs]
                    out=source.run_batch(batch,capture_latents=False)
                    for row,o in zip(batch,out):
                        pred=numeric_answer(o['answer']);rec={'id':row['id'],'question':row['question'],'gold':row['answer'],'answer':o['answer'],'pred':pred,'correct':pred==numeric_answer(row['answer']),'meta':o['meta']}
                        outputs.append(rec);run.logger.log('prediction',batch_size=bs,**rec)
                    if len(outputs)%64==0:print('batch',bs,'done',len(outputs),'/',len(rows),flush=True)
            results[str(bs)]=outputs;write_json(run.path/f'predictions_b{bs}.json',outputs)
            run.write_metrics(batch_sizes_finished=list(results))
        summary={}
        baseline={r['id']:r for r in results[str(c['batch_sizes'][0])]}
        for bs,pred in results.items():
            stat={'accuracy':bootstrap_mean([r['correct'] for r in pred]),'agreement_first_batch':bootstrap_mean([r['pred']==baseline[r['id']]['pred'] for r in pred])}
            if ref:stat['agreement_official']=bootstrap_mean([r['pred']==ref[r['id']] for r in pred])
            summary[bs]=stat
        run.write_metrics(status='completed',results=summary,protocol_sha256=hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest())
        print(run.path,summary,flush=True)
    finally:
        signal.alarm(0);elapsed=time.perf_counter()-t
        run.write_metrics(wall_seconds=elapsed,gpu_hours=elapsed/3600,source_forward_calls=source.forward_calls if source else 0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

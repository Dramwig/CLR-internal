import argparse,json,hashlib,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.dedup_v2 import audit,normalize
from clr.data.splits import freeze,problem_id
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 t=time.perf_counter();train=json.loads(Path(c['train']).read_text())
 if not isinstance(train,list):raise ValueError('Unexpected training corpus format')
 original={k:json.loads(Path(path).read_text()) for k,path in c['original'].items()}
 rows=[{**r,'original_split':k} for k,items in original.items() for r in items]
 if len({r['id'] for r in rows})!=len(rows):raise RuntimeError('Existing input split collision')
 result=audit(train,rows,13)
 summary={}
 for split,items in original.items():
    summary[split]={key:sum(bool(result[r['id']][field]) for r in items) for key,field in [('exact_overlap','exact_training_rows'),('13gram_overlap','ngram_training_row_count'),('union_overlap','overlap')]}
    summary[split]['n']=len(items)
 write_json(run.path/'per_problem_overlap.json',[{**r,**result[r['id']]} for r in rows])
 write_json(run.path/'resources.json',{'train_sha256':hashlib.sha256(Path(c['train']).read_bytes()).hexdigest(),'training_rows':len(train),'training_unique_normalized_questions':len({normalize(r['question']) for r in train}),'normalization':'NFKC+casefold+curly-apostrophe normalization; signed decimal numbers preserved; word tokens; punctuation removed; contiguous 13 tokens; ANY common 13gram excludes heldout problem'})
 clean=[r for r in rows if r['original_split']!='test' and not result[r['id']]['overlap']]
 clean=sorted(clean,key=lambda r:r['id'])
 clean_test=[r for r in original['test'] if not result[r['id']]['overlap']]
 write_json(run.path/'removed_from_cal_test.json',[{**r,'reason':'source_training_exact_or_13gram_overlap'} for r in rows if r['original_split'] in ['cal','test'] and result[r['id']]['overlap']])
 run.write_metrics(overlap=summary,eligible_validation_pool=len(clean),clean_test_n=len(clean_test),source_forward_calls=0,gpu_hours=0)
 if len(clean)<c['val_size']:raise RuntimeError(f'Only {len(clean)} clean validation problems; cannot satisfy val >= {c["val_size"]} without new authorized data')
 split={'train':sorted({problem_id(r['question']) for r in train}),'val':[r['id'] for r in clean[:c['val_size']]],'cal':[r['id'] for r in clean[c['val_size']:]],'test':[r['id'] for r in clean_test]}
 # Exact duplicate IDs have already been excluded by the stronger text audit.
 sha=freeze(c['split_path'],split)
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
 for name,items in [('val',clean[:c['val_size']]),('cal',clean[c['val_size']:]),('test',clean_test)]:
    p=out/(name+'.json');content=json.dumps(items,ensure_ascii=False,indent=2)+'\n'
    if p.exists() and p.read_text()!=content:raise RuntimeError('Frozen v2 dataset exists with different contents')
    p.write_text(content)
 write_json(run.path/'reassignment.json',[{'id':r['id'],'from':r['original_split'],'to':'val' if i<c['val_size'] else 'cal'} for i,r in enumerate(clean)])
 run.write_metrics(status='completed',split_sha256=sha,split_counts={k:len(v) for k,v in split.items()},wall_seconds=time.perf_counter()-t)
 print(run.path,run.metrics,flush=True)

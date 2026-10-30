import argparse,json,hashlib,random
from pathlib import Path
import torch
import requests
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.splits import problem_id
from clr.data.dedup_v2 import normalize
from clr.data.m1 import intermediate_values
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
    destination=Path(c['candidate_path']);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError('Candidate plan already exists; do not overwrite')
    protocol=load_config(c['protocol']);ph=hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest()
    tokenizer=AutoTokenizer.from_pretrained(protocol['source']['base_path'],local_files_only=True)
    old=json.loads(Path(c['v2_split']).read_text());allowed=set(old['train'])
    raw=json.loads(Path(c['source_train']).read_text());unique={};duplicates=0;conflicts=[]
    for q,cot,ans in zip(raw['question'],raw['cot'],raw['answer']):
        pid=problem_id(q)
        if pid not in allowed:continue
        key=normalize(q)
        row={'id':pid,'question':q.strip(),'cot':cot,'answer':ans,'source_distribution':'GSM8k-Aug source-model training distribution'}
        if key in unique:
            duplicates+=1
            if unique[key]['answer']!=ans:conflicts.append({'id':pid,'answer':ans,'first_answer':unique[key]['answer']})
        else:unique[key]=row
    # Deterministic normalization dedup precedes seeded sampling; retain first source row.
    pool=sorted(unique.values(),key=lambda r:r['id']);random.Random(c['seed']).shuffle(pool)
    selected=pool[:c['reader_train']+c['reader_val']]
    if len(selected)<c['reader_train']+c['reader_val']:raise ValueError('Insufficient eligible training problems')
    data={'reader_train':selected[:c['reader_train']], 'reader_val':selected[c['reader_train']:]}
    gold={};resources={}
    for split,url in load_config(c['original_data_config'])['urls'].items():
        response=requests.get(url,timeout=60);response.raise_for_status()
        resources[split]={'url':url,'sha256':hashlib.sha256(response.content).hexdigest()}
        for line in response.text.strip().splitlines():
            parts=line.split('||')
            if len(parts)!=2 or '##' not in parts[1]:raise ValueError('Official source format changed')
            gold[problem_id(parts[0])]=parts[1].rsplit('##',1)[0]
    def contexts(rows,name):
        for start in range(0,len(rows),128):
            group=rows[start:start+128]
            widths=[len(tokenizer.encode(r['question'].strip().replace('  ',' '))) for r in group];width=max(widths)
            if width+1+protocol['num_latent']+1+protocol['source']['max_new_tokens']>1024:
                raise ValueError(f'Context exceeds GPT2 limit in {name} cohort {start//128}; no truncation or resampling')
            for r,length in zip(group,widths):
                r.update(pad_to=width,left_padding=width-length,cohort_id=f'{name}:{start//128}',protocol_sha256=ph)
    for k in data:contexts(data[k],f'M1_candidates:{k}')
    held={}
    for split in ['val','cal','test']:
        rows=json.loads(Path(f'data/M01_v2/{split}.json').read_text())
        for r in rows:
            if r['id'] not in gold:raise ValueError('Missing original gold CoT')
            r['cot']=gold[r['id']];r['source_distribution']='official held-out benchmark'
        contexts(rows,f'M01_v2:{split}');held[split]=rows
    data['val']=held['val'];combined=sorted(held['cal']+held['test'],key=lambda r:r['id']);random.Random(c['seed']).shuffle(combined)
    data['cal']=combined[:c['cal']];data['test']=combined[c['cal']:]
    if len(data['test'])!=c['test']:raise ValueError('Unexpected heldout count')
    for rows in data.values():
        for r in rows:r.update(intermediate_values(r['cot']))
    # Choose the 1k all-layer panel from reader_val only; immune to train budget reduction.
    all_layers=sorted([r['id'] for r in data['reader_val']],key=lambda x:hashlib.sha256(f'{c["seed"]}:layers:{x}'.encode()).hexdigest())[:c['all_layers']]
    seen=set()
    for rows in data.values():
        ids={r['id'] for r in rows}
        if len(ids)!=len(rows) or ids&seen:raise ValueError('Candidate split leakage')
        seen|=ids
    write_json(destination,{'splits':data,'all_layer_ids':all_layers,'protocol_sha256':ph,'resources':resources,'status':'candidate; train count freezes only after smoke budget'})
    write_json(run.path/'duplicate_answer_conflicts.json',conflicts)
    write_json(run.path/'resources.json',resources)
    run.write_metrics(status='completed',counts={k:len(v) for k,v in data.items()},normalization_duplicates=duplicates,duplicate_answer_conflicts=len(conflicts),cot_parse_failed={k:sum(r['cot_parse_failed'] for r in rows) for k,rows in data.items()},candidate_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),gpu_hours=0,source_forward_calls=0)
    print(run.path)

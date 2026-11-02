"""Recovery after host restart; validate completed splits and held-out journals."""
import hashlib,json,os
from pathlib import Path
import torch
import pyarrow.parquet as pq
from safetensors import safe_open

def link_completed(source,dest,planned):
    source,dest=Path(source),Path(dest)
    completed=json.loads((source/'completed.json').read_text())
    rows=pq.read_table(source/'manifest.parquet',columns=['problem_id','kind','t','layer']).to_pylist()
    ids={r['problem_id'] for r in rows}
    if ids!={r['id'] for r in planned} or completed['problem_count']!=len(planned):
        raise ValueError('Completed split IDs/count mismatch')
    early=pq.read_table(source/'early_exit.parquet',columns=['problem_id','t']).to_pylist()
    if len(early)!=7*len(ids) or {(r['problem_id'],r['t']) for r in early}!={(pid,t) for pid in ids for t in range(7)}:
        raise ValueError('Completed split early exits incomplete')
    dest.mkdir(parents=True,exist_ok=False)
    hashes={}
    for p in source.iterdir():
        if not p.is_file():raise ValueError('Unexpected nested output')
        if p.suffix=='.safetensors':
            with safe_open(str(p),framework='pt',device='cpu') as f:
                if f.get_slice('states').get_shape()!=f.get_slice('feedback').get_shape():raise ValueError('Bad completed tensor shapes')
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        os.link(p,dest/p.name)
    return completed['storage'],hashes

def recover_test(source,dest,planned,old_run,pairs):
    source,dest,old_run=map(Path,(source,dest,old_run))
    manifests=[];files=[]
    for index,part in enumerate(sorted(source.glob('manifest_part_*.parquet'))):
        if part.name!=f'manifest_part_{index:05d}.parquet':raise ValueError('Missing manifest part')
        rows=pq.read_table(part).to_pylist();shard=source/f'shard_{index:05d}.safetensors'
        with safe_open(str(shard),framework='pt',device='cpu') as f:
            if f.get_slice('states').get_shape()[0]!=len(rows):raise ValueError('Bad tensor size')
        if [r['offset'] for r in rows]!=list(range(len(rows))):raise ValueError('Bad offsets')
        manifests.extend(rows);files.extend([part,shard])
    ids=list(dict.fromkeys(r['problem_id'] for r in manifests));idset=set(ids)
    if ids!=[r['id'] for r in planned[:len(ids)]]:raise ValueError('Wrong frozen prefix')
    expected={r['id']:r for r in planned[:len(ids)]}
    if len(manifests)!=7*len(ids):raise ValueError('Incomplete clean states')
    if {(r['problem_id'],r['t']) for r in manifests}!={(pid,t) for pid in ids for t in range(7)}:raise ValueError('Missing clean state')
    for r in manifests:
        if r['kind']!='clean' or r['layer']!=12:raise ValueError('Unexpected held-out tensor')
        for k in ['cohort_id','left_padding','pad_to','protocol_sha256']:
            if r[k]!=expected[r['problem_id']][k]:raise ValueError('Context changed')
    journals=[]
    for kind in ['early','mc','variance']:
        rows=json.loads((old_run/f'test_{kind}_partial.json').read_text())
        journals.append([r for r in rows if r['problem_id'] in idset])
    early,mc,variance=journals
    def check(rows,expected,key):
        keys=[key(r) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete or duplicate behavior journal')
    check(early,{(pid,t) for pid in ids for t in range(7)},lambda r:(r['problem_id'],r['t']))
    check(mc,{(pid,t,p) for pid in ids for t in range(6) for p in ['pi','pi_T']},lambda r:(r['problem_id'],r['t'],r['policy']))
    check(variance,{(pid,t,p) for pid,t in pairs if pid in idset for p in ['pi','pi_T']},lambda r:(r['problem_id'],r['t'],r['policy']))
    for row in mc+variance:
        if row['K']!=len(row['correct']) or row['K']!=len(row['answers']) or row['value']!=sum(row['correct'])/row['K']:raise ValueError('Invalid MC journal')
    dest.mkdir(parents=True,exist_ok=False);hashes={}
    for p in files:
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();os.link(p,dest/p.name)
    return manifests,early,mc,variance,hashes

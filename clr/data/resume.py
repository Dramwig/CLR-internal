"""Validated M1 recovery into a new output version; original files stay immutable."""
import hashlib
import math
import json
import os
import shutil
import subprocess
from pathlib import Path
import torch
import pyarrow.parquet as pq
from safetensors import safe_open
from clr.data.loader import ShardWriter


def ensure_space(path, minimum_gib=20):
    """Release managed ballast proactively; low free space itself never aborts."""
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum_gib:
        result = subprocess.run(['/home/me/tools/disk-reserve/reserve.sh', 'release', str(math.ceil(minimum_gib-free+5))], capture_output=True, text=True)
        print(json.dumps({'disk_release_returncode': result.returncode, 'output': result.stdout, 'error': result.stderr}), flush=True)


def recover(source, destination, planned_rows, old_run):
    source, destination, old_run = map(Path, (source, destination, old_run))
    parts = sorted(source.glob('manifest_part_*.parquet'))
    manifests = []
    hashes = {}
    for index, part in enumerate(parts):
        if part.name != f'manifest_part_{index:05d}.parquet':
            raise ValueError('Non-contiguous manifest sequence')
        rows = pq.read_table(part).to_pylist()
        shard = source / f'shard_{index:05d}.safetensors'
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            n = f.get_slice('states').get_shape()[0]
            if n != len(rows) or f.get_slice('feedback').get_shape()[0] != n:
                raise ValueError('Tensor/index size mismatch')
        if [r['offset'] for r in rows] != list(range(n)) or any(r['shard'] != shard.name for r in rows):
            raise ValueError('Invalid shard offsets')
        manifests.extend(rows)
        for p in [part, shard]:
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    ids = list(dict.fromkeys(r['problem_id'] for r in manifests))
    if ids != [r['id'] for r in planned_rows[:len(ids)]]:
        raise ValueError('Recovered IDs do not match frozen prefix')
    expected={r['id']:r for r in planned_rows[:len(ids)]}
    for row in manifests:
        for key in ['cohort_id','left_padding','pad_to','protocol_sha256']:
            if key in expected[row['problem_id']] and row.get(key)!=expected[row['problem_id']][key]:
                raise ValueError('Recovered context mismatch: '+key)
    by_id = {}
    for row in manifests:
        by_id.setdefault(row['problem_id'], []).append(row)
    for pid, rows in by_id.items():
        clean = [r for r in rows if r['kind'] == 'clean' and r['layer'] == 12]
        if sorted(r['t'] for r in clean) != list(range(7)):
            raise ValueError('Incomplete clean states')
        for k in range(8):
            traj = [r for r in rows if r['kind'] == 'traj' and r['traj_id'] == k]
            if not traj or sorted(r['t'] for r in traj) != list(range(traj[0]['t0'], 7)):
                raise ValueError('Incomplete trajectory')
    early = json.loads((old_run / 'reader_train_early_partial.json').read_text())
    idset = set(ids)
    early = [r for r in early if r['problem_id'] in idset]
    if len(early) != len(ids)*7 or len({(r['problem_id'],r['t']) for r in early}) != len(early):
        raise ValueError('Missing early-exit evidence')
    destination.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        os.link(source/name, destination/name)
    return manifests, early, hashes


class ResumedWriter(ShardWriter):
    def __init__(self, root, manifests, limit):
        self.root=Path(root); self.rows=[];self.tensors=[];self.feedback=[]
        self.manifests=list(manifests);self.index=len(list(self.root.glob('manifest_part_*.parquet')))
        self.limit=limit;self.quant=[]
    def close(self):
        result=super().close()
        result['quantization_scope']='newly_generated_states_only; interrupted_run had no durable quantization aggregates'
        result['recovered_rows']=self.recovered_rows
        return result

"""Merge independently generated feature shards with exact group-count checks."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import torch
from clr.audit.features import sha256
from clr.audit.m71_storage import load_bundle
from clr.data.m7 import commitment_route
from clr.audit.m71_runtime import snapshot,live,assert_final_lock
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json


def main(config):
    with recorded(config) as run:
        snapshot(run);root=Path(config['output'])
        opened='test' in config['source_splits']
        if opened:
            assert_final_lock(config)
            run.write_metrics(test_opened=True)
            live(test_opened=True)
        for split in config['source_splits']:
            folder=root/config['representation']/'source'/split;rows=[]
            for part in range(config['partitions']):
                shard=folder/f'part_{part}'
                if not (shard/'completed.json').exists():raise ValueError('Incomplete shard '+str(shard))
                rows.extend(json.loads(line) for line in (shard/'manifest.jsonl').open())
            if len(rows)!=config['counts'][split] or len({r['group_id'] for r in rows})!=len(rows):
                raise ValueError('M7.1 group count or shard-disjointness violation')
            rows.sort(key=lambda r:r['group_index'])
            for row in rows:
                if sha256(row['file'])!=row['sha256']:raise ValueError('Feature bundle hash mismatch')
                if row['e'][1]!=(row['e'][0]+1)%4:raise ValueError('Counterfactual fact mismatch')
                if row['route']!=[row['T'][row['z_pub']][e] for e in row['e']]:raise ValueError('Stored semantic label mismatch')
                if row['commitment_source']!='programmatic' or commitment_route(row['commitment'])!=row['route'][0]:
                    raise ValueError('Commitment protocol mismatch')
                bundle=load_bundle(row['file'])
                if len(bundle['messages'])!=2:raise ValueError('Missing paired message')
                for saved in [*bundle['messages'],*bundle.get('controls',{}).values()]:
                    if saved['values'].dtype!=torch.float16 or not torch.isfinite(saved['values']).all():
                        raise ValueError('Nonfinite or non-fp16 saved features')
                    if set(saved['layers'].unique().tolist())!=set(row['layers']):raise ValueError('Saved layer mismatch')
            if (folder/'manifest.jsonl').exists():raise FileExistsError('Merged split already exists')
            with (folder/'manifest.jsonl').open('x') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            spec={'groups':[r['group_id'] for r in rows],'episode_ids':[r['id'] for r in rows],'data_seed':config['data_seed']}
            splitfile=Path('data/splits')/f'M71_v1_{split}.json'
            if splitfile.exists() and json.loads(splitfile.read_text())!=spec:
                raise ValueError('Frozen M7.1 split changed between representations')
            if not splitfile.exists():write_json(splitfile,spec)
            write_json(folder/'completed.json',{'run':str(run.path),'groups':len(rows),'partitions':config['partitions']})
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=opened)
        splits={name:set(json.loads((Path('data/splits')/f'M71_v1_{name}.json').read_text())['groups'])
                for name in config['source_splits']}
        names=list(splits)
        for i,name in enumerate(names):
            for other in names[i+1:]:
                if splits[name]&splits[other]:raise ValueError('Cross-split episode-group leakage')
        live(stage='feature_shards_merged',representation=config['representation'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

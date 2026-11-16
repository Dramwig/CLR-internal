"""YAML-only M7.1 source smoke and compressed feature generation."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
from pathlib import Path

import torch

from clr.audit.features import KVFeatures,make_projection,record_metadata,sha256
from clr.audit.m71_runtime import Cost,snapshot,live,assert_final_lock
from clr.audit.m71_controls import raw_control,raw_norms
from clr.audit.m71_storage import save_bundle
from clr.data.m7 import counterfactual,format_commitment
from clr.data.m71 import examples
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.records import recorded


def projection_file(config):
    root=Path(config['output']);root.mkdir(exist_ok=True,parents=True)
    path=root/'projection.pt'
    if not path.exists():
        matrix=make_projection(2048,config['feature_dim'],config['projection_seed'])
        torch.save(matrix,path)
        write_json(root/'projection.json',{'file':str(path),'sha256':sha256(path),
                    'seed':config['projection_seed'],'shape':list(matrix.shape),
                    'orthogonality_max_abs':float((matrix.T@matrix-torch.eye(matrix.shape[1])).abs().max())})
    descriptor=json.loads((root/'projection.json').read_text())
    if descriptor['seed']!=config['projection_seed'] or descriptor['shape']!=[2048,config['feature_dim']]:
        raise ValueError('Existing projection differs from requested seed/shape')
    if sha256(path)!=descriptor['sha256']:raise ValueError('M7.1 projection hash mismatch')
    return torch.load(path,map_location='cpu',weights_only=True)


def geometry(example,tokenizer,prefix,index):
    metadata=record_metadata(example,tokenizer)
    if int(prefix.mask[index].sum())!=metadata['token_count']:
        raise ValueError('Offset tokenizer differs from frozen sender encoding')
    metadata['physical_offset']=prefix.length-metadata['token_count']
    return metadata


def persisted(sequence):
    result={key:getattr(sequence,key).detach().cpu().to(torch.float16 if key=='values' else getattr(sequence,key).dtype)
            for key in ['values','mask','layers','segments','token_ranges','kinds']}
    if not torch.isfinite(result['values']).all():raise ValueError('Nonfinite fp16 compressed features')
    return result


def main(config):
    torch.set_num_threads(2);seed_everything(config['seed']);torch.cuda.set_device(config['device'])
    with recorded(config) as run:
        snapshot(run);cost=Cost(run,config);relay=None
        live(stage='source_'+config['mode'],source_run=str(run.path),milestone_complete=False)
        try:
            if config['batch_size']!=2:raise ValueError('M7.1 source preserves independent clean/CF pairs of size 2')
            free,total=torch.cuda.mem_get_info(config['device'])
            if free/2**30<config['min_free_GiB']:
                raise RuntimeError(f'Source GPU free {free/2**30:.3g}GiB below configured requirement')
            matrix=projection_file(config)
            root=Path(config['output'])
            if config['mode']!='smoke':
                selection=json.loads((root/config.get('selection_name','layer_selection.json')).read_text())
                if not selection['projection_loss_below_2pp']:
                    raise ValueError('Projection loss gate has not passed')
                if config.get('representation','pool')=='pool' and config['layers']!=selection['layers']:
                    raise ValueError('Full generation layers differ from preregistered selection')
            if 'test' in config['source_splits']:assert_final_lock(config)
            relay=KVRelay(config)
            transform=KVFeatures(matrix,config['layers'],config['tail_tokens'],config['critical_tokens']).to(config['device'])
            totals=[];rows=[]
            for split in config['source_splits']:
                if split=='test':
                    cost.update(relay,test_opened=True)
                    live(test_opened=True)
                limit=config['smoke_groups'] if config['mode']=='smoke' else None
                parts=config.get('partitions',1);part=config.get('partition',0)
                folder=(run.path/'smoke'/split) if config['mode']=='smoke' else root/config.get('representation','pool')/'source'/split/f'part_{part}'
                if folder.exists() and not config.get('resume',False):raise FileExistsError(str(folder))
                folder.mkdir(parents=True,exist_ok=True)
                def shard(example):
                    if config.get('partition_rule')=='sha256_group':
                        return int(hashlib.sha256(example.meta['group_id'].encode()).hexdigest()[:16],16)%parts
                    return example.meta['group_index']%parts
                entries=[e for e in examples(config,split,limit) if shard(e)==part]
                group_ids=[e.meta['group_id'] for e in entries]
                previous={}
                if (folder/'manifest.jsonl').exists():
                    for line in (folder/'manifest.jsonl').open():
                        item=json.loads(line)
                        if sha256(item['file'])!=item['sha256']:raise ValueError('Resume bundle SHA mismatch')
                        previous[item['id']]=item
                for index,e in enumerate(entries):
                    if e.id in previous:
                        if previous[e.id]['layers']!=config['layers']:raise ValueError('Resume layer mismatch')
                        continue
                    cost.check();cf=counterfactual(e);batch=[e,cf];prefix=relay.encode(batch)
                    bundles=[];metadata=[];norms=[]
                    for bi,item in enumerate(batch):
                        meta=geometry(item,relay.s2.tokenizer,prefix,bi);metadata.append(meta)
                        z=transform(prefix,prefix.mask,meta,bi)
                        bundles.append(persisted(z))
                        if config['mode']=='smoke' and index==0 and bi==0:
                            pairs=list(prefix.layers);layer=config['layers'][0]
                            key,value=pairs[layer]
                            key=key.detach().float().clone().requires_grad_(True)
                            value=value.detach().float().clone().requires_grad_(True)
                            pairs[layer]=(key,value)
                            check=transform(pairs,prefix.mask,meta,bi)
                            check.values.square().mean().backward()
                            if not torch.isfinite(key.grad).all() or not torch.isfinite(value.grad).all():
                                raise ValueError('Nonfinite gradient through compressed KV features')
                            if key.grad.abs().sum()==0 or value.grad.abs().sum()==0:
                                raise ValueError('Missing raw-KV gradient through compressed features')
                            write_json(run.path/'feature_gradient_check.json',{
                                'K_gradient_L1':float(key.grad.abs().sum()),'V_gradient_L1':float(value.grad.abs().sum()),
                                'finite':True,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                                'layer':layer,'actual_sender_KV':True})
                            del check,pairs,key,value
                        norms.append(raw_norms(prefix,config['layers'],bi))
                    controls={};control_norms={}
                    noise_seed=config['seed']+e.meta['group_index']
                    if config.get('include_controls',False) and (split in ['cal_A','cal_B','test'] or
                                                               config['mode']=='smoke' and config.get('smoke_controls',False)):
                        for kind in ['truncate','zero_layer','noise']:
                            ctrl=raw_control(prefix,kind,config['layers'],noise_seed)
                            controls[kind]=persisted(transform(ctrl,ctrl.mask,metadata[0],0))
                            control_norms[kind]=raw_norms(ctrl,config['layers'],0)
                            del ctrl
                    path=folder/(f'{index:05d}.pt.gz' if config.get('compression','gzip1')=='gzip1' else f'{index:05d}.pt')
                    if path.exists():raise FileExistsError('Uncommitted source bundle: '+str(path))
                    save_bundle({'messages':bundles,'controls':controls},path)
                    entry={'id':e.id,'cf_id':cf.id,'group_id':e.meta['group_id'],'file':str(path),
                        'sha256':sha256(path),'e':[e.meta['e_priv'],cf.meta['e_priv']],
                        'route':[e.label,cf.label],'z_pub':e.meta['z_pub'],'T':e.meta['T'],
                        'question_type':e.question_type,'group_index':e.meta['group_index'],
                        'priv_len':e.meta['priv_len'],'d_pub':e.meta['d_pub'],
                        'commitment':format_commitment(e.id,e.label),'commitment_source':'programmatic',
                        'geometry':metadata,'raw_norms':norms,'layers':config['layers'],
                        'control_raw_norms':control_norms,'noise_seed':noise_seed,
                        'control_definition':{'truncate':'first half physical tokens','zero_layer':config['layers'][::2],'relative_noise':.1},
                        'active_layers':sorted(prefix.active_layers),'gates':None if prefix.gates is None else prefix.gates.detach().cpu().tolist(),
                        'K':'post-RoPE','cohort':[e.id,cf.id],'dtype':'float16',
                        'storage_compression':config.get('compression','gzip1'),
                        'projection_sha256':sha256(root/'projection.pt')}
                    with (folder/'manifest.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                    totals.append(path.stat().st_size);rows.append(entry)
                    cost.update(relay,split=split,groups_completed=index+1,test_opened=split=='test')
                    write_json(run.path/'progress.json',{'split':split,'groups_completed':index+1,'total':len(entries),'gpu_hours':cost.elapsed()/3600})
                    print(split,index+1,len(entries),round(cost.elapsed(),2),flush=True)
                    del prefix
                    if index%200==0:
                        used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                        if used>config['disk_cap_GiB']*2**30:raise RuntimeError('M7.1 disk cap exceeded')
                write_json(folder/'completed.json',{'run':str(run.path),'groups':len(entries),'test_opened':split=='test'})
            if config['mode']=='smoke':
                count=sum(config['counts'].values());mean=sum(totals)/len(totals)
                control_smoke=config.get('smoke_controls',False)
                actual_file_mean=mean
                if control_smoke:
                    mean=json.loads((root/'source_smoke.json').read_text())['mean_bytes_per_clean_CF_pair']
                corrupted_groups=sum(config['counts'][s] for s in ['cal_A','cal_B','test'])
                estimated=mean*count+mean/2*corrupted_groups*3
                result={'run':str(run.path),'groups':len(rows),'seconds':cost.elapsed(),
                        'mean_bytes_per_clean_CF_pair':mean,'features_projected_GiB':estimated/2**30,
                        'sender_prefill_estimated_GPUh':cost.elapsed()/len(rows)*count/3600,
                        'smoke_passed':cost.elapsed()<=900,'test_opened':False,
                        'counts':config['counts'],'layers':config['layers']}
                result.update(controls_included=control_smoke,mean_actual_bundle_bytes=actual_file_mean)
                result['storage_compression']=config.get('compression','gzip1')
                write_json(root/('source_controls_smoke.json' if control_smoke else 'source_smoke.json'),result)
                print(json.dumps(result),flush=True)
            cost.update(relay,status='completed',test_opened='test' in config['source_splits'],
                        peak_allocated_GiB=torch.cuda.max_memory_allocated(config['device'])/2**30,
                        peak_reserved_GiB=torch.cuda.max_memory_reserved(config['device'])/2**30)
            live(stage='source_'+config['mode']+'_completed',source_run=str(run.path))
        except Exception:
            cost.update(relay,status='failed');live(stage='source_failed',source_run=str(run.path));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

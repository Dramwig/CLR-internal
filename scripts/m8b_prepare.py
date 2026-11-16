"""Source-state extraction; no test read before an explicit final lock."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import torch
from clr.audit.features import make_projection,record_metadata
from clr.audit.m8b_features import ReceiverStates,cache_kv,cpu_sequence
from clr.audit.m8b_runtime import Cost,snapshot,live
from clr.audit.m71_storage import save_bundle
from clr.data.m71 import examples,fixed_hash_subset
from clr.data.m7 import counterfactual,format_commitment
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from scripts.m8a_evaluate import one,prepared_receiver

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
            p=root/'receiver_projection.pt'
            if not p.exists():torch.save(make_projection(2560,512,c['seed']),p)
            proj=torch.load(p,weights_only=True).to(c['device'])
            kvproj=torch.load('data/M71_v1/projection.pt',weights_only=True)[:,:128].to(c['device'])
            if torch.cuda.mem_get_info()[0]/2**30<c['min_free_GiB']:raise RuntimeError('GPU memory precheck')
            relay=KVRelay(c);observe=ReceiverStates(relay,proj)
            splits={'reader_train':c['train_groups'],'reader_val':c['val_groups']} if c['mode']=='smoke' else c['prepare_splits']
            for split,n in splits.items():
                if split=='test':
                    assert (root/'final_lock.json').exists();cost.test_opened=True
                pop=list(examples(c,split));ids=fixed_hash_subset([e.meta['group_id'] for e in pop],n)
                panel={e.meta['group_id']:e for e in pop};folder=root/('smoke_source' if c['mode']=='smoke' else 'source')/split;folder.mkdir(parents=True,exist_ok=True)
                entries=[]
                with (folder/'manifest.jsonl').open('x') as mf:
                    for i,gid in enumerate(ids):
                        cost.check();e=panel[gid];cf=counterfactual(e);prefix=relay.encode([e,cf]);cost.forward_calls+=1
                        bundle={'messages':[]};pred=[]
                        for bi,ex in enumerate([e,cf]):
                            raw=one(prefix,bi);meta=record_metadata(ex,relay.s2.tokenizer);meta['physical_offset']=raw.length-meta['token_count']
                            with torch.no_grad():
                                logp,seq=observe(raw,prepared_receiver(relay,ex,raw),cost)
                                bundle['messages'].append({'R4':cpu_sequence(seq),'KV':cache_kv(raw,meta,kvproj)});pred.append(int(logp.argmax()))
                        path=folder/f'{i:05d}.pt.gz';save_bundle(bundle,path)
                        entry={'id':e.id,'cf_id':cf.id,'group_id':gid,'e':[e.meta['e_priv'],cf.meta['e_priv']],'label':[e.label,cf.label],'T':e.meta['T'],'z_pub':e.meta['z_pub'],'commitment':format_commitment(e.id,e.label),'receiver_pred':pred,'file':str(path)}
                        entries.append(entry);mf.write(json.dumps(entry)+'\n');mf.flush()
                        cost.update(status='running',split=split,groups_completed=i+1);live(stage='prepare_'+split,run=str(run.path),groups_completed=i+1)
                        if i%10==0:print(split,i+1,n,'GPUh',round((time.monotonic()-cost.start)/3600,4),flush=True)
                        del prefix,bundle,raw,seq,logp
                write_json(folder/'completed.json',{'groups':len(entries),'run':str(run.path),'bytes':sum(p.stat().st_size for p in folder.glob('*'))})
            cost.update(status='completed');live(stage='prepare_'+c['mode']+'_completed',run=str(run.path))
        except Exception:
            cost.update(status='failed');live(stage='prepare_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

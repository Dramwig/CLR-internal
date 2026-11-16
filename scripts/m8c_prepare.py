"""Extract real direct-lookup receiver states; no test access here."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import torch
from clr.audit.m8b_features import ReceiverStates,cpu_sequence
from clr.audit.m8c_runtime import snapshot,Cost,live
from clr.audit.features import KVFeatures,record_metadata
from clr.audit.m71_storage import save_bundle
from clr.data.m71 import examples,fixed_hash_subset
from clr.data.m7 import counterfactual
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.audit.m8c_core import direct_example,short_example,direct_prepared
from scripts.m8a_evaluate import one

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
            gate=json.loads((root/'gate_short_val200_summary.json').read_text());assert gate['passed']
            projection=torch.load(c['receiver_projection'],weights_only=True).to(c['device'])
            kv=KVFeatures(torch.load('data/M71_v1/projection.pt',weights_only=True),c['layers']).to(c['device'])
            relay=KVRelay(c);observe=ReceiverStates(relay,projection)
            for split,n in c['prepare_splits'].items():
                if split=='test':raise RuntimeError('M8c prepare cannot read test before lock')
                pop=list(examples(c,split));ids=fixed_hash_subset([e.meta['group_id'] for e in pop],n);lookup={e.meta['group_id']:e for e in pop}
                folder=root/c.get('source_folder','source')/split;folder.mkdir(parents=True,exist_ok=True);entries=[]
                with (folder/'manifest.jsonl').open('x') as mf:
                    for i,gid in enumerate(ids):
                        cost.check();base=short_example(lookup[gid]);e=direct_example(base);cf=direct_example(counterfactual(base))
                        prefix=relay.encode([e,cf]);cost.forward_calls+=1;bundle={'messages':[]};pred=[]
                        for bi,ex in enumerate([e,cf]):
                            raw=one(prefix,bi);prepared=direct_prepared(relay,ex,raw)
                            md=record_metadata(ex,relay.s2.tokenizer);md['physical_offset']=raw.length-md['token_count']
                            with torch.no_grad():lp,seq=observe(raw,prepared,cost);d0=kv(raw,raw.mask,md)
                            bundle['messages'].append({'R4':cpu_sequence(seq),'D0':cpu_sequence(d0)})
                            pred.append(int(lp.argmax()))
                        path=folder/f'{i:05d}.pt.gz';save_bundle(bundle,path)
                        ent={'id':e.id,'cf_id':cf.id,'group_id':gid,'e':[e.meta['e_priv'],cf.meta['e_priv']],
                             'label':[e.label,cf.label],'T':e.meta['T'],'z_pub':e.meta['z_pub'],
                             'commitment':f'COMMITMENT: private team is {e.options[e.label]}.','receiver_pred':pred,'file':str(path)}
                        entries.append(ent);mf.write(json.dumps(ent)+'\n');mf.flush()
                        cost.update(status='preparing',split=split,groups_completed=i+1)
                        live(stage='prepare_'+split,run=str(run.path),groups_completed=i+1)
                        if (i+1)%25==0:print(split,i+1,n,'seconds',round(time.monotonic()-cost.start,2),flush=True)
                        del prefix,bundle,raw,seq,d0,lp
                write_json(folder/'completed.json',{'groups':len(entries),'run':str(run.path),'log_view':c['log_view']})
            cost.update(status='completed');live(stage='prepare_completed',run=str(run.path))
        except Exception:
            cost.update(status='failed');live(stage='prepare_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

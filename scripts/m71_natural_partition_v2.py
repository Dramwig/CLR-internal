"""Final natural diagnostic shards; preserve the preregistered batch pairs."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import torch
from clr.audit.m71_runtime import Cost,snapshot,assert_final_lock
from clr.data.m71 import examples,fixed_hash_subset
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded

def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);relay=None
        try:
            assert_final_lock(c)
            if torch.cuda.mem_get_info(c['device'])[0]/2**30<c['min_free_GiB']:
                raise RuntimeError('Natural diagnostic shard GPU memory guard failed')
            relay=KVRelay(c)
            # Replay the exact first completed val pair on every target device.
            reference=[json.loads(x) for x in Path(c['reference']).open()][:2]
            val={e.id:e for e in examples(c,'reader_val')};actual={};started=time.monotonic()
            for mode,key in [('direct','direct'),('cot','cot512')]:
                relay.score([val[r['id']] for r in reference],sender=True,answer_mode=mode)
                actual[key]=relay.last_sender
            error=0.;mismatches=[]
            for i,r in enumerate(reference):
                for key in actual:
                    a,b=actual[key][i],r[key]
                    error=max(error,max(abs(x-y) for x,y in zip(a['probs'],b['probs'])))
                    for field in ['pred','generated_ids','generated_text','commitment','correct']:
                        if a[field]!=b[field]:mismatches.append([r['id'],key,field])
            gate={'max_probability_error':error,'mismatches':mismatches,'seconds':time.monotonic()-started,
                  'passed':not mismatches and error<=1e-6,'device':c['device'],'test_opened':False}
            write_json(run.path/'val_device_gate.json',gate);cost.update(relay)
            if not gate['passed']:raise RuntimeError('Val pair replay failed; test shard forbidden')
            root=Path(c['output']);folder=root/c['representation']/'source'/'test'
            cost.update(relay,test_opened=True)
            objects=list(examples(c,'test'));by_id={e.id:e for e in objects}
            selected=set(fixed_hash_subset([e.meta['group_id'] for e in objects],300))
            entries=[json.loads(x) for x in (folder/'manifest.jsonl').open()]
            entries=[r for r in entries if r['group_id'] in selected]
            assert len(entries)==300
            chunks=[entries[o:o+2] for o in range(0,len(entries),2)]
            chunks=[pair for i,pair in enumerate(chunks) if i%c['partitions']==c['partition']]
            write_json(run.path/'fixed_batch_pairs.json',{'ids':[[r['id'] for r in x] for x in chunks],
                       'partition':c['partition'],'partitions':c['partitions'],'original_order_preserved':True})
            out=run.path/'test_natural_commitment.jsonl'
            with out.open('x') as handle:
                completed=0
                for chunk in chunks:
                    cost.check();batch=[by_id[r['id']] for r in chunk];outputs={}
                    for mode,key in [('direct','direct'),('cot','cot512')]:
                        relay.score(batch,sender=True,answer_mode=mode);outputs[key]=relay.last_sender
                    for i,row in enumerate(chunk):
                        handle.write(json.dumps({'id':row['id'],'group_id':row['group_id'],'diagnostic_only':True,
                                     **{key:values[i] for key,values in outputs.items()}})+'\n')
                    handle.flush();completed+=len(chunk);cost.update(relay,test_opened=True)
                    write_json(run.path/'progress.json',{'completed':completed,'total':2*len(chunks),'GPUh':cost.elapsed()/3600})
                    print('test natural shard',c['partition'],completed,2*len(chunks),round(cost.elapsed(),2),flush=True)
            target=root/'natural_partition_v2';target.mkdir(exist_ok=True)
            write_json(target/f"part_{c['partition']}.json",{'run':str(run.path),'file':str(out),'groups':completed,'gate':gate})
            cost.update(relay,status='completed',test_opened=True)
        except Exception:
            cost.update(relay,status='failed');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

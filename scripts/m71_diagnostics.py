"""Fixed receiver/text and natural subsets; natural outputs stay diagnostic-only."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import torch
from clr.audit.features import KVFeatures
from clr.audit.m71_controls import raw_control
from clr.audit.m71_storage import load_bundle
from clr.audit.m71_runtime import Cost,snapshot,live,assert_final_lock
from clr.data.m71 import examples,fixed_hash_subset
from clr.data.m7 import counterfactual
from clr.sources.kv_relay import KVRelay,permute_prefix
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded
from scripts.m71_source import projection_file,geometry,persisted


def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row)+'\n')


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);relay=None;root=Path(c['output']);rep=c['representation']
        try:
            if 'test' in c['source_splits']:assert_final_lock(c)
            free,total=torch.cuda.mem_get_info(c['device'])
            if free/2**30<c['min_free_GiB']:raise RuntimeError('Diagnostic GPU memory guard failed')
            relay=KVRelay(c);transform=KVFeatures(projection_file(c),c['layers'],8,c['critical_tokens']).to(c['device'])
            for split in c['source_splits']:
                if split=='test':
                    cost.update(relay,test_opened=True)
                    live(test_opened=True)
                folder=root/rep/'source'/split
                entries=[json.loads(line) for line in (folder/'manifest.jsonl').open()]
                objects=list(examples(c,split));by_id={e.id:e for e in objects}
                selected=fixed_hash_subset([e.meta['group_id'] for e in objects],c['subset_counts'][split])
                write_json(run.path/(split+'_fixed_subset.json'),{'groups':selected,'rule':'sha256_group_ascending'})
                destination=run.path if c.get('mode')=='smoke' else folder
                out=destination/(split+'_'+c['diagnostic']+'.jsonl' if c.get('mode')=='smoke' else c['diagnostic']+'.jsonl')
                if out.exists() and not c.get('resume',False):raise FileExistsError(str(out))
                seen={json.loads(line)['id'] for line in out.open()} if out.exists() else set()
                if c['diagnostic']=='natural_commitment':
                    pending=[e for e in entries if e['group_id'] in selected and e['id'] not in seen]
                    for off in range(0,len(pending),2):
                        cost.check();chunk=pending[off:off+2];batch=[by_id[e['id']] for e in chunk];outputs={}
                        for mode in ['direct','cot']:
                            relay.score(batch,sender=True,answer_mode=mode)
                            outputs['cot512' if mode=='cot' else mode]=relay.last_sender
                        for bi,e in enumerate(chunk):
                            append(out,{'id':e['id'],'group_id':e['group_id'],'diagnostic_only':True,
                                        **{mode:values[bi] for mode,values in outputs.items()}})
                            seen.add(e['id'])
                        cost.update(relay,test_opened=split=='test')
                        write_json(run.path/'progress.json',{'split':split,'completed':len(seen),'total':len(selected),'GPUh':cost.elapsed()/3600})
                        print(split,c['diagnostic'],len(seen),len(selected),round(cost.elapsed(),2),flush=True)
                    write_json(destination/(c['diagnostic']+'_completed.json'),{'run':str(run.path),'groups':len(selected),'diagnostic_only':True})
                    continue
                for index,entry in enumerate(entries):
                    if entry['group_id'] not in selected or entry['id'] in seen:continue
                    cost.check();e=by_id[entry['id']];cf=counterfactual(e)
                    if c['diagnostic']=='receiver':
                        prefix=relay.encode([e,cf]);saved=load_bundle(entry['file'])
                        for bi,item in enumerate([e,cf]):
                            z=persisted(transform(prefix,prefix.mask,geometry(item,relay.s2.tokenizer,prefix,bi),bi))
                            for key,value in z.items():
                                if not torch.equal(value,saved['messages'][bi][key]):
                                    raise ValueError('Recomputed receiver message differs from source feature '+key)
                        probs=relay.score([e,cf],prefix)
                        result={'id':e.id,'group_id':entry['group_id'],'clean':probs[0].tolist(),'counterfactual':probs[1].tolist(),
                                'none':relay.score([e])[0].tolist(),'feature_replay_max_abs':0.0}
                        for kind in ['truncate','zero_layer','noise']:
                            control=raw_control(prefix,kind,c['layers'],entry['noise_seed'])
                            result['corrupted_'+kind]=relay.score([e,cf],control)[0].tolist()
                            del control
                        del prefix
                        donor=entries[(index+1)%len(entries)];d=by_id[donor['id']]
                        prefix=permute_prefix(relay.encode([d,counterfactual(d)]),[0])
                        result['mismatch']=relay.score([e],prefix)[0].tolist();result['donor_id']=d.id
                        del prefix
                        append(out,result)
                    else:raise ValueError('Unknown diagnostic '+c['diagnostic'])
                    cost.update(relay,test_opened=split=='test')
                    write_json(run.path/'progress.json',{'split':split,'completed':len(seen)+1,'total':len(selected),'GPUh':cost.elapsed()/3600})
                    seen.add(e.id);print(split,c['diagnostic'],len(seen),len(selected),round(cost.elapsed(),2),flush=True)
                write_json(destination/(c['diagnostic']+'_completed.json'),{'run':str(run.path),'groups':len(selected),'diagnostic_only':c['diagnostic']=='natural_commitment'})
            cost.update(relay,status='completed',test_opened='test' in c['source_splits'])
            live(stage=c['diagnostic']+'_completed',diagnostic_run=str(run.path))
        except Exception:
            cost.update(relay,status='failed');live(stage='diagnostic_failed',diagnostic_run=str(run.path));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

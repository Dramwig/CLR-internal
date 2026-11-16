"""Versioned mismatch repair: receiver consumes the same padded KV as the audit."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import torch
from clr.sources.kv_relay import KVRelay,permute_prefix,audit_tensor
from clr.data.m7 import split_examples,counterfactual
from clr.audit.m7_runtime import Cost,snapshot,live
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);relay=None;root=Path(c['output'])
        try:
            relay=KVRelay(c);results={}
            for split in c['source_splits']:
                if split=='test':raise ValueError('Repair is pretest-only; final source generator already uses v2')
                folder=root/'source'/split;rows=list(split_examples(c,split,c['counts'][split]))
                target=run.path/(split+'_mismatch_v2.jsonl') if c['mode']=='smoke' else folder/'mismatch_v2.jsonl'
                if target.exists():raise FileExistsError('Refuse to overwrite mismatch v2 artifact')
                panel=rows[:c.get('smoke_groups',len(rows))] if c['mode']=='smoke' else rows
                equal=0
                for n,e in enumerate(panel):
                    cost.check();donor=rows[(n+1)%len(rows)]
                    prefix=permute_prefix(relay.encode([donor,counterfactual(donor)]),[0])
                    z,meta=audit_tensor(prefix,0,c['layers'])
                    saved=torch.load(folder/(donor.id+'.pt'),map_location='cpu',weights_only=True)['messages'][0]
                    if not torch.equal(z,saved['z']) or meta!=saved['meta']:
                        raise ValueError('Reconstructed mismatch tensor differs from saved donor')
                    p=relay.score([e],prefix)[0]
                    record={'id':e.id,'donor_id':donor.id,'receiver_probs':p.tolist(),
                            'message_geometry':'saved_original_CF_pair_v2','audit_tensor_max_abs':0.0}
                    with target.open('a') as f:f.write(json.dumps(record)+'\n')
                    equal+=1;cost.update(relay,test_opened=False)
                    if n%20==0:print(split,n+1,len(panel),flush=True)
                    del prefix
                results[split]={'messages':equal,'tensor_and_metadata_equal_fraction':1.,'max_abs':0.,'path':str(target)}
            if c['mode']=='smoke' and cost.elapsed()>900:raise RuntimeError('Mismatch v2 smoke exceeded 15 minutes')
            write_json(run.path/'repair.json',results)
            if c['mode']!='smoke':write_json(root/'mismatch_v2_complete.json',{'run':str(run.path),'splits':results})
            live(stage='mismatch_v2_'+c['mode']+'_completed',mismatch_repair_run=str(run.path))
            cost.update(relay,status='completed',test_opened=False,results=results)
        except Exception:
            cost.update(relay,status='failed',test_opened=False);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

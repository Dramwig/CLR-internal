"""Val-only M8c direct-lookup admission gate."""
import sys
sys.dont_write_bytecode=True
import argparse, json, time
from pathlib import Path
import torch
from clr.audit.m8c_runtime import snapshot, Cost, live
from clr.data.m71 import examples, fixed_hash_subset
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything, write_json
from clr.utils.records import recorded
from clr.audit.m8c_core import direct_example, short_example, direct_prepared, direct_receiver
from clr.data.m7 import counterfactual
from scripts.m8a_evaluate import one

def main(c):
    torch.set_num_threads(2); torch.cuda.set_device(c['device']); seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c); cost=Cost(run,c)
        try:
            relay=KVRelay(c); pop=list(examples(c,'reader_val')); ids=fixed_hash_subset([e.meta['group_id'] for e in pop],c['groups'])
            lookup={e.meta['group_id']:e for e in pop}; rows=[]
            for i,gid in enumerate(ids):
                cost.check()
                base=lookup[gid];donor=lookup[ids[(i+1)%len(ids)]]
                def convert(ex):return direct_example(short_example(ex) if c['log_view']=='short' else ex)
                e=convert(base)
                prefix=relay.encode([e,convert(counterfactual(base))]); cost.forward_calls+=1
                clean=one(prefix,0)
                donor_prefix=relay.encode([convert(donor),convert(counterfactual(donor))]);cost.forward_calls+=1
                mismatch=one(donor_prefix,0)
                for category,pfx in [('message',clean),('none',None),('mismatch',mismatch)]:
                    prepared=direct_prepared(relay,e,pfx)
                    with torch.no_grad(): lp=direct_receiver(relay,pfx,prepared,cost)
                    rows.append({'group_id':gid,'category':category,'pred':int(lp.argmax()),'label':e.label,
                                 'correct':int(lp.argmax())==e.label,'target_logp':float(lp[e.label])})
                cost.update(groups_completed=i+1); live(stage='m8c_gate',run=str(run.path),groups_completed=i+1)
                print('group',i+1,len(ids),'seconds',round(time.monotonic()-cost.start,2),flush=True)
            out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
            write_json(out/f'{c["mode"]}_rows.json',{'run':str(run.path),'groups':len(ids),'rows':rows,'test_opened':False})
            import pandas as pd
            df=pd.DataFrame(rows);summary=df.groupby('category').correct.mean().to_dict()
            passed=(summary.get('message',0)>=.8 and summary.get('none',1)<=.35 and summary.get('mismatch',1)<=.35)
            write_json(out/f'{c["mode"]}_summary.json',{'run':str(run.path),'accuracies':summary,'thresholds':{'message':.8,'none':.35,'mismatch':.35},'passed':passed,'groups':len(ids),'seconds':time.monotonic()-cost.start})
            cost.update(status='completed');live(stage='m8c_gate_completed',run=str(run.path),gate_passed=passed)
        except Exception:
            cost.update(status='failed');live(stage='m8c_gate_failed',run=str(run.path));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

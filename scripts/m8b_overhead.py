"""Val-only inference latency, including real receiver prefill for R4."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import torch
from scripts.m8b_evaluate import Panel
from scripts.m8a_evaluate import prepared_receiver,receiver
from clr.audit.features import record_metadata
from clr.audit.m8b_reader import rng_seed
from clr.audit.m8_p0 import public_constraints
from clr.audit.m8b_runtime import Cost,snapshot
from clr.data.m71 import examples,fixed_hash_subset
from clr.data.m7 import format_commitment
from clr.sources.kv_relay import KVRelay
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.records import recorded
from clr.utils.config import load_config

def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize();return result,(time.perf_counter()-start)*1000

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            relay=KVRelay(c);panel=Panel(c,relay);pop=list(examples(c,'reader_val'));ids=fixed_hash_subset([e.meta['group_id'] for e in pop],c['groups']);lookup={e.meta['group_id']:e for e in pop};rows=[]
            with torch.no_grad():
                for gid in ids:
                    cost.check();e=lookup[gid];from clr.data.m7 import counterfactual
                    from scripts.m8a_evaluate import one
                    raw=one(relay.encode([e,counterfactual(e)]),0);cost.forward_calls+=1;md=record_metadata(e,relay.s2.tokenizer);md['physical_offset']=raw.length-md['token_count'];prep=prepared_receiver(relay,e,raw)
                    entry={'id':e.id,'group_id':gid,'commitment':format_commitment(e.id,e.label),**public_constraints(e.x_pub)}
                    _,baseline=timed(lambda:receiver(relay,raw,prep,cost));(_,state),r4=timed(lambda:panel.observe(raw,prep,cost))
                    for (d,s),a in panel.audits.items():
                        for k in ([1,4,16,64] if d=='R1' else [16]):
                            seq,source_ms=timed(lambda:panel.feature(d,raw,md,rng_seed(gid,d,s),state))
                            # Serialization is an audit cost, included in reader latency.
                            _,reader_ms=timed(lambda:a.loss(seq,a.prepare(entry,rng_seed(s,'latency',gid),k)))
                            rows.append({'group_id':gid,'design':d,'seed':s,'k':k,'source_ms':source_ms,'reader_ms':reader_ms,'receiver_ms':baseline,'receiver_with_state_ms':r4,'total_observation_audit_ms':source_ms+reader_ms+(r4 if d=='R4' else 0),'incremental_over_receiver_ms':source_ms+reader_ms+(r4-baseline if d=='R4' else 0)})
                    cost.update(status='benchmarking',groups_completed=len(rows));del raw,state
            write_json(Path(c['output'])/'overhead.json',{'run':str(run.path),'rows':rows,'basis':'wall ms with CUDA synchronization; 20 val messages, all three seeds; receiver prefill is already necessary but R4 prevents pre-consumption rejection'})
            cost.update(status='completed')
        except Exception:cost.update(status='failed');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

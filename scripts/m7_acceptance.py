"""Validation-only smoke: recorded historical replay, raw-KV equality, causal ablations."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import time
import torch
import numpy as np
from clr.sources.kv_relay import KVRelay, sender_text, permute_prefix, audit_tensor
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json, seed_everything
from clr.audit.m7_runtime import live, snapshot, Cost


def historical(relay,c,run):
    from latdef.inject.adapter import KVAdapter
    from latdef.inject.cache import left_pad, prefix_forward
    from latdef.handoff.base import S2Input
    from latdef.s2.budgets import BudgetArm
    historical_arm=BudgetArm('direct',c['model_path'],'direct',0)
    ref=Path(c['reference_run']); core=json.loads((ref/'core_subset.json').read_text())
    data=Path(c['idea_a'])/'data/synthetic/m1-v2-r2/val.jsonl'
    wanted={r['id'] for r in core}
    rows={r['id']:r for r in map(json.loads,data.open()) if r['id'] in wanted}
    messages=torch.load(ref/'seeded_messages.pt',map_location=relay.s2.device,weights_only=True)
    adapter=KVAdapter(relay.s2.model.config,d_msg=64,rank=16,gate_init=1e-12,
                      kv_rms=json.loads((ref/'real_kv_rms.json').read_text())).to(relay.s2.device)
    adapter.load_state_dict(torch.load(ref/'adapter_initial_layers36.pt',map_location=relay.s2.device,weights_only=True))
    max_kv=0.; equal=0; total=0; logits=[]
    with torch.no_grad():
        for offset in range(0,len(core),2):
            m=messages[offset:offset+2]
            prefix=adapter(m)
            # Independent equation-level construction from frozen weights.
            normalized=torch.nn.functional.layer_norm(m.float(),(64,))
            for i,(k,v) in enumerate(prefix.layers):
                manual=(normalized@adapter.down[str(i)].weight.T)@adapter.up[str(i)].weight.T
                manual=manual.reshape(len(m),32,2,8,128)
                for actual,expected in [(k,manual[:,:,0].permute(0,2,1,3)*adapter.kv_rms[i,0]),
                                        (v,manual[:,:,1].permute(0,2,1,3)*adapter.kv_rms[i,1])]:
                    max_kv=max(max_kv,float((actual-expected).abs().max()))
                    equal+=int((actual==expected).sum());total+=actual.numel()
            batch=[S2Input(rows[x['id']]['x_pub'],tuple(rows[x['id']]['options'])) for x in core[offset:offset+2]]
            prompts,_,_,_=relay.s2._prepare(batch,historical_arm,None)
            ids,mask=left_pad([p+relay.s2.direct_suffix for p in prompts],relay.s2.tokenizer.pad_token_id,relay.s2.device)
            out=prefix_forward(relay.s2.model,ids,mask,prefix=prefix,logits_to_keep=1)
            relay.forward_calls+=1;relay.forward_examples+=len(batch);relay.prefill_tokens+=int(mask.sum())
            logits.append(out.logits[:,-1].float().cpu())
        actual=torch.cat(logits);expected=torch.load(ref/'next_logits_S32_L36.pt',map_location='cpu',weights_only=True)
        diff=(actual-expected).abs();agreement=float((actual.argmax(-1)==expected.argmax(-1)).float().mean())
        torch.save(actual,run.path/'historical_replayed_logits.pt')
    return {'episodes':len(core),'reconstructed_message_equal_fraction':equal/total,
            'message_reconstruction_max_abs':max_kv,'historical_logits_max_abs':float(diff.max()),
            'historical_logits_mean_abs':float(diff.mean()),'receiver_argmax_agreement':agreement,
            'historical_raw_KV_saved':False,
            'passed':float(diff.max())<=c['historical_logit_atol'] and agreement==1 and max_kv==0,
            'scope':'historical seeded prefix reconstructed from frozen adapter; raw relay compared separately'}


def select_val(c):
    from latdef.data.schema import Example
    path=Path(c['idea_a'])/'data/synthetic/m1-v2-r2/val.jsonl'
    rows=[Example(**r) for r in map(json.loads,path.open()) if r['meta']['alpha']==1
          and r['meta']['d_pub']==0 and r['meta']['priv_len']==12]
    rows.sort(key=lambda e:e.meta['group_index'])
    return rows[:c['smoke_groups']]


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run); cost=Cost(run,c);live(stage='acceptance_running',acceptance_run=str(run.path),milestone_complete=False)
        print(run.path,flush=True)
        relay=None
        try:
            relay=KVRelay(c)
            hist=historical(relay,c,run);cost.update(relay,historical=hist);print(json.dumps(hist),flush=True)
            rows=select_val(c);preds=[];storage=[];raw_error=0.;raw_equal=0;raw_count=0
            from latdef.inject.transfer import extract_private_kv
            for j in range(0,len(rows),c['batch_size']):
                cost.check();batch=rows[j:j+c['batch_size']]
                if len(batch)<2:raise ValueError('Causal mismatch requires >=2 rows per batch')
                prefix=relay.encode(batch)
                ref=extract_private_kv(relay.s2.model,relay.s2.tokenizer,[sender_text(e) for e in batch],max_context=c['max_context'])
                relay.forward_calls+=1;relay.forward_examples+=len(batch)
                for (a,b),(x,y) in zip(prefix.layers,ref.layers):
                    for lhs,rhs in ((a,x),(b,y)):
                        raw_error=max(raw_error,float((lhs-rhs).abs().max()));raw_equal+=int((lhs==rhs).sum());raw_count+=lhs.numel()
                del ref
                clean=relay.score(batch,prefix);none=relay.score(batch)
                mismatch=relay.score(batch,permute_prefix(prefix,list(range(1,len(batch)))+[0]))
                commitments,cp=relay.commitments(batch)
                for i,e in enumerate(batch):
                    z,meta=audit_tensor(prefix,i,c['layers']);storage.append(z.numel()*z.element_size())
                    # Save one smoke tensor as evidence, no test samples.
                    if j==0 and i==0:torch.save({'z':z,'meta':meta},run.path/'smoke_KV.pt')
                    preds.append({'id':e.id,'group_id':e.meta['group_id'],'e':e.meta['e_priv'],
                                  'label':e.label,'kind':e.question_type,'commitment':commitments[i],
                                  'sender_correct':int(cp[i].argmax())==e.label,
                                  **{name:{'probs':p[i].tolist(),'pred':int(p[i].argmax()),'correct':int(p[i].argmax())==e.label}
                                     for name,p in [('message',clean),('none',none),('mismatch',mismatch)]}})
                write_json(run.path/'causal_predictions.json',preds)
                cost.update(relay,episodes=len(preds));print(f'causal {len(preds)}/{len(rows)}',flush=True)
                del prefix
            causal={name:float(np.mean([p[name]['correct'] for p in preds])) for name in ['message','none','mismatch']}
            result={'historical':hist,'raw_KV_max_abs':raw_error,'raw_KV_equal_fraction':raw_equal/raw_count,
                    'causal_accuracy':causal,'sender_accuracy':float(np.mean([p['sender_correct'] for p in preds])),
                    'episodes':len(rows),'groups':len(rows),'storage_mean_bytes_per_clean':float(np.mean(storage)),
                    'storage_max_bytes_per_clean':max(storage),'projected_target_KV_GiB_2messages_per_episode':float(np.mean(storage))*12000*2/2**30,
                    'smoke_passed':cost.elapsed()<=900 and raw_error==0 and hist['passed'],
                    'causal_gain_vs_none':causal['message']-causal['none'],
                    'causal_gain_vs_mismatch':causal['message']-causal['mismatch']}
            cost.update(relay,status='completed',**result)
            write_json(Path(c['output'])/'acceptance.json',{'run':str(run.path),**result})
            live(stage='acceptance_smoke_completed',acceptance_run=str(run.path),acceptance=result)
            print(json.dumps(result),flush=True)
        except Exception:
            cost.update(relay,status='failed');live(stage='acceptance_failed',acceptance_run=str(run.path));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

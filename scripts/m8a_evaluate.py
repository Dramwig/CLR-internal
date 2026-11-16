"""YAML-driven val smoke and locked raw-KV robustness evaluation."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import torch
from clr.audit.features import KVFeatures,record_metadata
from clr.audit.m8a_runtime import snapshot,Cost,live
from clr.audit.m8a_perturb import Perturbation,feature_difference
from clr.audit.m8a_reader import Auditors
from clr.data.m71 import examples,fixed_hash_subset
from clr.data.m7 import counterfactual,format_commitment
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded


def one(prefix,index):
    from latdef.inject.cache import PrefixKV
    return PrefixKV(tuple((k[index:index+1],v[index:index+1]) for k,v in prefix.layers),prefix.mask[index:index+1],prefix.active_layers,prefix.gates)


def prepared_receiver(relay,e,prefix):
    from latdef.handoff.base import S2Input
    prompts,letters,_,_=relay.s2._prepare([S2Input(e.x_pub,tuple(e.options))],relay.arm,prefix)
    from latdef.inject.cache import left_pad
    ids,mask=left_pad([prompts[0]+relay.s2.direct_suffix],relay.s2.tokenizer.pad_token_id,relay.s2.device)
    return ids,mask,torch.tensor(letters,device=ids.device)


def receiver(relay,prefix,prepared,cost):
    from latdef.inject.cache import prefix_forward
    ids,mask,letters=prepared
    out=prefix_forward(relay.s2.model,ids,mask,prefix=prefix,logits_to_keep=1)
    cost.forward_calls+=1
    logits=out.logits[:,-1].float().gather(1,letters)
    return logits.log_softmax(-1)[0]


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
            if not json.loads((root/'P0.json').read_text())['passed']:raise ValueError('P0 not passed')
            if torch.cuda.mem_get_info(c['device'])[0]/2**30<c['min_free_GiB']:raise RuntimeError('M8a GPU memory precheck failed')
            live(stage='evaluate_'+c['mode'],run=str(run.path),milestone_complete=False)
            relay=KVRelay(c);auditors=Auditors(c)
            transform=KVFeatures(torch.load('data/M71_v1/projection.pt',weights_only=True),c['layers']).to(c['device'])
            if c['split']=='test':
                lock=json.loads((root/'final_lock.json').read_text())
                from clr.audit.features import sha256
                for path,digest in lock['sha256'].items():
                    if sha256(path)!=digest:raise ValueError('M8a final lock changed: '+path)
                cost.test_opened=True;cost.update()
            population=list(examples(c,c['split']))
            panel_ids=fixed_hash_subset([e.meta['group_id'] for e in population],c['groups'])
            panel=sorted([e for e in population if e.meta['group_id'] in set(panel_ids)],key=lambda e:panel_ids.index(e.meta['group_id']))
            if c['split']=='test':
                frozen=json.loads((root/'test500_ids.json').read_text())
                assert panel_ids==frozen['group_ids']
            partition=c.get('partition',0);parts=c.get('partitions',1)
            panel=[e for i,e in enumerate(panel) if i%parts==partition]
            results=[];timings=[]
            with (run.path/'results.jsonl').open('x') as handle:
                for index,e in enumerate(panel):
                    start=time.monotonic();cost.check()
                    cf=counterfactual(e);prefix=relay.encode([e,cf]);cost.forward_calls+=1
                    base=one(prefix,0);cfprefix=one(prefix,1)
                    meta=record_metadata(e,relay.s2.tokenizer);meta['physical_offset']=base.length-meta['token_count']
                    commitment=format_commitment(e.id,e.label)
                    fixed={s:auditors.prepare(e,commitment,s) for s in c['reader_seeds']}
                    with torch.no_grad():
                        clean_feature=transform(base,base.mask,meta)
                        replay_exact=None
                        if c['mode'].startswith('smoke'):
                            from clr.audit.m71_storage import load_bundle
                            if index==0:
                                stored_entries={r['id']:r for r in [json.loads(line) for line in Path(f'data/M71_v1/pool/source/{c["split"]}/manifest.jsonl').read_text().splitlines()]}
                            stored=load_bundle(stored_entries[e.id]['file'])['messages'][0]['values'].to(c['device'])
                            replay_exact=float((stored==clean_feature.values.half()).float().mean())
                            if replay_exact!=1.:raise ValueError('Frozen sender feature replay differs')
                        prepared=prepared_receiver(relay,e,base)
                        clean_logp=receiver(relay,base,prepared,cost)
                        original=relay.score([e],base)[0].to(c['device']);cost.forward_calls+=1
                        err=float((clean_logp.exp()-original).abs().max())
                        if err>1e-5:raise ValueError('Receiver differentiable path differs from frozen direct scoring')
                        clean_audit=auditors.evaluate(clean_feature,fixed)
                        clean_pred=int(clean_logp.argmax())
                    common={'id':e.id,'group_id':e.meta['group_id'],'target':cf.label,'clean_pred':clean_pred,
                            'clean_correct':clean_pred==e.label,'clean_audit':clean_audit,'receiver_replay_error':err,
                            'frozen_feature_replay_exact_fraction':replay_exact}
                    for kind in c['kinds']:
                        if kind.startswith('E0'):
                            from clr.audit.m8a_literature import baseline,ratios as baseline_ratios
                            choices=[None] if kind in ['E0_CF','E0_scale8','E0_sign','E0_randomnorm'] else c['epsilons']
                            for epsilon in choices:
                                with torch.no_grad():
                                    if kind=='E0_CF':
                                        changed=cfprefix;metadata=record_metadata(cf,relay.s2.tokenizer)
                                        metadata['physical_offset']=changed.length-metadata['token_count']
                                    else:
                                        metadata=meta
                                        direction=torch.load(root/'diffmean.pt',weights_only=True,map_location=c['device']) if kind=='E0_diffmean' else None
                                        rng_seed=int(hashlib.sha256((e.id+str(c['seed'])).encode()).hexdigest()[:12],16)
                                        changed=baseline(base,kind,epsilon,meta,c['layers'],rng_seed,direction)
                                    logp=receiver(relay,changed,prepared,cost);feature=transform(changed,changed.mask,metadata)
                                    audit=auditors.evaluate(feature,fixed);ratios=baseline_ratios(base,changed)
                                    maximum=max(ratios.values())
                                    if epsilon is not None and maximum>epsilon*(1+1e-5):raise ValueError('Baseline raw-KV constraint violation')
                                    item={**common,'kind':kind,'epsilon':epsilon,'steps':0,'step_budget':0,'attack_seed':None,'lambda':0,
                                        'pred':int(logp.argmax()),'target_logp':float(logp[cf.label]),'target_achieved':int(logp.argmax())==cf.label,
                                        'changed_to_target':int(logp.argmax())==cf.label and clean_pred!=cf.label,'audit':audit,
                                        'max_relative_norm':maximum,'layer_norms':ratios,'feature_difference':None,'early_success':False}
                                    handle.write(json.dumps(item)+'\n');handle.flush();results.append(item)
                                del changed,feature,logp
                            continue
                        for epsilon in c['epsilons']:
                            for attack_seed in (c['reader_seeds'] if kind=='E3' else [None]):
                                for lam in (c['lambdas'] if kind=='E3' else [0.]):
                                    attack=Perturbation(base,meta,kind,c['layers'])
                                    maxsteps=max(c['steps']);step_start=time.monotonic()
                                    eot=None
                                    if kind=='E3':
                                        eot=[auditors.prepare(e,commitment,attack_seed+(j+1)*100003) for j in range(c['eot_panels'])]
                                    stopped=False
                                    for step in range(1,maxsteps+1):
                                        attack.leaf();changed=attack.build()
                                        logp=receiver(relay,changed,prepared,cost)
                                        objective=logp[cf.label]
                                        if kind=='E3':objective=objective-lam*auditors.score(transform(changed,changed.mask,meta),attack_seed,eot[(step-1)%len(eot)])
                                        grads=torch.autograd.grad(objective,list(attack.delta.values()))
                                        attack.step(grads,epsilon,maxsteps,c['step_multiplier'])
                                        del objective,logp,changed,grads
                                        if kind=='E2':attack.enforce_fp16_identity(transform,meta,clean_feature)
                                        if c.get('early_success',False):
                                            with torch.no_grad():
                                                probe=attack.build(False);plogp=receiver(relay,probe,prepared,cost)
                                                hit=int(plogp.argmax())==cf.label
                                                if kind=='E3' and hit:
                                                    ps=float(auditors.score(transform(probe,probe.mask,meta),attack_seed,eot[step%len(eot)]))
                                                    hit=ps<=auditors.cal[attack_seed]['thresholds']['16']['0.01']
                                                stopped=hit
                                            del probe,plogp
                                        if step in c['steps'] or stopped:
                                            with torch.no_grad():
                                                changed=attack.build(False);logp=receiver(relay,changed,prepared,cost)
                                                feature=transform(changed,changed.mask,meta);audit=auditors.evaluate(feature,fixed)
                                                ratios=attack.ratios(changed);maximum=max(ratios.values())
                                                if epsilon is not None and maximum>epsilon*(1+1e-5):raise ValueError('Transmitted KV norm constraint violation')
                                                item={**common,'kind':kind,'epsilon':epsilon,'steps':step,'attack_seed':attack_seed,'lambda':lam,
                                                    'pred':int(logp.argmax()),'target_logp':float(logp[cf.label]),'target_achieved':int(logp.argmax())==cf.label,
                                                    'changed_to_target':int(logp.argmax())==cf.label and clean_pred!=cf.label,
                                                    'audit':audit,'max_relative_norm':maximum,'layer_norms':ratios,
                                                    'feature_difference':feature_difference(clean_feature,feature),'seconds':time.monotonic()-step_start}
                                                if kind=='E2' and item['feature_difference']['fp16_exact_fraction']!=1.:
                                                    raise ValueError('E2 exact reader-input identity check failed')
                                                requested=[n for n in c['steps'] if n>=step] if stopped else [step]
                                                for n in requested:
                                                    output_item={**item,'step_budget':n,'early_success':stopped}
                                                    handle.write(json.dumps(output_item)+'\n');handle.flush();results.append(output_item)
                                            del changed,feature,logp
                                        if stopped:break
                                    del attack,eot
                                    cost.update(groups_completed=index,rows=len(results));cost.check()
                    timings.append({'group_id':e.meta['group_id'],'seconds':time.monotonic()-start,'prefix_tokens':base.length})
                    write_json(run.path/'timings.json',timings)
                    print('group',index+1,len(panel),'rows',len(results),'seconds',round(timings[-1]['seconds'],3),flush=True)
                    del prefix,base,cfprefix,clean_feature
            elapsed=time.monotonic()-cost.start
            write_json(root/f'{c["mode"]}_part{partition}.json',{'run':str(run.path),'groups':len(panel),'rows':len(results),'seconds':elapsed,'test_opened':cost.test_opened})
            cost.update(status='completed',peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            live(stage='evaluate_'+c['mode']+'_completed',run=str(run.path))
        except Exception:
            cost.update(status='failed');live(stage='evaluation_failed',run=str(run.path));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

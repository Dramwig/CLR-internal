"""Val/test binary-search attack-cost diagnostic with explicit failure censoring."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time,secrets
from pathlib import Path
import torch
from clr.audit.m8c_runtime import snapshot,Cost,live
from clr.audit.m8c_reader import load_audits,commitment,rng_seed
from clr.audit.m8c_core import direct_example,short_example,direct_prepared,direct_receiver
from clr.audit.m8b_features import ReceiverStates
from clr.audit.features import KVFeatures,record_metadata,sha256
from clr.audit.m8a_perturb import Perturbation
from clr.data.m71 import examples,fixed_hash_subset
from clr.data.m7 import counterfactual
from clr.sources.kv_relay import KVRelay
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from scripts.m8a_evaluate import one

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output'])
            if torch.cuda.mem_get_info()[0]/2**30<c['min_free_GiB']:raise RuntimeError('M8c minimum GPU memory precheck failed')
            if c['split']=='test':
                lock=json.loads((root/'final_lock.json').read_text())
                for p,h in lock['sha256'].items():assert sha256(p)==h,p
                cost.test_opened=True
                cost.update(status='opening_locked_test');live(test_opened=True,background_tasks_active=True)
            relay=KVRelay(c);audits=load_audits(c);transform=KVFeatures(torch.load('data/M71_v1/projection.pt',weights_only=True),c['layers']).to(c['device'])
            observe=ReceiverStates(relay,torch.load(c['receiver_projection'],weights_only=True).to(c['device']))
            report_key=secrets.token_hex(16);write_json(run.path/'private_report_rng.json',{'key':report_key})
            pop=list(examples(c,c['split']));ids=fixed_hash_subset([e.meta['group_id'] for e in pop],c['groups']);lookup={e.meta['group_id']:e for e in pop}
            if c['split']=='test':
                frozen=json.loads((root/'minimum_ids.json').read_text())['group_ids'];assert ids==frozen
            selected=[gid for i,gid in enumerate(ids) if i%c.get('partitions',1)==c.get('partition',0)]
            with (run.path/'results.jsonl').open('x') as handle:
                for index,gid in enumerate(selected):
                    started=time.monotonic();cost.check();b=short_example(lookup[gid]);e=direct_example(b);cf=direct_example(counterfactual(b));cohort=relay.encode([e,cf]);cost.forward_calls+=1;base=one(cohort,0)
                    md=record_metadata(e,relay.s2.tokenizer);md['physical_offset']=base.length-md['token_count'];prep=direct_prepared(relay,e,base);entry={'id':e.id,'group_id':gid,'commitment':commitment(e.label)}
                    with torch.no_grad():cleanlp=direct_receiver(relay,base,prep,cost);clean_pred=int(cleanlp.argmax())
                    def solve(eps,design=None,seed=None):
                        candidates=[]
                        for lam in (c['lambdas'] if design else [0.]):
                            attack=Perturbation(base,md,'E3',c['layers']);selection=None;hit=False
                            for step in range(1,c['steps']+1):
                                cost.check();attack.leaf();pfx=attack.build()
                                if design=='R4':lp,state=observe(pfx,prep,cost)
                                else:lp=direct_receiver(relay,pfx,prep,cost);state=None
                                obj=lp[cf.label]
                                if design:
                                    audit=audits[design,seed];seq=state if design=='R4' else transform(pfx,pfx.mask,md)
                                    obj=obj-lam*audit.loss(seq,audit.prepare(entry,rng_seed(e.id,design,seed,'minimum-EOT',step),16))
                                gradients=torch.autograd.grad(obj,list(attack.delta.values()));attack.step(gradients,eps,c['steps'],c['step_multiplier']);del obj,lp,pfx,gradients,state
                                with torch.no_grad():
                                    probe=attack.build(False)
                                    if design=='R4':plp,state=observe(probe,prep,cost)
                                    else:plp=direct_receiver(relay,probe,prep,cost);state=None
                                    hit=int(plp.argmax())==cf.label
                                    if design and (hit or step==c['steps']):
                                        seq=state if design=='R4' else transform(probe,probe.mask,md)
                                        selection=sum(float(audit.loss(seq,audit.prepare(entry,rng_seed(e.id,design,seed,'minimum-selection',j),16))) for j in range(c['selection_panels']))/c['selection_panels']
                                        hit=hit and selection<=audit.cal['thresholds']['16']['0.05']
                                del probe,plp,state
                                if hit and c['early_success']:break
                            with torch.no_grad():
                                pfx=attack.build(False);lp=direct_receiver(relay,pfx,prep,cost);target=int(lp.argmax())==cf.label
                                ratio=max(attack.ratios(pfx).values());assert ratio<=eps*(1+1e-5)
                            candidates.append((not target,selection if selection is not None else float('inf'),-float(lp[cf.label]),lam,attack))
                        chosen=min(candidates,key=lambda x:x[:3]);attack=chosen[-1]
                        with torch.no_grad():
                            pfx=attack.build(False)
                            if design=='R4':lp,state=observe(pfx,prep,cost)
                            else:lp=direct_receiver(relay,pfx,prep,cost);state=None
                            target=int(lp.argmax())==cf.label;score=None
                            if design:
                                seq=state if design=='R4' else transform(pfx,pfx.mask,md);audit=audits[design,seed]
                                # This check is part of the attacker search, not the held-out report panel.
                                score=sum(float(audit.loss(seq,audit.prepare(entry,rng_seed(e.id,design,seed,'minimum-selection',j),16))) for j in range(c['selection_panels']))/c['selection_panels']
                            success=target and (not design or score<=audit.cal['thresholds']['16']['0.05'])
                            heldout=None
                            if design:
                                heldout=float(audit.loss(seq,audit.prepare(entry,rng_seed(report_key,e.id,design,seed),16)))
                            norm=max(attack.ratios(pfx).values())
                        del candidates,attack,pfx,lp,state
                        return {'success':success,'lambda':chosen[3],'selection_nll':score,'report_nll':heldout,
                                'report_detected_5':None if design is None else heldout>audit.cal['thresholds']['16']['0.05'],
                                'report_detected_1':None if design is None else heldout>audit.cal['thresholds']['16']['0.01'],
                                'max_relative_norm':norm}
                    for design in [None,*c['designs']]:
                        for seed in ([None] if design is None else c['reader_seeds']):
                            if clean_pred==cf.label:
                                row={'group_id':gid,'design':design or 'receiver','seed':seed,'already_target':True,'lower':0.,'upper':0.,'censored':False}
                            else:
                                low=0.;high=c['epsilon_max'];trace=[];hit=False;best=None
                                for bound in c.get('bracket_epsilons',[c['epsilon_max']]):
                                    result=solve(bound,design,seed);trace.append({'epsilon':bound,**result})
                                    if result['success']:high=bound;best=result;hit=True;break
                                    low=bound
                                if hit:
                                    for j in range(c['binary_steps']):
                                        if low>0 and (high-low)/high<=c.get('relative_tolerance',0.):break
                                        mid=(low+high)/2;result=solve(mid,design,seed);trace.append({'epsilon':mid,**result})
                                        if result['success']:high=mid;best=result
                                        else:low=mid
                                row={'group_id':gid,'design':design or 'receiver','seed':seed,'already_target':False,'lower':low if hit else c['epsilon_max'],'upper':high if hit else None,'censored':not hit,'best':best if hit else None,'trace':trace,
                                     'interpretation':'finite PGD search boundary; failures are not certified lower bounds'}
                            handle.write(json.dumps(row)+'\n');handle.flush();cost.update(status='binary_search',groups_completed=index)
                    print('group',index+1,len(selected),'seconds',round(time.monotonic()-started,2),flush=True)
                    with (run.path/'timings.jsonl').open('a') as timing:timing.write(json.dumps({'group_index':index+1,'group_id':gid,'seconds':time.monotonic()-started})+'\n')
                    live(stage='minimum_'+c['mode'],run=str(run.path),groups_completed=index+1)
            write_json(root/f'{c["mode"]}_part{c.get("partition",0)}.json',{'run':str(run.path),'groups':len(selected),'seconds':time.monotonic()-cost.start,'test_opened':cost.test_opened});cost.update(status='completed')
        except BaseException:cost.update(status='failed');live(stage='minimum_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Locked raw-KV robustness evaluation for a message-dependent receiver."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time,secrets
from pathlib import Path
import torch
from clr.audit.m8c_runtime import snapshot,Cost,live
from clr.audit.m8c_pgd import optimize
from clr.audit.m8c_reader import load_audits,commitment,questions,rng_seed,qa_rows
from clr.audit.m8c_core import direct_example,short_example,direct_prepared,direct_receiver
from clr.audit.m8b_features import ReceiverStates,as_source
from clr.audit.features import KVFeatures,record_metadata,sha256
from clr.audit.m8a_perturb import Perturbation,feature_difference
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
            if torch.cuda.mem_get_info()[0]/2**30<c['min_free_GiB']:raise RuntimeError('M8c attack GPU memory precheck failed')
            if c['split']=='test':
                lock=json.loads((root/'final_lock.json').read_text())
                for p,h in lock['sha256'].items():
                    if sha256(p)!=h:raise ValueError('M8c lock changed: '+p)
                cost.test_opened=True
                cost.update(status='opening_locked_test');live(test_opened=True,background_tasks_active=True)
            relay=KVRelay(c);audits=load_audits(c)
            transform=KVFeatures(torch.load('data/M71_v1/projection.pt',weights_only=True),c['layers']).to(c['device'])
            observe=ReceiverStates(relay,torch.load(c['receiver_projection'],weights_only=True).to(c['device']))
            private=secrets.token_hex(16);write_json(run.path/'private_report_rng.json',{'key':private,'availability':'not exposed to attack, saved for reproduction'})
            pop=list(examples(c,c['split']));ids=fixed_hash_subset([e.meta['group_id'] for e in pop],c['groups']);lookup={e.meta['group_id']:e for e in pop}
            if c['split']=='test':
                frozen=json.loads((root/c.get('id_manifest','test_ids.json')).read_text())['group_ids']
                assert ids==frozen and len(ids)==c['groups']
            selected=[gid for i,gid in enumerate(ids) if i%c.get('partitions',1)==c.get('partition',0)]
            if c.get('replay_check',False):
                from clr.audit.m71_storage import load_bundle
                replay_entries={r['group_id']:r for r in map(json.loads,(root/'source'/'reader_val'/'manifest.jsonl').read_text().splitlines())}
                replay=[]
            with (run.path/'results.jsonl').open('x') as handle:
                for index,gid in enumerate(selected):
                    started=time.monotonic();cost.check();base0=short_example(lookup[gid]);e=direct_example(base0);cf=direct_example(counterfactual(base0))
                    cohort=relay.encode([e,cf]);cost.forward_calls+=1;base=one(cohort,0)
                    md=record_metadata(e,relay.s2.tokenizer);md['physical_offset']=base.length-md['token_count'];prep=direct_prepared(relay,e,base)
                    entry={'id':e.id,'group_id':gid,'commitment':commitment(e.label)}
                    @torch.no_grad()
                    def report(prefix,tag,only=None,cached=None):
                        if cached is None:lp,state=observe(prefix,prep,cost)
                        else:lp,state=cached
                        d0=transform(prefix,prefix.mask,md);out={}
                        for (d,s),audit in audits.items():
                            if only and (d,s)!=only:continue
                            seq=state if d=='R4' else d0
                            score=float(audit.loss(seq,audit.prepare(entry,rng_seed(private,s,e.id,tag),16)))
                            out[f'{d}:{s}']={'score':score,**{str(a):score>audit.cal['thresholds']['16'][str(a)] for a in [.01,.05]}}
                        return lp,out
                    with torch.no_grad():
                        cleanlp,cleanstate=observe(base,prep,cost);clean_feature=transform(base,base.mask,md)
                        _,clean_audit=report(base,'clean',cached=(cleanlp,cleanstate));clean_pred=int(cleanlp.argmax());qa={}
                        if c.get('replay_check',False):
                            saved=load_bundle(replay_entries[gid]['file'])['messages'][0];row={'group_id':gid,'receiver_pred_equal':clean_pred==replay_entries[gid]['receiver_pred'][0]}
                            for d,seq in [('D0',clean_feature),('R4',cleanstate)]:
                                err=(seq.values.half()-saved[d]['values'].to(c['device'])).abs()
                                row[d]={'exact_fraction':int((err==0).sum())/err.numel(),'max_absolute':float(err.max())}
                                if not bool((err==0).all()):raise ValueError('M8c saved/live source replay mismatch: '+str(row))
                            assert row['receiver_pred_equal'];replay.append(row);write_json(run.path/'replay.json',replay)
                        for (d,s),audit in audits.items():
                            seq=cleanstate if d=='R4' else clean_feature;src=as_source(seq);src['values']=src['values'].half();rr=qa_rows(entry,e.label)
                            qa[f'{d}:{s}']={}
                            for task in ['A1','A4']:
                                rows=[r for r in rr if r['task']==task]
                                with torch.autocast('cuda',dtype=torch.bfloat16):ll=audit.model(audit.serializer.batch(rows,c['device']),{k:v.expand(len(rows),*v.shape[1:]) for k,v in src.items()},task)
                                qa[f'{d}:{s}'][task]=[int(p==r['label']) for p,r in zip(ll.argmax(-1).tolist(),rows)]
                    common={'id':e.id,'group_id':gid,'clean_pred':clean_pred,'clean_correct':clean_pred==e.label,'target':cf.label,'clean_audit':clean_audit,'qa':qa}
                    for kind in c['kinds']:
                        for eps in c['epsilons']:
                            trials=[(d,s,l) for d in c['designs'] for s in c['reader_seeds'] for l in c['lambdas']] if kind=='E3' else [(None,None,0.)]
                            for design,seed,lam in trials:
                                attack,step,selection,stopped,plp,pstate=optimize(base,md,prep,entry,cf.label,kind,eps,design,seed,lam,relay,transform,observe,audits,cost,c,clean_feature)
                                with torch.no_grad():
                                    final=attack.build(False);ratios=attack.ratios(final);maximum=max(ratios.values());assert maximum<=eps*(1+1e-5)
                                    lp,audit=report(final,f'{kind}:{eps}:{design}:{seed}:{lam}',only=(design,seed) if kind=='E3' else None,cached=(plp,pstate) if kind=='E3' else None)
                                    feat=transform(final,final.mask,md);exact=bool((feat.values.half()==clean_feature.values.half()).all())
                                    if kind=='E2':assert exact
                                    row={**common,'kind':kind,'epsilon':eps,'attack_design':design,'attack_seed':seed,'lambda':lam,'steps':step,'step_budget':c['steps'],'early_success':stopped,'selection_nll':selection,'pred':int(lp.argmax()),'target_logp':float(lp[cf.label]),'target_achieved':int(lp.argmax())==cf.label,'changed_to_target':int(lp.argmax())==cf.label and clean_pred!=cf.label,'audit':audit,'max_relative_norm':maximum,'original_f_fp16_equal':exact}
                                    handle.write(json.dumps(row)+'\n');handle.flush()
                                del attack,final,feat,lp,plp,pstate;cost.update(status='evaluating',groups_completed=index,kind=kind);cost.check()
                    print('group',index+1,len(selected),'seconds',round(time.monotonic()-started,2),'GPUh',round((time.monotonic()-cost.start)/3600,3),flush=True)
                    with (run.path/'timings.jsonl').open('a') as timing:
                        timing.write(json.dumps({'group_index':index+1,'group_id':gid,'seconds':time.monotonic()-started})+'\n')
                    cost.update(status='evaluating',groups_completed=index+1)
                    live(stage='evaluate_'+c['mode'],run=str(run.path),groups_completed=index+1,test_opened=cost.test_opened)
                    del base,cohort,cleanstate,clean_feature
            write_json(root/f'{c["mode"]}_part{c.get("partition",0)}.json',{'run':str(run.path),'groups':len(selected),'seconds':time.monotonic()-cost.start,'test_opened':cost.test_opened})
            cost.update(status='completed');live(stage='evaluate_'+c['mode']+'_completed',run=str(run.path))
        except BaseException:cost.update(status='failed');live(stage='evaluation_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

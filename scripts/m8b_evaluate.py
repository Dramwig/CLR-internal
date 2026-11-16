"""Design-adaptive raw-KV PGD; report randomness never enters optimization."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time,secrets
from pathlib import Path
import torch
from transformers import AutoTokenizer
from clr.audit.features import KVFeatures,record_metadata,sha256
from clr.audit.m8b_features import RandomKV,ReceiverStates,as_source
from clr.audit.m8b_reader import new_model,Audit,rng_seed,make_row,public_questions
from clr.audit.m71_reader import Serializer
from clr.audit.m8a_perturb import Perturbation,feature_difference
from clr.audit.m8b_runtime import Cost,snapshot,live
from clr.data.m71 import examples,fixed_hash_subset
from clr.data.m7 import format_commitment
from clr.sources.kv_relay import KVRelay
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.records import recorded
from clr.utils.config import load_config
from scripts.m8a_evaluate import one,prepared_receiver,receiver

class Panel:
    def __init__(self,c,relay):
        self.c=c;self.relay=relay;self.audits={};self.report_key=secrets.token_hex(16);self.serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
        p=torch.load('data/M71_v1/projection.pt',weights_only=True).to(c['device'])
        self.old=KVFeatures(p,c['layers']).to(c['device']);self.random=RandomKV(p[:,:128])
        self.observe=ReceiverStates(relay,torch.load(Path(c['output'])/'receiver_projection.pt',weights_only=True).to(c['device']))
        root=Path(c['output'])
        for d in c['designs']:
            for s in c['reader_seeds']:
                ref=json.loads((root/('smoke_training' if c.get('smoke_models',False) else c.get('training_folder','training'))/f'{d}_{s}.json').read_text()) if d!='R1' else None
                model=new_model(s,d,c['device'],ref['checkpoint'] if ref else None).eval().requires_grad_(False)
                if c.get('smoke_models',False):
                    cal={'temperatures':{t:1. for t in ['A1','A2','A3','A4']},'thresholds':{str(k):{'0.01':10.,'0.05':5.} for k in [1,4,16,64]}}
                else:cal=json.loads((root/c.get('calibration_folder','calibration')/f'{d}_{s}.json').read_text())
                self.audits[d,s]=Audit(c,self.serializer,d,s,model,cal)
    def feature(self,d,prefix,meta,draw,state=None):
        if d=='R4':return state
        if d=='R1':return self.old(prefix,prefix.mask,meta)
        return self.random(prefix,meta,draw,d)
    @torch.no_grad()
    def report(self,prefix,meta,prepared,entry,cost,tag,only=None):
        logp,state=self.observe(prefix,prepared,cost);out={};deterministic={}
        for (d,s),audit in self.audits.items():
            if only is not None and (d,s)!=only:continue
            if d in ['R1','R3'] and d in deterministic:seq=deterministic[d]
            else:
                seq=self.feature(d,prefix,meta,rng_seed(self.report_key,entry['id'],d,s,'report_read',tag),state)
                if d in ['R1','R3']:deterministic[d]=seq
            ks=[1,4,16,64] if d=='R1' else [16]
            rr=public_questions(entry,rng_seed(self.report_key,s,'report',entry['id']),max(ks),d=='R1' and not self.c.get('matched_questions',False))
            loss=audit.ordered_losses(seq,rr)
            for k in ks:
                score=float(loss[:k].mean())
                out[f'{d}:{s}:{k}']={'score':score,**{str(a):score>audit.cal['thresholds'][str(k)][str(a)] for a in [.01,.05]}}
        return logp,out
    @torch.no_grad()
    def qa(self,prefix,meta,prepared,entry,cost,state):
        out={}
        for (d,s),audit in self.audits.items():
            seq=self.feature(d,prefix,meta,rng_seed(entry['id'],d,s,'QA'),state);v=entry['private_value']
            rows=[make_row(entry,v,'A1',4,v)]+[make_row(entry,v,'A1',4,(v+j)%4) for j in [1,2,3]]+[make_row(entry,v,'A4',4,None,rng_seed(entry['id'],'options'))]
            source={k:z.to(self.c['device']) for k,z in as_source(seq).items()}
            pred={}
            for t in ['A1','A4']:
                rr=[r for r in rows if r['task']==t]
                with torch.autocast('cuda',dtype=torch.bfloat16):ll=audit.model(self.serializer.batch(rr,self.c['device']),{k:z.expand(len(rr),*z.shape[1:]) for k,z in source.items()},t)
                pred[t]=[int(i==r['label']) for i,r in zip(ll.argmax(-1).tolist(),rr)]
                if t=='A1':pred[t]=pred[t][:1]*3+pred[t][1:]
            out[f'{d}:{s}']=pred
        return out

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output'])
            if c['split']=='test':
                lock=json.loads((root/'final_lock.json').read_text())
                for p,h in lock['sha256'].items():
                    if sha256(p)!=h:raise ValueError('Final lock changed: '+p)
                cost.test_opened=True;live(stage='final_test_opened',test_opened=True,run=str(run.path))
            if torch.cuda.mem_get_info()[0]/2**30<c['min_free_GiB']:raise RuntimeError('GPU memory precheck')
            relay=KVRelay(c);panel=Panel(c,relay)
            write_json(run.path/'private_report_rng.json',{'key':panel.report_key,'availability':'held out from the optimization interface, retained after run for reproduction'})
            pop=list(examples(c,c['split']));ids=fixed_hash_subset([e.meta['group_id'] for e in pop],c['groups']);lookup={e.meta['group_id']:e for e in pop}
            replay_rows=[]
            if c.get('replay_check',False):
                replay_new={r['id']:r for r in map(json.loads,(root/'source'/'reader_val'/'manifest.jsonl').read_text().splitlines())}
                replay_old={r['id']:r for r in map(json.loads,Path('data/M71_v1/pool/source/reader_val/manifest.jsonl').read_text().splitlines())}
            if c['split']=='test':assert ids==json.loads(Path('data/M8a_v1/test500_ids.json').read_text())['group_ids']
            selected=[lookup[g] for i,g in enumerate(ids) if i%c.get('partitions',1)==c.get('partition',0)]
            with (run.path/'results.jsonl').open('x') as handle:
                for index,e in enumerate(selected):
                    started=time.monotonic();cost.check();from clr.data.m7 import counterfactual
                    cohort=relay.encode([e,counterfactual(e)]);base=one(cohort,0);cost.forward_calls+=1
                    meta=record_metadata(e,relay.s2.tokenizer);meta['physical_offset']=base.length-meta['token_count']
                    prepared=prepared_receiver(relay,e,base)
                    from clr.audit.m8_p0 import public_constraints
                    pub=public_constraints(e.x_pub);entry={'id':e.id,'group_id':e.meta['group_id'],'commitment':format_commitment(e.id,e.label),**pub}
                    target=e.meta['T'][e.meta['z_pub']][(e.meta['e_priv']+1)%4]
                    with torch.no_grad():
                        clean_lp,clean_state=panel.observe(base,prepared,cost);clean_pred=int(clean_lp.argmax())
                        _,clean=panel.report(base,meta,prepared,entry,cost,'clean')
                        qa=panel.qa(base,meta,prepared,{**entry,'private_value':e.meta['e_priv']},cost,clean_state)
                        clean_feature=panel.old(base,base.mask,meta)
                        if c.get('replay_check',False):
                            from clr.audit.m71_storage import load_bundle
                            from clr.audit.m8b_features import cached_sequence,cache_on_device
                            saved=load_bundle(replay_new[e.id]['file'])['messages'][0]
                            tests={'R1':(clean_feature.values,load_bundle(replay_old[e.id]['file'])['messages'][0]['values']),
                                   'R4':(clean_state.values,saved['R4']['values'])}
                            for d in ['R2','R3']:
                                live_seq=panel.feature(d,base,meta,37,clean_state)
                                tests[d]=(live_seq.values,cached_sequence(cache_on_device(saved['KV'],c['device']),37,d).values)
                            rr={'id':e.id}
                            for d,(current,stored) in tests.items():
                                error=(current.half()-stored.to(current.device)).abs()
                                rr[d]={'exact_fraction':int((error==0).sum())/error.numel(),'max_absolute':float(error.max())}
                                if rr[d]['exact_fraction']!=1.:raise ValueError('Source replay differs on matched device: '+d+' '+str(rr[d]))
                            replay_rows.append(rr);write_json(run.path/'replay.json',replay_rows)
                    common={'id':e.id,'group_id':e.meta['group_id'],'target':target,'clean_pred':clean_pred,'clean_correct':clean_pred==e.label,'clean_audit':clean,'qa':qa}
                    def emit(attack,kind,eps,design,seed,lam,step,budgets,early,selection_score):
                        with torch.no_grad():
                            changed=attack.build(False);ratios=attack.ratios(changed);maximum=max(ratios.values())
                            if eps is not None and maximum>eps*(1+1e-5):raise ValueError('Actual BF16 KV constraint violated')
                            lp,aud=panel.report(changed,meta,prepared,entry,cost,f'{kind}_{eps}_{design}_{seed}_{lam}_{step}',only=(design,seed) if kind=='E3' else None)
                            actual_feature=panel.old(changed,changed.mask,meta)
                            diff=feature_difference(clean_feature,actual_feature)
                            exact=(clean_feature.values.half()==actual_feature.values.half())
                            diff['fp16_exact_fraction']=int(exact.sum())/exact.numel()
                            diff['fp16_bitwise_equal']=bool(exact.all())
                            if kind=='E2' and diff['fp16_exact_fraction']!=1.:raise ValueError('Original E2 feature identity violated')
                            row={**common,'kind':kind,'epsilon':eps,'attack_design':design,'attack_seed':seed,'lambda':lam,'steps':step,'step_budget':None,'early_success':early,'selection_nll':selection_score,'pred':int(lp.argmax()),'target_logp':float(lp[target]),'target_achieved':int(lp.argmax())==target,'changed_to_target':int(lp.argmax())==target and clean_pred!=target,'audit':aud,'max_relative_norm':maximum,'layer_norms':ratios,'original_feature_difference':diff}
                            for budget in budgets:handle.write(json.dumps({**row,'step_budget':budget})+'\n')
                            handle.flush()
                    for kind in c['kinds']:
                        for eps in c['epsilons']:
                            trials=[(d,s,lam) for d in c['designs'] for s in c['reader_seeds'] for lam in c['lambdas']] if kind=='E3' else [(None,None,0.)]
                            for design,seed,lam in trials:
                                cost.check();attack=Perturbation(base,meta,kind,c['layers']);maxsteps=max(c['steps']);stopped=False;selection_score=None
                                for step in range(1,maxsteps+1):
                                    attack.leaf();changed=attack.build()
                                    if kind=='E3' and design=='R4':lp,state=panel.observe(changed,prepared,cost)
                                    else:lp=receiver(relay,changed,prepared,cost);state=None
                                    objective=lp[target]
                                    if kind=='E3':
                                        auditor=panel.audits[design,seed];draw=rng_seed(e.id,design,seed,'opt',step)
                                        seq=panel.feature(design,changed,meta,draw,state)
                                        q=auditor.prepare(entry,rng_seed(e.id,design,seed,'opt_questions',step),c['attack_k'] if design=='R1' else c['eot_k'])
                                        objective=objective-lam*auditor.loss(seq,q)
                                    grads=torch.autograd.grad(objective,list(attack.delta.values()));attack.step(grads,eps,maxsteps,c['step_multiplier'])
                                    del objective,lp,changed,grads,state
                                    if kind=='E2':attack.enforce_fp16_identity(panel.old,meta,clean_feature)
                                    with torch.no_grad():
                                        probe=attack.build(False)
                                        if kind=='E3' and design=='R4':plp,pstate=panel.observe(probe,prepared,cost)
                                        else:plp=receiver(relay,probe,prepared,cost);pstate=None
                                        hit=int(plp.argmax())==target
                                        if kind=='E3' and (hit or step in c['steps']):
                                            losses=[]
                                            for j in range(c['selection_panels']):
                                                seq=panel.feature(design,probe,meta,rng_seed(e.id,design,seed,'selection_read',j),pstate)
                                                q=auditor.prepare(entry,rng_seed(e.id,design,seed,'selection_questions',j),c['attack_k'] if design=='R1' else c['eot_k'])
                                                losses.append(float(auditor.loss(seq,q)))
                                            selection_score=sum(losses)/len(losses)
                                            hit=hit and selection_score<=min(auditor.cal['thresholds']['16'].values())
                                        stopped=hit and c['early_success']
                                    del probe,plp,pstate
                                    if stopped or step in c['steps']:
                                        budgets=[n for n in c['steps'] if n>=step] if stopped else [step]
                                        emit(attack,kind,eps,design,seed,lam,step,budgets,stopped,selection_score)
                                    if stopped:break
                                del attack
                                cost.update(status='evaluating',groups_completed=index,kind=kind);cost.check()
                    print('group',index+1,len(selected),'seconds',round(time.monotonic()-started,2),'GPUh',round((time.monotonic()-cost.start)/3600,3),flush=True)
                    live(stage='evaluate_'+c['mode'],run=str(run.path),groups_completed=index+1,test_opened=cost.test_opened);del base,cohort,clean_feature,clean_state
            write_json(root/f'{c["mode"]}_part{c.get("partition",0)}.json',{'run':str(run.path),'groups':len(selected),'test_opened':cost.test_opened,'seconds':time.monotonic()-cost.start})
            cost.update(status='completed',peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30);live(stage='evaluate_'+c['mode']+'_completed',run=str(run.path))
        except Exception:
            cost.update(status='failed');live(stage='evaluation_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

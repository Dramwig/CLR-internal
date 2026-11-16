"""M6 B: val-only smoke/noise selection and immutable LLaMA M1-format dumps."""
import argparse, hashlib, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything, write_json
from clr.utils.m6_runtime import Budget, live, hours
from clr.sources.codi_m6 import LlamaM6, prepare, rollout, early_exit
from clr.data.labels import numeric_answer
from clr.data.m1 import rng_seed
from clr.data.loader import ShardWriter

def contexts(rows,tokenizer,ph):
    groups=defaultdict(list)
    for r in rows: groups[r['cohort_id']].append(r)
    for group in groups.values():
        lengths=[len(tokenizer.encode(r['question'].strip().replace('  ',' '))) for r in group]; width=max(lengths)
        for r,n in zip(group,lengths):
            r.update(gpt2_pad_to=r['pad_to'],pad_to=width,left_padding=width-n,protocol_sha256=ph)
    return rows

def proxy_rows(c,split,tok):
    # Read only permitted split; original cohort membership retained before subsampling.
    rows=pq.read_table(Path(c['gpt2_root'])/split/'manifest.parquet',filters=[('kind','=','clean'),('t','=',0),('layer','=',12)]).to_pylist()
    rs=[{'id':r['problem_id'],'question':r['question'],'answer':r['gold_answer'],'cot':r['gold_cot'],
         **{k:r[k] for k in ['intermediate_values','cot_parse_failed','cohort_id','pad_to','left_padding','source_distribution']}} for r in rows]
    return contexts(rs,tok,hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest())

def seeds(c,pid,kind,t=0,K=16):
    return [rng_seed(c['seed'],pid,kind,t,k) for k in range(K)]

def trial_noise(source,row,c):
    pre=prepare(row,source); gold=numeric_answer(row['answer']); clean=early_exit(row,pre,6)
    result=[]
    for t in range(6):
        for sigma in c['sigmas']:
            out=rollout(row,pre,t,sigma,0,seeds(c,row['id'],'noise',t))
            ok=[numeric_answer(o['answer'])==gold for o in out]
            result.append({'problem_id':row['id'],'t':t,'sigma':sigma,'clean_correct':numeric_answer(clean['answer'])==gold,
                'correct':ok,'answers':[o['answer'] for o in out],'value':sum(ok)/16,'nondegenerate':0<sum(ok)<16})
    return result

def dump_one(source,row,c,split,writer):
    ts=time.monotonic(); pre=prepare(row,source); gold=numeric_answer(row['answer'])
    early=[early_exit(row,pre,t) for t in range(7)]; answer=early[-1]['answer']; layer=len(pre.layers[0])-1
    meta={'problem_id':row['id'],'source':'codi_llama1b','split':split,'question':row['question'],'gold_answer':row['answer'],'gold_cot':row['cot'],
        **{k:row[k] for k in ['intermediate_values','cot_parse_failed','cohort_id','pad_to','left_padding','protocol_sha256','source_distribution']},
        'model_answer':answer,'correct':numeric_answer(answer)==gold,'seed':0,'state_definition':'post-forward observation z_t; feedback stored separately'}
    state=torch.stack([z[0,0].cpu() for z in pre.states])[:,None,:]
    writer.add(state,{**meta,'kind':'clean','traj_id':-1,'t0':0,'label':meta['correct'],'steps':list(range(7)),'layers':[layer]})
    mc=[]
    if split.startswith('reader'):
        groups=defaultdict(list)
        for k in range(8):
            sd=rng_seed(c['seeds'][k%3],row['id'],'traj',k=k); groups[random.Random(sd).randrange(6)].append((k,sd))
        for t0,items in groups.items():
            outs=rollout(row,pre,t0,c['sigma'],0,[sd for _,sd in items])
            for (k,sd),o in zip(items,outs):
                correct=numeric_answer(o['answer'])==gold
                writer.add(o['states'][:,None,:],{**meta,'kind':'traj','traj_id':k,'t0':t0,'label':correct,'correct':correct,'model_answer':o['answer'],'seed':sd,'steps':list(range(t0,7)),'layers':[layer]},o['feedback'])
    else:
        for t in range(6):
            for policy,sigma,temp in [('pi',c['sigma'],0),('pi_T',0,1.)]:
                ss=[rng_seed(c['seeds'][k%3],row['id'],policy,t,k) for k in range(16)]; outs=rollout(row,pre,t,sigma,temp,ss)
                correct=[numeric_answer(o['answer'])==gold for o in outs]
                mc.append({'problem_id':row['id'],'split':split,'t':t,'policy':policy,'K':16,'seeds':ss,'answers':[o['answer'] for o in outs],'correct':correct,'value':sum(correct)/16,'invalid_rate':sum(numeric_answer(o['answer']) is None for o in outs)/16})
    er=[{'problem_id':row['id'],'split':split,'t':t,'answer':o['answer'],'correct':numeric_answer(o['answer'])==gold} for t,o in enumerate(early)]
    return er,mc,time.monotonic()-ts

def main(c):
 with recorded(c) as run:
    print(run.path,flush=True); start=time.monotonic(); budget=Budget(run,c,start); source=None;load_attempted=False
    try:
        torch.set_num_threads(2); seed_everything(c['seed']); protocol=load_config(c['protocol'])
        if c['mode']!='smoke':
            g=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
            if not g.get('gate_passed'): raise RuntimeError('M6 source smoke required')
        load_attempted=True;source=LlamaM6({**protocol['source'],'device':c['device'],'min_free_gib':c['min_free_gib']}); budget.check()
        val=proxy_rows(c,'val',source.tokenizer)
        if c['stage']=='smoke':
            val=val[:c['smoke_val']]; train=proxy_rows(c,'reader_train',source.tokenizer)[:c['smoke_train']]
            # Compare unchanged audited full run to exact continuation batch1.
            p=val[0]; old=source.run(p); pre=prepare(p,source); end=early_exit(p,pre,6)
            diffs=[float((old['latent_steps'][j]-pre.states[j][0,0].cpu()).abs().max()) for j in range(6)]
            if old['answer']!=end['answer'] or max(diffs)!=0: raise RuntimeError('Audited LLaMA clean continuation mismatch')
            for t in [0,3,5]:
                out=rollout(p,pre,t,0,0,11)
                if out['answer']!=end['answer']: raise RuntimeError('Zero-noise continuation mismatch')
            write_json(run.path/'continuation_equivalence.json',{'max_state_differences':diffs,'same_answer':True,'immutable_cache_lengths':[x[0][0].shape[-2] for x in pre.caches]})
            noise=[]; timings=defaultdict(list)
            for row in val:
                ts=time.monotonic(); noise+=trial_noise(source,row,c); timings['noise'].append(time.monotonic()-ts); budget.check()
            # Smoke generated state data is val-only proxy; training traces from train.
            for split,rows in [('reader_train',train),('val',val)]:
                writer=ShardWriter(run.path/'dumps'/split); ers=[]; mcs=[]
                for row in rows:
                    er,mc,seconds=dump_one(source,row,c,split,writer); ers+=er;mcs+=mc;timings[split].append(seconds); budget.check()
                writer.close(); pq.write_table(pa.Table.from_pylist(ers),run.path/'dumps'/split/'early_exit.parquet')
                if mcs:pq.write_table(pa.Table.from_pylist(mcs),run.path/'dumps'/split/'mc.parquet')
            write_json(run.path/'noise.json',noise);write_json(run.path/'timing.json',dict(timings))
            run.write_metrics(status='completed',gate_passed=True,questions=len(val)+len(train),timing=dict(timings),test_opened=False)
            live(stage='B_smoke_passed',source_smoke=str(run.path),milestone_complete=False)
        elif c['stage']=='noise':
            result=[]
            for i,row in enumerate(val):
                result+=trial_noise(source,row,c);pq.write_table(pa.Table.from_pylist(result[-24:]),run.path/f'noise_{i:03d}.parquet')
                budget.check();write_json(run.path/'progress.json',{'completed':i+1,'total':len(val),'own_gpu_hours':(time.monotonic()-start)/3600})
                print(f'noise {i+1}/{len(val)}',flush=True)
            summary=[]
            for sigma in c['sigmas']:
                rs=[r for r in result if r['sigma']==sigma];acc=float(np.mean([r['value'] for r in rs])); clean=float(np.mean([r['clean_correct'] for r in rs]));nondeg=float(np.mean([r['nondegenerate'] for r in rs]))
                summary.append({'sigma':sigma,'accuracy':acc,'clean_accuracy':clean,'drop':clean-acc,'nondegenerate':nondeg,'eligible':clean-acc<.05 and nondeg>=.30})
            eligible=[r for r in summary if r['eligible']]
            if c.get('selection_deferred'):
                write_json(run.path/'scan_summary.json',{'summary':summary,'selection_deferred':True,'unit':'problem x t, K16; equal weight t=0..5','seed':c['seed']})
                run.write_metrics(status='completed',test_opened=False,selection_deferred=True)
                live(stage='B_noise_extension_completed',noise_extension_run=str(run.path))
                return
            selected=next((r for r in eligible if r['sigma']==1.5),None) or (max(eligible,key=lambda r:(r['nondegenerate'],-r['drop'],-r['sigma'])) if eligible else None)
            write_json(run.path/'selection.json',{'summary':summary,'selected':selected,'unit':'problem x t, K16; equal weight t=0..5','seed':c['seed'],'preference':'sigma1.5 if eligible; else max nondegenerate with drop<5pp, nondeg>=30%'})
            if selected is None:raise RuntimeError('No LLaMA sigma passes M0.1 drop/nondegeneracy gates')
            run.write_metrics(status='completed',selected_sigma=selected['sigma'],test_opened=False)
            live(stage='B_noise_selected',noise_run=str(run.path),selected_sigma=selected['sigma'])
        elif c['stage']=='plan':
            # Executed only after val-only smoke and noise; no evaluation/test outputs.
            if not c.get('noise_run'):raise ValueError('Frozen noise selection required')
            sel=json.loads(Path(c['noise_run'],'selection.json').read_text())['selected'];c['sigma']=sel['sigma']
            if not sel['eligible'] and not(sel.get('degraded') and sel.get('authorized_selection_version')=='M6_B1_user_extension_v2'):raise ValueError('Unqualified noise requires the explicit user-authorized v2 fallback')
            plan=json.loads(Path(c['gpt2_plan']).read_text());flat=[r for rs in plan['splits'].values() for r in rs]
            contexts(flat,source.tokenizer,hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest())
            tm=json.loads(Path(c['smoke_run'],'timing.json').read_text());factor=c['cost_safety_factor']
            train_sec=float(np.mean(tm['reader_train']));held_sec=float(np.mean(tm['val']));used=sum(r['gpu_hours'] for r in hours() if r['section']=='B')
            fixed=1000*train_sec+1805*held_sec;available=max(0,30-used-fixed*factor/3600)
            n=min(10000,int(available*3600/(train_sec*factor)))
            if n<100:raise RuntimeError(f'Fixed heldout/reader_val data alone exceed B budget: {fixed*factor/3600:.3g}h')
            plan['splits']['reader_train']=plan['splits']['reader_train'][:n];plan['splits']['reader_val']=plan['splits']['reader_val'][:1000]
            plan['sigma']=c['sigma'];plan['noise_selection']={'run':c['noise_run'],'selected':sel};plan['cost_estimate']={'used_gpu_hours':used,'train_seconds_per_problem':train_sec,'held_seconds_per_problem':held_sec,'safety_factor':factor,'train_count':n,'projected_gpu_hours':used+factor*(fixed+n*train_sec)/3600}
            write_json(Path(c['output'])/'plan.json',plan);write_json(run.path/'estimate.json',plan['cost_estimate']);run.write_metrics(status='completed',test_evaluated=False)
            live(stage='B_plan_locked',plan=str(Path(c['output'])/'plan.json'),cost_estimate=plan['cost_estimate'])
        elif c['stage']=='generate':
            plan=json.loads(Path(c['output'],'plan.json').read_text()); c['sigma']=plan['sigma']
            write_json(run.path/'effective_policy.json',{'sigma':c['sigma'],'plan_sha256':hashlib.sha256(Path(c['output'],'plan.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest()})
            for split in c['splits']:
                rows=plan['splits'][split];outroot=Path(c['dump_root'])/split
                if (outroot/'completed.json').exists():continue
                outroot.mkdir(parents=True,exist_ok=True);manifests=[];ers=[];mcs=[];storages=[]
                shard_count=c.get('generation_shards',1);shard_index=c.get('generation_shard',0)
                if not 0<=shard_index<shard_count:raise ValueError('Invalid generation shard')
                for chunk_index,first in enumerate(range(0,len(rows),8)):
                    if chunk_index%shard_count!=shard_index:continue
                    chunk_rows=rows[first:first+8];marker=outroot/f'chunk_{first:05d}.json'
                    if marker.exists():
                        chunk=json.loads(marker.read_text())
                        if chunk['ids']!=[r['id'] for r in chunk_rows] or chunk['sigma']!=c['sigma'] or chunk['plan_sha256']!=hashlib.sha256(Path(c['output'],'plan.json').read_bytes()).hexdigest():raise ValueError('Resumed chunk protocol/IDs changed')
                    else:
                        # Each attempt is immutable; interrupted chunk attempts remain as evidence.
                        part=outroot/'parts'/f'{first:05d}_{run.path.name}';writer=ShardWriter(part);ce=[];cm=[]
                        for j,row in enumerate(chunk_rows):
                            er,mc,sec=dump_one(source,row,c,split,writer);ce+=er;cm+=mc;budget.check()
                            run.write_metrics(progress={'split':split,'completed':first+j+1,'total':len(rows)})
                            write_json(run.path/'progress.json',{'split':split,'completed':first+j+1,'total':len(rows),'seconds_last':sec,'gpu_hours':(time.monotonic()-start)/3600})
                            print(f'{split} {first+j+1}/{len(rows)} {sec:.3f}s',flush=True)
                        storage=writer.close();cr=pq.read_table(part/'manifest.parquet').to_pylist()
                        for r in cr:r['shard']=str(part.relative_to(outroot)/r['shard'])
                        chunk={'ids':[r['id'] for r in chunk_rows],'sigma':c['sigma'],'plan_sha256':hashlib.sha256(Path(c['output'],'plan.json').read_bytes()).hexdigest(),'manifest':cr,'early':ce,'mc':cm,'storage':storage,'run':str(run.path)}
                        write_json(marker,chunk)
                    manifests+=chunk['manifest'];ers+=chunk['early'];mcs+=chunk['mc'];storages.append(chunk['storage'])
                if shard_count>1:
                    write_json(outroot/f'shard_complete_{shard_index}.json',{'run':str(run.path),'shard':shard_index,'shards':shard_count,'sigma':c['sigma']})
                    continue
                pq.write_table(pa.Table.from_pylist(manifests),outroot/'manifest.parquet');pq.write_table(pa.Table.from_pylist(ers),outroot/'early_exit.parquet')
                if mcs:pq.write_table(pa.Table.from_pylist(mcs),outroot/'mc.parquet')
                write_json(outroot/'completed.json',{'run':str(run.path),'problem_count':len(rows),'storage':storages,'sigma':c['sigma']})
                live(stage='B_generating',last_completed_split=split,generation_run=str(run.path))
            run.write_metrics(status='completed',test_predictions_inspected=False)
        else:raise ValueError(c['stage'])
    finally:
        run.write_metrics(gpu_hours=(time.monotonic()-start)/3600 if load_attempted else 0,source_forward_calls=source.forward_calls if source else 0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

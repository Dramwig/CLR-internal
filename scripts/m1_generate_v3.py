"""Gated, immutable M1 generation. Runs clean/early/traj/MC; trains no reader."""
import argparse,json,time,random,hashlib,shutil
from pathlib import Path
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.sources.codi_v2 import load
from clr.policy import prepare,rollout,early_exit
from clr.data.labels import numeric_answer
from clr.data.m1 import rng_seed
from clr.data.loader import ShardWriter
from clr.data.resume import ResumedWriter, ensure_space
from clr.data.resume_v3 import link_completed, recover_test

def main(c):
 with recorded(c) as run:
    start=time.perf_counter();source=None
    try:
        if (c['trajectory_K'],c['mc_K'],c['variance_K'],c['sigma'])!=(8,16,64,1.5):raise ValueError('M1 fixed policy changed; requires a new audited version')
        resources=json.loads(Path(c['candidate_path']).with_name('resource_hashes.json').read_text())
        if hashlib.sha256(Path(resources['source_train']['path']).read_bytes()).hexdigest()!=resources['source_train']['sha256']:raise ValueError('Training corpus changed since audit')
        write_json(run.path/'resources.json',resources)
        gate=json.loads(Path(c['gate_run'],'metrics.json').read_text());budget=json.loads(Path(c['budget_run'],'metrics.json').read_text())
        if not gate.get('gate_passed') or budget['status']!='completed':raise ValueError('Gate and measured budget required')
        plan=json.loads(Path(c['candidate_path']).with_name('frozen.json').read_text());protocol=load_config(c['protocol'])
        if hashlib.sha256(Path(c['protocol']).read_bytes()).hexdigest()!=plan['protocol_sha256']:raise ValueError('Protocol changed')
        if hashlib.sha256(Path(c['split_path']).read_bytes()).hexdigest()!=budget['split_sha256']:raise ValueError('Frozen split changed')
        output=Path(c['output'])
        if output.exists():raise FileExistsError('Output already exists; never overwrite prior generation')
        output.mkdir(parents=True)
        ensure_space(output)
        old_run=Path(c['resume_run']);old_metrics=json.loads((old_run/'metrics.json').read_text())
        old_config=json.loads((old_run/'config.json').read_text())
        for key in ['protocol','split_path','seed','stochastic_seeds','sigma','trajectory_K','mc_K','variance_K']:
            if old_config[key]!=c[key]:raise ValueError('Resume protocol changed: '+key)
        old_output=Path(old_config['output'])
        ancestors=[];cursor=old_run
        while cursor:
            m=json.loads((cursor/'metrics.json').read_text());cfg=json.loads((cursor/'config.json').read_text())
            ancestors.append({'run':str(cursor),'gpu_hours':m.get('gpu_hours',0),'source_forward_calls':m.get('source_forward_calls',0)})
            cursor=Path(cfg['resume_run']) if cfg.get('resume_run') else None
        prior_hours=sum(r['gpu_hours'] for r in ancestors)
        write_json(run.path/'ancestors.json',ancestors)
        write_json(output/'provenance.json',{'run':str(run.path),'config':c,'protocol':protocol,'split_sha256':plan['split_sha256'],'generation_started':True})
        all_layers=set(plan['all_layer_ids']);torch.set_num_threads(c['threads']);seed_everything(c['seed'])
        source=load({**protocol['source'],'device':c['device'],'min_free_gib':c['min_free_gib']})
        held=[r for k,rs in plan['splits'].items() if k in ['val','cal','test'] for r in rs]
        pairs=sorted([(r['id'],t) for r in held for t in range(6)],key=lambda x:rng_seed(c['seed'],x[0],'variance_selection',x[1]))[:c['variance_pairs']]
        pairset=set(pairs);write_json(run.path/'variance_pairs.json',pairs)
        counts={};storages={};done=0
        for split,rows in plan['splits'].items():
            skip=0;early_rows=[];mc_rows=[];variance_rows=[]
            if (old_output/split/'completed.json').exists():
                storage,hashes=link_completed(old_output/split,output/split,rows)
                storages[split]=storage;counts[split]=len(rows);done+=len(rows)
                write_json(run.path/f'recovered_{split}.json',{'questions':len(rows),'sha256':hashes})
                run.write_metrics(completed_splits=counts,storage=storages,completed_questions=done)
                continue
            if split!='test':raise ValueError('Unexpected unfinished split in reboot recovery')
            manifests,early_rows,mc_rows,variance_rows,hashes=recover_test(old_output/split,output/split,rows,old_run,pairs)
            skip=len(early_rows)//7;done+=skip
            writer=ResumedWriter(output/split,manifests,c['fp16_relative_l2_limit']);writer.recovered_rows=len(manifests)
            write_json(run.path/'recovery.json',{'recovered_questions':skip,'source_run':str(old_run),'sha256':hashes})
            for i,row in enumerate(rows):
                if i<skip:continue
                if i%20==0:ensure_space(output)
                prefix=prepare(row,source);gold=numeric_answer(row['answer'])
                early=[early_exit(row,prefix,t) for t in range(7)];model=early[-1]['answer']
                for t,e in enumerate(early):early_rows.append({'problem_id':row['id'],'split':split,'t':t,'answer':e['answer'],'correct':numeric_answer(e['answer'])==gold})
                meta={'problem_id':row['id'],'source':'codi_gpt2','split':split,'question':row['question'],'gold_answer':row['answer'],'gold_cot':row['cot'],
                      'intermediate_values':row['intermediate_values'],'cot_parse_failed':row['cot_parse_failed'],'cohort_id':row['cohort_id'],'left_padding':row['left_padding'],
                      'pad_to':row['pad_to'],'protocol_sha256':row['protocol_sha256'],'model_answer':model,'correct':numeric_answer(model)==gold,'seed':0,
                      'state_definition':'post-forward observation z_t; feedback stored separately','source_distribution':row['source_distribution']}
                layers=list(range(13)) if row['id'] in all_layers else [12]
                states=torch.stack(prefix.layers) if len(layers)>1 else torch.stack([x[0,0].cpu() for x in prefix.states])[:,None,:]
                writer.add(states,{**meta,'kind':'clean','traj_id':-1,'t0':0,'label':numeric_answer(model)==gold,'steps':list(range(7)),'layers':layers})
                if split in ['reader_train','reader_val']:
                    groups={}
                    for k in range(c['trajectory_K']):
                        seed=rng_seed(c['stochastic_seeds'][k%3],row['id'],'traj',k=k);t0=random.Random(seed).randrange(6)
                        groups.setdefault(t0,[]).append((k,seed))
                    for t0,items in groups.items():
                        outputs=rollout(row,prefix,t0,c['sigma'],0,[s for k,s in items])
                        for (k,seed),o in zip(items,outputs):
                            correct=numeric_answer(o['answer'])==gold
                            writer.add(o['states'][:,None,:],{**meta,'kind':'traj','traj_id':k,'t0':t0,'label':correct,'correct':correct,'model_answer':o['answer'],'seed':seed,'steps':list(range(t0,7)),'layers':[12]},o['feedback'])
                else:
                    for t in range(6):
                        for policy,sigma,temp in [('pi',c['sigma'],0),('pi_T',0,1.)]:
                            seeds=[rng_seed(c['stochastic_seeds'][k%3],row['id'],policy,t,k) for k in range(c['mc_K'])]
                            outputs=rollout(row,prefix,t,sigma,temp,seeds)
                            correct=[numeric_answer(o['answer'])==gold for o in outputs]
                            mc_rows.append({'problem_id':row['id'],'split':split,'t':t,'policy':policy,'K':len(seeds),'seeds':seeds,'answers':[o['answer'] for o in outputs],'correct':correct,'value':sum(correct)/len(correct),'invalid_rate':sum(numeric_answer(o['answer']) is None for o in outputs)/len(outputs)})
                            if (row['id'],t) in pairset:
                                # Four independent blocks with distinct seed namespace; no reuse of K16 outcomes.
                                draws=[];answers=[]
                                for block in range(4):
                                    seeds64=[rng_seed(c['stochastic_seeds'][k%3],row['id'],policy+'_variance64',t,block*16+k) for k in range(16)]
                                    o64=rollout(row,prefix,t,sigma,temp,seeds64)
                                    draws.extend(numeric_answer(o['answer'])==gold for o in o64);answers.extend(o['answer'] for o in o64)
                                variance_rows.append({'problem_id':row['id'],'split':split,'t':t,'policy':policy,'K':64,'correct':draws,'answers':answers,'blocks16':[sum(draws[b:b+16])/16 for b in range(0,64,16)],'value':sum(draws)/64})
                done+=1
                if (i+1)%64==0:
                    writer.flush()
                    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>c['max_disk_gib']*2**30:raise RuntimeError('M1 disk cap reached; no silent data reduction')
                if (i+1)%20==0:
                    # Durable behavior evidence even if a later GPU failure interrupts the split.
                    write_json(run.path/f'{split}_early_partial.json',early_rows)
                    write_json(run.path/f'{split}_mc_partial.json',mc_rows)
                    write_json(run.path/f'{split}_variance_partial.json',variance_rows)
                elapsed=time.perf_counter()-start
                run.write_metrics(completed_questions=done,progress={'split':split,'completed':i+1,'total':len(rows)},gpu_hours=elapsed/3600,source_forward_calls=source.forward_calls)
                if elapsed/3600+prior_hours+budget['preflight_gpu_hours']>c['max_gpu_hours']:raise RuntimeError('M1 GPU hour cap reached; stopped without altering remaining settings')
                # User authorized managed reserve release, not low-space termination.
            storages[split]=writer.close();counts[split]=len(rows)
            pq.write_table(pa.Table.from_pylist(early_rows),output/split/'early_exit.parquet')
            if mc_rows:pq.write_table(pa.Table.from_pylist(mc_rows),output/split/'mc.parquet')
            if variance_rows:pq.write_table(pa.Table.from_pylist(variance_rows),output/split/'variance64.parquet')
            write_json(output/split/'completed.json',{'run':str(run.path),'problem_count':len(rows),'storage':storages[split]})
            run.write_metrics(completed_splits=counts,storage=storages)
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>c['max_disk_gib']*2**30:raise RuntimeError('Final data exceed8GiB cap; retained as failed output for review')
        manifest={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
        write_json(run.path/'artifacts_sha256.json',manifest)
        run.write_metrics(status='completed',counts=counts,storage=storages,all_layers_questions=len(all_layers),variance_pairs=len(pairs),output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        run.write_metrics(prior_generation_gpu_hours=prior_hours,prior_source_forward_calls=sum(r['source_forward_calls'] for r in ancestors),total_generation_gpu_hours=prior_hours+(time.perf_counter()-start)/3600)
        print(run.path,flush=True)
    finally:
        elapsed=time.perf_counter()-start;run.write_metrics(wall_seconds=elapsed,gpu_hours=elapsed/3600 if source else 0,source_forward_calls=source.forward_calls if source else 0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

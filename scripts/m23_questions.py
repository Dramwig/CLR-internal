"""Generate immutable QA tables from M1, with no source forward calls."""
import argparse,json,time,random,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.data.questions.core import seed,number,choices,matched_values
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

COLS=['problem_id','kind','traj_id','t0','t','layer','model_answer','correct','gold_answer','intermediate_values','cot_parse_failed']
def main(c):
 with recorded(c) as run:
    if c['mode']=='full':
        if not c.get('smoke_run') or json.loads(Path(c['smoke_run'],'metrics.json').read_text())['status']!='completed':raise ValueError('Successful question smoke required')
    start=time.monotonic();out=Path(c['output'])/c['mode'];out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(c['plan']).read_text());calids=sorted([r['id'] for r in plan['splits']['cal']],key=lambda p:seed(c['seed'],'cal',p))
    frozen={'seed':c['seed'],'cal_A':calids[:250],'cal_B':calids[250:]}
    splitpath=Path(c['cal_split'])
    if splitpath.exists():
        if json.loads(splitpath.read_text())!=frozen:raise ValueError('Frozen calibration split mismatch')
    else:write_json(splitpath,frozen)
    global_donors=list(dict.fromkeys(number(r['answer']) for r in plan['splits']['reader_train'][:1000]))[:64]
    summary={}
    for split,problems in plan['splits'].items():
        if c['mode']=='smoke':problems=sorted(problems,key=lambda r:seed(c['seed'],r['id']))[:max(1,int(len(problems)*c['smoke_fraction']))]
        ids={r['id'] for r in problems}
        frame=pq.read_table(Path(c['root'])/split/'manifest.parquet',columns=COLS,filters=[('layer','=',12)]).to_pylist()
        frame=[r for r in frame if r['problem_id'] in ids]
        groups=defaultdict(list)
        for r in frame:groups[r['problem_id']].append(r)
        early={(r['problem_id'],r['t']):int(r['correct']) for r in pq.read_table(Path(c['root'])/split/'early_exit.parquet').to_pylist() if r['problem_id'] in ids}
        mcpath=Path(c['root'])/split/'mc.parquet'
        mc={(r['problem_id'],r['t']):r for r in pq.read_table(mcpath).to_pylist() if r['problem_id'] in ids and r['policy']=='pi'} if mcpath.exists() else {}
        mcanswers=defaultdict(list)
        for (pid,t),m in mc.items():mcanswers[pid].extend(m['answers'])
        intermediate={p:{number(v) for v in rs[0]['intermediate_values']} for p,rs in groups.items() if not rs[0]['cot_parse_failed'] and rs[0]['intermediate_values']}
        matched,unpaired=matched_values(intermediate,seed(c['seed'],split,'q4'))
        pool=[r['model_answer'] for r in frame if r['kind']=='clean' and r['t']==6]
        writer=None;buf=[];counts=Counter();positive=Counter();positions=Counter();coverage=Counter();labeled=set();n=0
        def emit(pid,ref,q,options,y,soft=None,draws=None,rep=0):
            nonlocal n,buf,writer
            if q=='Q1' and ref['traj_id']==-1:labeled.add((pid,ref['t']))
            key=(q,ref['t'],'clean' if ref['traj_id']==-1 else 'traj');counts[key]+=1;positive[key]+=float(y if soft is None else soft)
            if q=='Q3':positions[y]+=1
            tid=seed(c['seed'],pid,q,ref['traj_id'],ref['t'],rep)%4
            tids=[tid] if split=='reader_train' else [tid,4,5]
            for template in tids:
                row={'problem_id':pid,'split':split,'cal_part':('cal_A' if pid in frozen['cal_A'] else 'cal_B') if split=='cal' else '',
                     'state_ref':ref,'qtype':q,'template_id':template,'is_heldout_template':template>=4,
                     'options':options,'label':int(y),'label_soft':float(y if soft is None else soft),
                     'bernoulli_outcomes':draws or [int(y)],'replicate':rep}
                buf.append(row);n+=1
            if len(buf)>=10000:
                table=pa.Table.from_pylist(buf)
                if writer is None:writer=pq.ParquetWriter(out/f'{split}.parquet',table.schema,compression='zstd')
                writer.write_table(table);buf=[]
        for pid,rs in groups.items():
            own=[r['model_answer'] for r in rs]+[rs[0]['gold_answer']]
            own += mcanswers[pid]
            rng=random.Random(seed(c['seed'],pid));distractors=own+rng.sample(pool,min(64,len(pool)))+global_donors
            for r in rs:
                t=r['t'];traj=r['traj_id'];ref={'kind':r['kind'],'traj_id':traj,'t':t}
                if traj<0:
                    if t==6:emit(pid,ref,'Q1',[],int(r['correct']))
                    elif (pid,t) in mc:
                        m=mc[pid,t];emit(pid,ref,'Q1',[],int(m['correct'][0]),m['value'],list(map(int,m['correct'])))
                    emit(pid,ref,'Q2',[],early[pid,t])
                    if pid in matched:
                        for y,v in zip([1,0],matched[pid]):emit(pid,ref,'Q4',[v],y,rep=y)
                elif t==r['t0']:
                    emit(pid,{'kind':'clean','traj_id':-1,'t':t},'Q1',[],int(r['correct']),rep=traj+1)
                else:emit(pid,ref,'Q1',[],int(r['correct']),rep=traj+1)
                opts,y=choices(r['model_answer'],distractors,random.Random(seed(c['seed'],pid,traj,t,'choices')))
                emit(pid,ref,'Q3',opts,y)
            if c['mode']=='smoke' and time.monotonic()-start>c['smoke_seconds']:raise TimeoutError('M23 question smoke >15min')
        if buf:
            table=pa.Table.from_pylist(buf)
            if writer is None:writer=pq.ParquetWriter(out/f'{split}.parquet',table.schema,compression='zstd')
            writer.write_table(table)
        writer.close()
        summary[split]={'problems':len(ids),'rows':n,'q4_unpaired':unpaired,'q4_parse_failed':sum(r['cot_parse_failed'] for r in problems),
           'by_qtype_step_kind':[{ 'qtype':q,'t':t,'kind':kind,'n':n,'label_mean':positive[q,t,kind]/n} for (q,t,kind),n in sorted(counts.items())],
           'q3_positions':dict(positions),'q1_missing_clean_fraction':{str(t):1-sum((p,t) in labeled for p in ids)/len(ids) for t in range(7)}}
        run.write_metrics(progress=split,counts={k:v['rows'] for k,v in summary.items()})
    write_json(out/'summary.json',summary)
    write_json(out/'provenance.json',{'run':str(run.path),'source':c['root'],'cal_split_sha256':hashlib.sha256(splitpath.read_bytes()).hexdigest(),'templates_sha256':hashlib.sha256(Path(c['templates']).read_bytes()).hexdigest()})
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=0,wall_seconds=time.monotonic()-start,summary=summary)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

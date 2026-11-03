"""Fixed M23 question semantics on M4 unseen data; no source model calls."""
import argparse,json,random,time
from pathlib import Path
from collections import defaultdict,Counter
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.questions.core import seed,number,choices,matched_values

def main(c):
 with recorded(c) as run:
    root=Path(c['output']);source=root/('unseen_smoke' if c['mode']=='smoke' else 'unseen');out=root/('qa_smoke' if c['mode']=='smoke' else 'qa');out.mkdir(exist_ok=False)
    raw=pq.read_table(Path(c['root'])/'reader_train/manifest.parquet',columns=['gold_answer'],filters=[('kind','=','clean'),('layer','=',12),('t','=',6)]).column('gold_answer').to_pylist()
    donors=list(dict.fromkeys(number(x) for x in raw[:1000]))[:64];summary={}
    for split in ['unseen_train','unseen_eval']:
        rows=pq.read_table(source/split/'manifest.parquet').to_pylist();groups=defaultdict(list)
        for r in rows:groups[r['problem_id']].append(r)
        early={(r['problem_id'],r['t']):int(r['correct']) for r in pq.read_table(source/split/'early_exit.parquet').to_pylist()}
        mcpath=source/split/'mc.parquet';mc={(r['problem_id'],r['t']):r for r in pq.read_table(mcpath).to_pylist()} if mcpath.exists() else {}
        pairs={};unpaired=[]
        for pool in sorted({r['pool'] for r in rows}):
            values={p:{number(x) for x in rs[0]['intermediate_values']} for p,rs in groups.items() if rs[0]['pool']==pool and not rs[0]['cot_parse_failed']}
            found,missing=matched_values(values,seed(c['seed'],'M4Q4',split,pool));pairs.update(found);unpaired+=missing
        pool=[r['model_answer'] for r in rows if r['kind']=='clean' and r['t']==6];result=[];counts=Counter()
        def emit(pid,ref,q,opts,y,soft=None,draws=None,rep=0):
            tid=seed(c['seed'],pid,q,ref['traj_id'],ref['t'],rep)%4
            for template in ([tid] if split=='unseen_train' else [tid,4,5]):
                result.append({'problem_id':pid,'split':split,'cal_part':'','state_ref':ref,'qtype':q,'template_id':template,'is_heldout_template':template>=4,'options':opts,'label':int(y),'label_soft':float(y if soft is None else soft),'bernoulli_outcomes':draws or [int(y)],'replicate':rep})
            counts[q]+=1
        for pid,rs in groups.items():
            own=[r['model_answer'] for r in rs]+[rs[0]['gold_answer']]
            for t in range(6):own+=mc.get((pid,t),{}).get('answers',[])
            distractors=own+random.Random(seed(c['seed'],pid)).sample(pool,min(64,len(pool)))+donors
            for r in rs:
                t=r['t'];traj=r['traj_id'];ref={'kind':r['kind'],'traj_id':traj,'t':t}
                if traj<0:
                    if t==6:emit(pid,ref,'Q1',[],int(r['correct']))
                    elif (pid,t) in mc:
                        m=mc[pid,t];emit(pid,ref,'Q1',[],m['correct'][0],m['value'],list(map(int,m['correct'])))
                    emit(pid,ref,'Q2',[],early[pid,t])
                    if pid in pairs:
                        for y,v in zip([1,0],pairs[pid]):emit(pid,ref,'Q4',[v],y,rep=y)
                else:
                    target={'kind':'clean','traj_id':-1,'t':t} if t==r['t0'] else ref
                    emit(pid,target,'Q1',[],int(r['correct']),rep=traj+1)
                opts,y=choices(r['model_answer'],distractors,random.Random(seed(c['seed'],pid,traj,t,'choices')));emit(pid,ref,'Q3',opts,y)
        pq.write_table(pa.Table.from_pylist(result),out/f'{split}.parquet',compression='zstd')
        summary[split]={'problems':len(groups),'rows':len(result),'counts':dict(counts),'q4_unpaired':unpaired,'q4_parse_failed':sum(rs[0]['cot_parse_failed'] for rs in groups.values())}
    write_json(out/'summary.json',summary);run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=0,summary=summary);print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

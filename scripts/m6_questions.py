"""M6 source-specific behavior labels; Q4 values copied from GPT2's frozen pairs."""
import argparse,json,random,time
from pathlib import Path
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
from clr.data.questions.core import seed,number,choices
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.m6_runtime import snapshot
from clr.utils.idea_a import write_json

def main(c):
 with recorded(c) as run:
    snapshot(run,{**c,'section':c.get('section','CD')})
    print(run.path,flush=True)
    started=time.monotonic();out=Path(c['qa_root']);out.mkdir(parents=True,exist_ok=True);cal=json.loads(Path(c['cal_split']).read_text())
    if c['mode']=='smoke' and any(s not in ['val','reader_train'] for s in c['splits']):raise ValueError('QA smoke only train/val proxies')
    for split in c['splits']:
        dest=out/f'{split}.parquet'
        if dest.exists():raise FileExistsError(dest)
        frame=pq.read_table(Path(c['root'])/split/'manifest.parquet').to_pylist();groups=defaultdict(list)
        for r in frame:groups[r['problem_id']].append(r)
        early={(r['problem_id'],r['t']):r['correct'] for r in pq.read_table(Path(c['root'])/split/'early_exit.parquet').to_pylist()}
        path=Path(c['root'])/split/'mc.parquet';mc={(r['problem_id'],r['t']):r for r in pq.read_table(path).to_pylist() if r['policy']=='pi'} if path.exists() else {}
        old=pq.read_table(Path(c['gpt2_qa'])/f'{split}.parquet',columns=['problem_id','qtype','options','label'],filters=[('qtype','=','Q4'),('is_heldout_template','=',False)]).to_pylist()
        paired={}
        for r in old:
            if r['problem_id'] in groups:paired.setdefault(r['problem_id'],{})[r['label']]=r['options'][0]
        pool=[r['model_answer'] for r in frame if r['traj_id']==-1 and r['t']==6];buf=[];writer=None;n=0
        def emit(pid,ref,q,options,y,soft=None,draws=None,rep=0):
            nonlocal n,buf,writer
            tid=seed(c['seed'],pid,q,ref['traj_id'],ref['t'],rep)%4
            for template in ([tid] if split=='reader_train' else [tid,4,5]):
                buf.append({'problem_id':pid,'split':split,'cal_part':('cal_A' if pid in cal['cal_A'] else 'cal_B') if split=='cal' else '',
                    'state_ref':ref,'qtype':q,'template_id':template,'is_heldout_template':template>=4,'options':options,'label':int(y),
                    'label_soft':float(y if soft is None else soft),'bernoulli_outcomes':list(map(int,draws)) if draws else [int(y)],'replicate':rep});n+=1
            if len(buf)>=10000:
                tab=pa.Table.from_pylist(buf)
                if writer is None:writer=pq.ParquetWriter(dest,tab.schema,compression='zstd')
                writer.write_table(tab);buf=[]
        for pid,rs in groups.items():
            distractors=[r['model_answer'] for r in rs]+[rs[0]['gold_answer']]+random.Random(seed(c['seed'],pid)).sample(pool,min(64,len(pool)))
            for (p,t),m in mc.items():
                if p==pid:distractors+=m['answers']
            # Original training donor numeric pool, no test-derived distractors.
            distractors+=c['numeric_donor_pool']
            for r in rs:
                t,traj=r['t'],r['traj_id'];ref={'kind':r['kind'],'traj_id':traj,'t':t}
                if traj<0:
                    if t==6:emit(pid,ref,'Q1',[],r['correct'])
                    elif (pid,t) in mc:
                        m=mc[pid,t];emit(pid,ref,'Q1',[],m['correct'][0],m['value'],m['correct'])
                    emit(pid,ref,'Q2',[],early[pid,t])
                    for y,v in paired.get(pid,{}).items():emit(pid,ref,'Q4',[v],y,rep=y)
                elif t==r['t0']:emit(pid,{'kind':'clean','traj_id':-1,'t':t},'Q1',[],r['correct'],rep=traj+1)
                else:emit(pid,ref,'Q1',[],r['correct'],rep=traj+1)
                opts,y=choices(r['model_answer'],distractors,random.Random(seed(c['seed'],pid,traj,t,'choices')));emit(pid,ref,'Q3',opts,y)
        if buf:
            tab=pa.Table.from_pylist(buf)
            if writer is None:writer=pq.ParquetWriter(dest,tab.schema,compression='zstd')
            writer.write_table(tab)
        writer.close();run.write_metrics(progress=split,questions=len(groups),rows=n,q4_paired=len(paired),source_forward_calls=0,gpu_hours=0)
    run.write_metrics(status='completed',wall_seconds=time.monotonic()-started,test_opened='test' in c['splits'],gpu_hours=0,source_forward_calls=0)
    print(run.path,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

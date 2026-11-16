"""M6 behavior-label/split audit, with test excluded in every smoke."""
import argparse,json,time
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from clr.data.questions.core import number
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.m6_runtime import snapshot
from clr.utils.idea_a import write_json

def main(c):
 with recorded(c) as run:
    snapshot(run,{**c,'section':c.get('section','CD')})
    print(run.path,flush=True)
    start=time.monotonic();seen=set();summary={}
    if c['mode']=='smoke' and set(c['splits'])-{'reader_train','val'}:raise ValueError('Data audit smoke excludes test/cal')
    for split in c['splits']:
        manifest=pq.read_table(Path(c['root'])/split/'manifest.parquet').to_pylist();states={(r['problem_id'],r['traj_id'],r['t']):r for r in manifest}
        ids={r['problem_id'] for r in manifest}
        if seen&ids:raise ValueError('M6 problem ID leakage')
        seen|=ids
        if {r['dim'] for r in manifest}!={2048} or {r['layer'] for r in manifest}!={16}:raise ValueError('Wrong LLaMA state source/layer')
        qa=pq.read_table(Path(c['qa_root'])/f'{split}.parquet').to_pylist()
        early={(r['problem_id'],r['t']):int(r['correct']) for r in pq.read_table(Path(c['root'])/split/'early_exit.parquet').to_pylist()}
        original=pq.read_table(Path(c['gpt2_qa'])/f'{split}.parquet',columns=['problem_id','label','options'],filters=[('qtype','=','Q4')]).to_pylist();q4={(r['problem_id'],r['label']):r['options'] for r in original if r['problem_id'] in ids}
        mcpath=Path(c['root'])/split/'mc.parquet';mc={(r['problem_id'],r['t']):r for r in pq.read_table(mcpath).to_pylist() if r['policy']=='pi'} if mcpath.exists() else {}
        for r in qa:
            pid=r['problem_id'];ref=r['state_ref'];traj,t=ref['traj_id'],ref['t'];state=states[pid,traj,t];q=r['qtype']
            if q=='Q3':
                if len(set(r['options']))!=4 or r['options'][r['label']]!=number(state['model_answer']):raise ValueError('Incorrect Q3 option/label')
            elif q=='Q2':
                if traj!=-1 or r['label']!=early[pid,t]:raise ValueError('Q2 label differs from observed early exit')
            elif q=='Q4':
                if traj!=-1 or r['options']!=q4[pid,r['label']]:raise ValueError('Q4 must reuse frozen GPT2 pairing exactly')
            elif q=='Q1':
                if traj>=0:
                    if t<=state['t0'] or r['label']!=int(state['correct']):raise ValueError('Q1 trajectory supervision must follow t0')
                elif t<6 and r['replicate']>0:
                    tr=r['replicate']-1;origin=states[pid,tr,t]
                    if origin['t0']!=t or r['label']!=int(origin['correct']) or r['label_soft']!=r['label']:raise ValueError('Q1 start is independent Bernoulli, no earlier prefixes/averaging')
                elif t<6:
                    m=mc[pid,t]
                    if m['K']!=16 or len(m['correct'])!=16 or r['label_soft']!=sum(m['correct'])/16:raise ValueError('Bad Q1 MC label')
                elif r['label']!=int(state['correct']):raise ValueError('Q1 endpoint must use own clean/traj correctness')
        marginals={y:Counter(r['options'][0] for r in qa if r['qtype']=='Q4' and r['label']==y and not r['is_heldout_template']) for y in [0,1]}
        totals={y:sum(v.values()) for y,v in marginals.items()}
        if bool(totals[0])!=bool(totals[1]):raise ValueError('Q4 panel missing only one label')
        tv=.5*sum(abs(marginals[1][v]/totals[1]-marginals[0][v]/totals[0]) for v in marginals[0].keys()|marginals[1].keys()) if totals[0] else None
        summary[split]={'problems':len(ids),'states':len(manifest),'qa_rows':len(qa),'q4_reused':True,'q1_protocol':True,
          'q4_numeric_marginals':{'positive_instances':totals[1],'negative_instances':totals[0],'total_variation':tv,'exactly_equal':marginals[0]==marginals[1] if totals[0] else None,'scope':'seen templates; fixed GPT2 pair values retained; ID prefixes need not be pair-closed'}}
    write_json(run.path/'audit.json',summary);run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start,summary=summary,test_opened='test' in c['splits'])
    print(run.path,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

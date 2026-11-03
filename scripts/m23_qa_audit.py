"""Full QA label/referential audit without source calls; no performance tuning."""
import argparse,json
from pathlib import Path
from collections import Counter
import torch
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.data.questions.core import number

def main(c):
 with recorded(c) as run:
    root=Path(c['output'])/c['mode'];allids={};counts={}
    for split in ['reader_train','reader_val','val','cal','test']:
        meta=pq.read_table(Path(c['root'])/split/'manifest.parquet',columns=['problem_id','traj_id','t','t0','layer','correct','model_answer','intermediate_values','cot_parse_failed'],filters=[('layer','=',12)]).to_pylist()
        by={(r['problem_id'],r['traj_id'],r['t']):r for r in meta}
        early={(r['problem_id'],r['t']):r for r in pq.read_table(Path(c['root'])/split/'early_exit.parquet').to_pylist()}
        rows=pq.read_table(root/f'{split}.parquet',filters=[('is_heldout_template','=',False)]).to_pylist()
        ids=set();checked=Counter();margins={0:Counter(),1:Counter()}
        for r in rows:
            pid=r['problem_id'];ids.add(pid);ref=r['state_ref'];traj=ref['traj_id'];t=ref['t'];state=by[pid,traj,t];q=r['qtype'];checked[q]+=1
            if q=='Q1':
                if traj>=0:
                    assert t>state['t0'];assert r['label']==int(state['correct'])
                elif t==6:assert r['label']==int(state['correct'])
                elif split in ['reader_train','reader_val']:
                    donor=by[pid,r['replicate']-1,t];assert donor['t0']==t;assert r['label']==int(donor['correct'])
                else:
                    assert len(r['bernoulli_outcomes'])==16
                    assert r['label_soft']==sum(r['bernoulli_outcomes'])/16
            elif q=='Q2':assert traj==-1 and r['label']==int(early[pid,t]['correct'])
            elif q=='Q3':
                assert len(set(r['options']))==4
                assert r['options'][r['label']]==number(state['model_answer'])
            elif q=='Q4':
                assert not state['cot_parse_failed'] and traj==-1
                v=r['options'][0];assert (v in {number(x) for x in state['intermediate_values']})==bool(r['label'])
                margins[r['label']][t,v]+=1
            else:raise ValueError(q)
        assert margins[0]==margins[1]
        for previous,prev_ids in allids.items():assert not ids&prev_ids,(split,previous)
        allids[split]=ids;counts[split]=dict(checked)
    calibration=json.loads(Path(c['cal_split']).read_text());assert len(calibration['cal_A'])==len(calibration['cal_B'])==250
    assert not set(calibration['cal_A'])&set(calibration['cal_B'])
    assert set(calibration['cal_A'])|set(calibration['cal_B'])==allids['cal']
    run.write_metrics(status='completed',checked_counts=counts,source_forward_calls=0,gpu_hours=0)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

"""Audit generated labels and split/state provenance without reader evaluation."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import torch
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.questions.core import number

def main(c):
 with recorded(c) as run:
    root=Path(c['output']);ids={};report={}
    for split in ['unseen_train','unseen_eval']:
        rows=pq.read_table(root/'unseen'/split/'manifest.parquet').to_pylist();lookup={(r['problem_id'],r['traj_id'],r['t']):r for r in rows};ids[split]={r['problem_id'] for r in rows}
        qa=pq.read_table(root/'qa'/f'{split}.parquet').to_pylist();clean={r['problem_id']:r for r in rows if r['traj_id']==-1 and r['t']==6};positions=Counter();values={0:Counter(),1:Counter()};counts=Counter()
        early={(r['problem_id'],r['t']):bool(r['correct']) for r in pq.read_table(root/'unseen'/split/'early_exit.parquet').to_pylist()}
        for r in qa:
            pid=r['problem_id'];t=r['state_ref']['t'];traj=r['state_ref']['traj_id'];q=r['qtype'];state=lookup[pid,traj,t];counts[q]+=1
            assert state['protocol_sha256']==clean[pid]['protocol_sha256'] and state['pad_to']==clean[pid]['pad_to']
            if q=='Q1' and split=='unseen_train':
                if traj==-1 and t<6:
                    source=lookup[pid,r['replicate']-1,t];assert source['t0']==t and r['replicate']>0
                else:source=state
                assert int(source['correct'])==r['label']
                if traj>=0:assert t>state['t0']
            if q=='Q2':assert traj==-1 and r['label']==early[pid,t]
            if q=='Q3':
                assert len(r['options'])==len(set(r['options']))==4
                assert r['options'][r['label']]==number(state['model_answer']);positions[r['label']]+=1
            if q=='Q4':
                assert not clean[pid]['cot_parse_failed'];v=r['options'][0];gold={number(x) for x in clean[pid]['intermediate_values']}
                assert int(v in gold)==r['label'];values[r['label']][v]+=1
        assert values[0]==values[1], 'Q4 numeric marginal mismatch'
        report[split]={'problems':len(ids[split]),'states':len(rows),'question_rows':len(qa),'qtype_rows':dict(counts),'choice_positions':dict(positions),'q4_numeric_marginals_exact':True,'q1_label_policy_passed':True,'cohort_protocol_consistent':True}
    assert not ids['unseen_train']&ids['unseen_eval']
    frozen=json.loads(Path(c['unseen_split']).read_text())
    write_json(root/'data_audit.json',report);run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,report=report);print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

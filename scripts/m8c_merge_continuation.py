"""CPU derivation merging unchanged saved trials; never impersonates a GPU run."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from clr.audit.m8c_runtime import snapshot,Cost,live
from clr.audit.features import sha256
from clr.utils.records import recorded
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);rows={};parents=[]
        lock=json.loads((root/'final_lock.json').read_text());ids=json.loads((root/'test_ids.json').read_text())['group_ids']
        for parent in c['parent_runs']:
            p=Path(parent);met=json.loads((p/'metrics.json').read_text())
            assert met['status'] in ['completed','failed']
            parents.append({'run':str(p),'results_sha256':sha256(p/'results.jsonl'),'status':met['status']})
            for line in (p/'results.jsonl').read_text().splitlines():
                row=json.loads(line);key=tuple(row[x] for x in ['group_id','kind','epsilon','attack_design','attack_seed','lambda'])
                assert key not in rows;rows[key]=line
        expected={(g,'E3',e,d,c['reader_seed'],1.) for g in ids for e in lock['main']['epsilons'] for d in lock['main']['designs']}
        assert set(rows)==expected and len(ids)==200
        (run.path/'results.jsonl').write_text('\n'.join(rows[k] for k in sorted(rows))+'\n')
        write_json(run.path/'derivation.json',{'kind':'CPU concatenation of exact source lines','parents':parents,'row_count':len(rows),'test_ids_sha256':sha256(root/'test_ids.json')})
        cost.test_opened=True;cost.update(status='completed',derivation_only=True,source_gpu_calls=0,groups_completed=200)
        write_json(root/f'final_test_seed{c["reader_seed"]}_part0.json',{'run':str(run.path),'groups':200,'test_opened':True,'derived_from':parents,'cpu_derivation_only':True})
        live(stage='primary_continuation_merged',test_opened=True,primary_complete=True)
        print(run.path,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

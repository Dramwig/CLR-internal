"""Preserve already audited input-only cohort contexts before final v3 freeze."""
import argparse,json,hashlib
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
    plan=json.loads(Path(c['candidate_path']).read_text())
    evidence=Path('reports/evidence_v2/protocol_all_questions.json')
    # Read padding audit only; no prediction/gold fields inform context selection.
    original={r['problem_id']:{'pad_to':r['padding_audit']['v2_pad_to'],'left_padding':r['padding_audit']['v2_pad_count'],'cohort_id':f'M0_official_test:{r["index"]//128}'} for r in json.loads(evidence.read_text())}
    old_val={r['id']:r['pad_to'] for r in json.loads(Path('runs/20261028T131534016226Z_83bfd1aab8dd_d4e55b/panel.json').read_text())}
    if any(old_val[r['id']]!=r['pad_to'] for r in plan['splits']['val']):raise RuntimeError('Val context differs from accepted M0.1 noise panel')
    changes=[];updates={}
    for split in ['cal','test']:
        for r in plan['splits'][split]:
            if r['id'] in original:
                update=original[r['id']];updates[r['id']]=update
                if r['pad_to']!=update['pad_to']:changes.append({'id':r['id'],'split':split,'candidate_pad_to':r['pad_to'],'production_pad_to':update['pad_to']})
    destination=Path(c['candidate_path']).with_name('contexts.json')
    if destination.exists():raise FileExistsError(destination)
    write_json(destination,{'updates':updates,'source':str(evidence),'source_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),'reason':'Preserve original1319-question audited production cohorts despite removal/repartition; candidate-only contexts were never used on cal/test'})
    write_json(run.path/'candidate_context_corrections.json',changes)
    run.write_metrics(status='completed',val_contexts_verified=300,heldout_contexts_restored=len(updates),candidate_width_changes=len(changes),source_forward_calls=0,gpu_hours=0)
    print(run.path)

"""Optional D source data only; final reader inference remains behind its lock.

Reuses the same M6 source/QA modules and their completed val proxy smokes.
The frozen M4 pool IDs/halves and Q4 pairs are retained in full.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from scripts.m6_pipeline import Controller
from scripts.m6_source import contexts
from scripts.m6_merge_source import merge
from clr.utils.m6_unseen import input_rows
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import snapshot,live


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);start=time.monotonic();snapshot(run,{**c,'section':'CD'})
    output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
    ctl=Controller(c);ctl.state['status']='running';ctl.save()
    try:
        parent=json.loads(Path(c['parent_pipeline']).read_text())
        if Path(c['training_lock']).exists():raise ValueError('Optional D source admission belongs before the all-training lock')
        if not all(f'single_llama_{kind}_{sd}' in parent['models'] for kind in ['lite','b3'] for sd in [11,22,33]):raise ValueError('C1 must finish before optional D')
        source=load_config(c['source_config']);protocol=load_config(source['protocol'])
        smoke=parent['steps']['source_selected_sigma_smoke']['run']
        if not json.loads(Path(smoke,'metrics.json').read_text()).get('gate_passed'):raise ValueError('Same-policy val source smoke required')
        qa_smoke=Controller.find_run(c['qa_smoke_log'])
        if not qa_smoke or json.loads(Path(qa_smoke,'metrics.json').read_text()).get('status')!='completed':raise ValueError('M6 val QA smoke required')
        frozen=json.loads(Path(c['input_plan']).read_text());origin_members=defaultdict(set)
        origins={(r['origin_root'],r['origin_split']) for rows in frozen['splits'].values() for r in rows}
        for root,split in sorted(origins):
            for r in pq.read_table(Path(root)/split/'manifest.parquet',columns=['problem_id','cohort_id']).to_pylist():origin_members[r['cohort_id']].add(r['problem_id'])
        splits=input_rows(frozen,dict(origin_members))
        tokenizer=AutoTokenizer.from_pretrained(protocol['source']['base_path'],local_files_only=True)
        ph=hashlib.sha256(Path(source['protocol']).read_bytes()).hexdigest()
        contexts([r for rows in splits.values() for r in rows],tokenizer,ph)
        original=json.loads(Path(c['source_plan']).read_text())
        plan={'splits':splits,'sigma':original['sigma'],'noise_selection':original['noise_selection'],
            'input_plan_sha256':hashlib.sha256(Path(c['input_plan']).read_bytes()).hexdigest(),'protocol_sha256':ph}
        path=output/'plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Unseen production plan changed on resume')
        write_json(path,plan)
        write_json(run.path/'input_context_audit.json',{'problems':sum(map(len,splits.values())),
            'cohorts':len(origin_members),'complete_original_cohorts':True,'split_ids_reused':True,
            'padding':'LLaMA tokenizer over all original cohort members, across both fixed halves'})
        base={**source,'mode':'full','stage':'generate','section':'CD','output':str(output),
            'dump_root':str(output/'dumps'),'smoke_run':smoke,'sigma':plan['sigma'],'sigmas':[plan['sigma']],
            'splits':c['splits']}
        devices=c['source_devices']
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures=[pool.submit(ctl.execute,f'unseen_source_{i}','scripts/m6_source.py',
                {**base,'device':d,'generation_shards':len(devices),'generation_shard':i}) for i,d in enumerate(devices)]
            for future in futures:future.result()
        for split in c['splits']:
            if not (output/'dumps'/split/'completed.json').exists():merge({**base,'splits':[split]})
        qa=load_config(c['questions_config']);qa.update(mode='full',root=str(output/'dumps'),
            qa_root=str(output/'qa'),gpt2_qa=c['reference_qa'],splits=c['splits'])
        qr=ctl.execute('unseen_questions','scripts/m6_questions.py',qa)
        ar=ctl.execute('unseen_audit','scripts/m6_audit.py',{**qa,'stage':'audit'})
        completion={'status':'completed','source_data_only':True,'reader_test_opened':False,
            'root':str(output/'dumps'),'qa':str(output/'qa'),'questions_run':qr,'audit_run':ar,
            'input_context_audit':str(run.path/'input_context_audit.json'),'splits':{s:len(v) for s,v in splits.items()}}
        write_json(output/'completion.json',completion);ctl.state['status']='completed';ctl.save()
        live(stage='D_optional_llama_source_completed',optional_llama_unseen=completion)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start)
    except Exception as exc:
        ctl.state.update(status='failed',reason=str(exc));ctl.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

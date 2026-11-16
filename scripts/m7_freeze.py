"""Read-only upstream verification and immutable M7 artifact manifest."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
import torch
from clr.data.m7 import commitment_route
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import snapshot,live


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);report=json.loads((root/'report.json').read_text())
        review=Path(c['review'])
        if review.stat().st_size>8000:raise ValueError('Review size limit exceeded')
        # Verify true split groups only now that final test is materialized.
        groups={};overlap=[];source_checks={}
        for split in ['reader_train','reader_val','cal_A','cal_B','test']:
            rows=[json.loads(l) for l in (root/'source'/split/'manifest.jsonl').open()]
            gg={e['group_id'] for e in rows}
            if len(gg)!=len(rows):raise ValueError('Repeated base group in '+split)
            for old,prev in groups.items():
                if gg&prev:overlap.append([old,split,len(gg&prev)])
            groups[split]=gg
            for e in rows:
                if digest(e['file'])!=e['sha256']:raise ValueError('Source bundle hash mismatch: '+e['file'])
                if e['commitment_source']!='programmatic' or commitment_route(e['commitments'][0])!=e['route'][0]:
                    raise ValueError('Main commitment protocol mismatch: '+e['id'])
                bundle=torch.load(e['file'],map_location='cpu',weights_only=True)
                for msg,meta in zip(bundle['messages'],e['audit_meta']):
                    z=msg['z']
                    if msg['meta']!=meta or z.dtype!=torch.float16 or tuple(z.shape)!=(len(meta['layers'])*meta['tokens_per_layer'],2048):
                        raise ValueError('Saved KV contract mismatch: '+e['id'])
                    if sum(meta['mask'])!=meta['tokens_per_layer'] or meta['K']!='post-RoPE':
                        raise ValueError('Saved KV mask/position mismatch: '+e['id'])
            source_checks[split]={'groups':len(rows),'bundle_hashes_verified':len(rows),'contract_mismatches':[]}
            mismatch=[json.loads(l) for l in (root/'source'/split/'mismatch_v2.jsonl').open()]
            lookup={e['id']:e for e in rows}
            if len(mismatch)!=len(rows) or {e['id'] for e in mismatch}!=set(lookup):
                raise ValueError('Incomplete mismatch v2 output: '+split)
            for e in mismatch:
                if e['donor_id'] not in lookup or lookup[e['donor_id']]['group_id']==lookup[e['id']]['group_id']:
                    raise ValueError('Invalid mismatch donor group')
                if e['message_geometry']!='saved_original_CF_pair_v2' or e['audit_tensor_max_abs']!=0:
                    raise ValueError('Mismatch tensor-equivalence evidence missing')
            source_checks[split]['mismatch_v2_verified']=len(mismatch)
            if split in ['reader_val','test']:
                diagnostics=[json.loads(l) for l in (root/'source'/split/'natural_commitment.jsonl').open()]
                lookup={e['id']:e for e in rows}
                if len(diagnostics)!=len(rows) or {e['id'] for e in diagnostics}!=set(lookup):
                    raise ValueError('Incomplete sender diagnostics: '+split)
                for e in diagnostics:
                    for mode in ['direct','cot512']:
                        d=e[mode]
                        if commitment_route(d['commitment'])!=d['pred'] or d['correct']!=(d['pred']==lookup[e['id']]['route'][0]):
                            raise ValueError('Sender diagnostic prediction/label mismatch: '+e['id'])
                        if mode=='cot512' and len(d['generated_ids'])>512:raise ValueError('Sender diagnostic budget exceeded')
                source_checks[split]['sender_direct_and_cot512_complete']=len(diagnostics)
        if overlap:raise ValueError('Cross-split group leakage: '+str(overlap))
        write_json(root/'source_integrity.json',source_checks)
        # Do not hide pre-existing or concurrent upstream changes.
        upstream={};cache={}
        for path in ['data/M23_v1a/freeze.json','data/M4_v0/freeze.json','data/M41_v1/freeze.json','data/M5a_v13/freeze.json','data/M5b_v1/freeze.json']:
            doc=json.loads(Path(path).read_text());bad=[]
            for name,expected in doc['sha256'].items():
                p=Path(name)
                if name not in cache:cache[name]=digest(p) if p.exists() else None
                if cache[name]!=expected:bad.append(name)
            upstream[path]={'files':len(doc['sha256']),'mismatches':bad,'freeze_sha256':digest(path)}
        write_json(root/'upstream_verification.json',upstream)
        if any(v['mismatches'] for v in upstream.values()):raise ValueError('Upstream frozen artifacts changed; inspect upstream_verification.json')
        paths=[*Path('clr').rglob('*m7*.py'),Path('clr/sources/kv_relay.py'),Path('tests/test_m7.py'),
               *Path('scripts').glob('m7*'),*Path('configs/M7').glob('*'),Path('docs/M7_TASK.md'),
               Path('reports/M7_REVIEW.md'),Path('reports/M7_details.md'),*Path('reports/M7_evidence').glob('*'),
               *root.rglob('*'),*Path('data/splits').glob('M7_v1_*.json')]
        for pointer in (root/'trained').glob('*.json'):
            trained=json.loads(pointer.read_text());paths.extend(Path(trained['run']).glob('*'))
        for pointer in (root/'evaluated').glob('*.json'):
            ev=json.loads(pointer.read_text());paths.extend(Path(ev['run']).glob('*'))
        # Snapshot every closed M7 run; exclude the active freeze/controller logs.
        for path in Path('runs').glob('*/config.json'):
            cfg=json.loads(path.read_text())
            if cfg.get('milestone')!='M7':continue
            metrics=json.loads((path.parent/'metrics.json').read_text())
            if metrics['status'] in ['completed','failed','interrupted']:paths.extend(path.parent.rglob('*'))
        mutable={'freeze.json','verification.json','finish_jobs.json','freeze_final.log','finish_controller.log','finish_controller_v2.log'}
        hashes={str(p):digest(p) for p in sorted(set(paths)) if p.is_file() and p.name not in mutable}
        frozen={'status':'M7_completed_pending_review_with_documented_limits','created_utc':datetime.now(timezone.utc).isoformat(),
                'review':str(review),'report':report,'upstream_verification':str(root/'upstream_verification.json'),
                'sha256':hashes,'test_scope':'final evaluation only after data/M7_v1/final_lock.json',
                'gates':{'historical_raw_KV_artifact_comparison':False,'scale_targets_met':False,'M8_started':False}}
        if (root/'freeze.json').exists():raise FileExistsError('M7 freeze already exists')
        write_json(root/'freeze.json',frozen)
        # Hash write verification; content equality was already checked while hashing.
        write_json(root/'verification.json',{'files':len(hashes),'mismatches':[],'upstream':upstream,'split_overlap':[]})
        text='M7非自适应实验完成待审查；唯一审查reports/M7_REVIEW.md。冻结data/M7_v1/freeze.json；缩量/历史raw缺证据/1%无穷门限等限制见审查文件；未启动M8。'
        for name in ['AGENTS.md','docs/STATUS.md']:
            p=Path(name);old=p.read_text();block='<!-- M7_LIVE -->\n'+text+'\n<!-- /M7_LIVE -->'
            p.write_text(re.sub(r'<!-- M7_LIVE -->.*?<!-- /M7_LIVE -->',block,old,flags=re.S))
        date=datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        def cell(row):return f"{100*row['mean']:.3g}% [{100*row['lo']:.3g},{100*row['hi']:.3g}]"
        primary=report['primary_counterfactual_k16_alpha005']['test'];qa=report['qa_CLR_minus_B3_seen']
        with Path('docs/EXPERIMENTS.md').open('a') as f:
            f.write('\n## '+date+' — M7完成待审查\n'+text+
                    '\nCF/test/k16/名义α=.05：CLR '+cell(primary['CLR'])+'；B3 '+cell(primary['B3'])+'；实测FPR与其他基线见唯一审查文件。'+
                    '\nA1/A4 test CLR−B3：'+cell(qa['A1']).replace('%','pp')+' / '+cell(qa['A4']).replace('%','pp')+
                    f"；累计{report['total_gpu_hours']:.3g}GPUh。"+
                    '\n三seed主比较、按group bootstrap10000次95%CI、natural_commitment附加分析齐全；详见reports/M7_REVIEW.md、analysis/CSV。'+
                    '\n命令/源码快照见configs/M7/、runs/；上游冻结哈希全部一致，test仅最终锁后评估。\n')
        live(stage='completed_pending_review',milestone_complete=True,freeze=str(root/'freeze.json'),review=str(review),
             upstream_verification=str(root/'upstream_verification.json'),
             historical_raw_KV_artifact_comparison_passed=False,scale_targets_met=False,
             alpha001_threshold_is_infinite=True,primary_results=str(root/'report.json'),M8_started=False)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,
                          frozen_files=len(hashes),upstream_files=sum(v['files'] for v in upstream.values()),split_overlap=[]) 
        print(len(hashes),'frozen',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

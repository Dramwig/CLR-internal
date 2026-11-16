"""Descriptive CPU exports for paper appendices, from frozen artifacts only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


def scalar_rows(obj, prefix=''):
    for key, value in obj.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            yield from scalar_rows(value, name + '.')
        elif isinstance(value, (str, int, float, bool)) or value is None:
            yield {'field': name, 'value': value}


def export_appendices(e):
    noise = []
    path = 'runs/20261028T131534016226Z_83bfd1aab8dd_d4e55b/seed_summary.json'
    for record in e.js(path):
        for metric, values in record['metrics'].items():
            ci = values.get('ci95', [values['mean'] - values.get('ci95_halfwidth', 0),
                                     values['mean'] + values.get('ci95_halfwidth', 0)])
            noise.append(dict(source='gpt2', **record['setting'], metric=metric,
                              mean=values['mean'], lo=ci[0], hi=ci[1],
                              interval=record['statistics']))
    e.save('noise_gpt2', noise)
    # Preserve exact early dedup counts, including historical test exposure.
    dedup_path = 'runs/20261028T113854046823Z_6b6596ff9777_f8f563/metrics.json'
    d = e.js(dedup_path)
    dedup = [dict(stage='M0.1 overlap', split=s, **v, evidence=dedup_path)
             for s, v in d['overlap'].items()]
    e.save('deduplication', dedup)

    labels, scales = [], []
    summary = e.js('data/M23_v1a/full/summary.json')
    for split, data in summary.items():
        scales.append(dict(source='gpt2', split=split, problems=data['problems'],
                           qa_rows=data['rows'], unit='problem'))
        labels.extend(dict(source='gpt2', split=split, **r) for r in data['by_qtype_step_kind'])
    plan = e.js('data/M6_v1/plan.json')
    for split, problems in plan['splits'].items():
        p = Path('data/M6_v1/qa') / f'{split}.parquet'
        f = e.pq(p, columns=['qtype', 'state_ref', 'label_soft', 'is_heldout_template']).to_pandas()
        f = f[~f.is_heldout_template].copy()
        f['t'] = f.state_ref.map(lambda r: r['t'])
        f['kind'] = f.state_ref.map(lambda r: r['kind'])
        for (q, t, kind), part in f.groupby(['qtype', 't', 'kind']):
            labels.append(dict(source='llama', split=split, qtype=q, t=t, kind=kind,
                               n=len(part), label_mean=part.label_soft.mean()))
        scales.append(dict(source='llama', split=split, problems=len(problems),
                           qa_rows=len(f), unit='problem; seen templates only'))
    for split in ('reader_train', 'reader_val', 'cal_A', 'cal_B', 'test'):
        p = f'data/splits/M71_v1_{split}.json'
        ids = e.js(p)
        if isinstance(ids, dict):
            ids = ids.get('groups', ids.get('group_ids', ids.get('ids', ids.get('episode_ids', ids))))
        scales.append(dict(source='synthetic_v2_M7.1', split=split, problems=len(ids),
                           qa_rows=np.nan, unit='counterfactual group'))
    e.save('label_statistics', labels)
    e.save('data_scales', scales)

    protocol = []
    for name, path in [('GPT2 production', 'configs/protocol_codi_v2.yaml'),
                       ('LLaMA production', 'configs/protocol_codi_llama_modelscope_v2_gpu4.yaml')]:
        c = yaml.safe_load(e.remember(path).read_text())
        # Only protocol fields: no task examples or problem text.
        spec = c.get('source', c)
        if isinstance(spec, dict):
            for row in scalar_rows(spec):
                if row['field'] in ['dtype','family','layer','num_latent','position_ids','prefill_attention_mask','carry_attention_mask','prj_dim']:
                    protocol.append(dict(section=name, **row))
    selected = e.js('data/M6_v1/noise_selection_v2.json')['selected']
    for key in ['sigma','degraded','status']:
        protocol.append(dict(section='LLaMA actual noise selection', field=key, value=selected[key]))
    selected = e.js('runs/20261028T131534016226Z_83bfd1aab8dd_d4e55b/recommendation.json')['selected']
    for key in ['sigma','from_step']:
        protocol.append(dict(section='GPT2 actual noise selection', field=key, value=selected[key]))
    p = e.js('data/M6_v1/verification.json')
    protocol.append(dict(section='LLaMA context metadata', field='protocol_errors', value=len(p['protocol_errors'])))
    p = e.js('data/M71_v1/storage_comparison.json')
    protocol.extend(dict(section='KV storage verification', **r) for r in scalar_rows(p)
                    if any(k in r['field'].lower() for k in ('error', 'exact', 'elements', 'fraction', 'passed')))
    for section, path in [('M23 cal split', 'data/splits/gsm8k_M23_cal_v1.json'),
                          ('M7.1 projection', 'data/M71_v1/projection.json')]:
        p = e.js(path)
        if section.endswith('split'):
            for key, values in p.items():
                if isinstance(values, list):protocol.append(dict(section=section, field=key, value=len(values)))
        else:
            protocol.extend(dict(section=section, **r) for r in scalar_rows(p)
                            if r['field'] in ['seed','orthogonality_max_abs'])
    e.save('protocol_alignment_v2', protocol)

    # CPU summaries already frozen upstream; do not run models to fill holes.
    for name, path in [('single_source_paired', 'data/M6_v1/analysis/single_source_paired.csv'),
                       ('communication_paired', 'data/M71_v1/analysis_v3/counterfactual_CLR_paired_differences.csv'),
                       ('communication_qa_paired', 'data/M71_v1/analysis_v3/qa_CLR_minus_B3.csv'),
                       ('coverage_recovery', 'data/M6_v1/analysis/coverage_recovery.csv')]:
        e.save(name, e.csv(path))
    e.save('type_transfer_paired', e.csv('reports/M41_paired_bootstrap.csv'))
    for name,path in [('robustness_qa_weak','data/M8b_v1/analysis_v1/QA_main_summary.csv'),
                      ('robustness_qa_dependent','data/M8c_v1/analysis_final21/qa_summary.csv'),
                      ('minimum_cost','data/M8c_v1/analysis_final21/minimum_summary.csv')]:
        e.save(name,e.csv(path))


def export_legacy_costs(e, costs):
    """Use only per-run fields in frozen manifests, retaining incomplete fields.

    Legacy costs have no single closed cumulative ledger. They are documented
    lower bounds, not made comparable by silently counting missing runs as zero.
    """
    manifests = ['data/M4_v0/freeze.json', 'data/M23_v1a/freeze.json',
                 'reports/M1_completed_v3_manifest.json', 'reports/evidence_v2/frozen_data_manifest.json']
    allowed = set()
    for manifest in manifests:
        d = e.js(manifest)
        allowed.update(d.get('sha256', {}))
    # The M1 final artifact map includes its predecessor generation ledgers.
    manifest = e.js('reports/M1_completed_v3_manifest.json')['data_manifest']
    d = e.js(manifest)
    if isinstance(d, dict):allowed.update(d.get('sha256', d))
    runs = sorted({str(Path(p).parent) for p in allowed
                   if len(Path(p).parts) == 3 and Path(p).parts[0] == 'runs'
                   and Path(p).name == 'metrics.json'})
    ledger = []
    for root in runs:
        cfg = Path(root) / 'config.json'
        if str(cfg) not in allowed or not cfg.exists():continue
        c = e.js(cfg);m = e.js(Path(root) / 'metrics.json')
        name = c.get('milestone', '')
        if name not in ['M0', 'M0.1', 'M0b', 'M1', 'M2-M3', 'M2–M3', 'M23', 'M2', 'M3', 'M4']:continue
        group = 'M2–M3' if name in ['M2-M3','M2–M3','M23','M2','M3'] else ('M0/M0.1/M0b' if name in ['M0','M0.1','M0b'] else name)
        ledger.append(dict(milestone=group, run=root, gpu_hours=m.get('gpu_hours'),
                           source_forward_calls=m.get('source_forward_calls'), status=m.get('status')))
    e.save('legacy_cost_runs', ledger)
    if ledger:
        for name, f in pd.DataFrame(ledger).groupby('milestone'):
            costs.insert(0, dict(milestone=name, gpu_hours=f.gpu_hours.sum(min_count=1),
                                 source_forward_calls=f.source_forward_calls.sum(min_count=1),
                                 missing_source_counts=int(f.source_forward_calls.isna().sum()),
                                 cost_scope='lower bound: frozen legacy per-run fields; uninstrumented/pre-freeze runs may be absent',
                                 evidence='data/M9_v1/paper_inputs/legacy_cost_runs.csv'))
    # The old milestones have explicit cumulative fields outside run manifests.
    # Preserve their stated scope, instead of silently omitting those milestones.
    noise_path = 'runs/20261028T131534016226Z_83bfd1aab8dd_d4e55b/metrics.json'
    noise = e.js(noise_path);pre = e.js('reports/evidence_v2/costs.json')
    costs.insert(0,dict(milestone='M0.1',gpu_hours=noise['gpu_hours']+sum(x.get('gpu_hours') or 0 for x in pre),
                        source_forward_calls=noise['source_forward_calls']+sum(x.get('source_forward_calls') or 0 for x in pre),
                        missing_source_counts=sum(x.get('source_forward_calls') is None for x in pre),
                        cost_scope='lower bound: completed v2 noise plus its saved preparatory ledger; earlier M0/M0b not fully instrumented',
                        evidence=noise_path+'; reports/evidence_v2/costs.json'))
    costs.insert(0,dict(milestone='M0/M0b',gpu_hours=np.nan,source_forward_calls=np.nan,
                        missing_source_counts=1,
                        cost_scope='unavailable: no complete frozen cumulative ledger; missing costs are not zero',
                        evidence='reports/M0_env.md; reports/M0_repro.md; reports/evidence/M0b_coconut/manifest.json'))
    first=e.js('reports/M1_completed_v3_manifest.json')['runs']
    generation=e.js(Path(first['generation_run'])/'metrics.json');budget=e.js(Path(first['budget_run'])/'metrics.json')
    calls=generation['prior_source_forward_calls']+generation['source_forward_calls']
    for run in budget['preflight_runs']:
        calls+=e.js(Path(run)/'metrics.json').get('source_forward_calls',0)
    costs.insert(1,dict(milestone='M1',gpu_hours=generation['total_generation_gpu_hours']+budget['preflight_gpu_hours'],
                        source_forward_calls=calls,missing_source_counts=0,
                        cost_scope='lower bound: cumulative v3 generation plus saved smoke/gate preflight; legacy preparations may be absent',
                        evidence='reports/M1_completed_v3_manifest.json; generation/budget run metrics'))
    return costs

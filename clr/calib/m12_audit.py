"""Requirement checks for M12 delivered summaries, without recomputing science."""
import numpy as np

REUSED={'CLR-lite':'M4.1','CLR-v1':'M4.1','CLR-lite-single':'M5b','CLR-per-type':'M10','CLR-lite-multi':'M9'}
METRICS={'Q1':'nll','Q2':'auroc','Q3':'accuracy','Q4':'accuracy'}

def audit_summary(frame,retained,dropped):
    variants=set(REUSED)|set(retained)|set(dropped)
    if set(frame.variant)!=variants:raise ValueError('Variant coverage differs from retained/dropped registry')
    if frame.duplicated(['variant','split','qtype']).any():raise ValueError('Duplicate summary cells')
    keys={(v,s,q) for v in variants for s in ['reader_val','test'] for q in METRICS}
    if set(map(tuple,frame[['variant','split','qtype']].to_numpy()))!=keys:raise ValueError('Summary split/type coverage incomplete')
    for r in frame.to_dict('records'):
        if r['metric']!=METRICS[r['qtype']]:raise ValueError('Wrong metric definition')
        if r['variant'] in dropped:
            if r['status']!='not_completed_budget_or_deadline' or np.isfinite(r['mean']):raise ValueError('Dropped item represented as complete')
            continue
        if r['status']!='completed':raise ValueError('Delivered row incomplete')
        if r['source_milestone']!=REUSED.get(r['variant'],'M12'):raise ValueError('Wrong result provenance')
        if r['bootstrap_draws']!=10000 or r['bootstrap_unit']!='problem':raise ValueError('Wrong uncertainty method')
        seeds=np.array([r[f'seed_{s}'] for s in [11,22,33]],float)
        values=[r[k] for k in ['mean','ci_low','ci_high','paired_delta','paired_ci_low','paired_ci_high']]
        if not np.isfinite(seeds).all() or not np.isfinite(values).all():raise ValueError('Missing metric/CI')
        if not np.isclose(seeds.mean(),r['mean'],rtol=0,atol=1e-12):raise ValueError('Mean is not the three-seed mean')
        if r['ci_low']>r['ci_high'] or r['paired_ci_low']>r['paired_ci_high']:raise ValueError('Reversed interval')
        base=frame[(frame.variant=='CLR-lite')&(frame.split==r['split'])&(frame.qtype==r['qtype'])].iloc[0]
        if not np.isclose(r['paired_delta'],r['mean']-base['mean'],rtol=0,atol=1e-12):raise ValueError('Paired difference has wrong direction/value')
        if r['qtype'] in ['Q3','Q4']:
            if r['conformal_alpha']!=.1:raise ValueError('Wrong conformal alpha')
            fields=['conformal_size','conformal_size_ci_low','conformal_size_ci_high','conformal_size_paired_delta','conformal_size_paired_ci_low','conformal_size_paired_ci_high']
            if not np.isfinite([r[k] for k in fields]).all():raise ValueError('Missing conformal result')
            if not 0<=r['conformal_size']<=(4 if r['qtype']=='Q3' else 2):raise ValueError('Invalid conformal size')
            if not np.isclose(r['conformal_size_paired_delta'],r['conformal_size']-base['conformal_size'],rtol=0,atol=1e-12):raise ValueError('Conformal paired difference mismatch')
    return {'summary_cells':len(frame),'completed_variants':sorted(set(REUSED)|set(retained)),'dropped_variants':list(dropped),'required_metrics':METRICS,'seeds':[11,22,33],'passed':True}

def audit_permutation(summary,observations,retained):
    variants={'main'}|({'A1'} if 'A1' in retained else set())
    keys={(v,s,c) for v in variants for s in ['reader_val','test'] for c in [False,True]}
    if summary.duplicated(['variant','split','canonical']).any() or set(map(tuple,summary[['variant','split','canonical']].to_numpy()))!=keys:
        raise ValueError('Permutation summary coverage incomplete')
    index=['seed','problem_id','t','template_id','replicate']
    if observations.duplicated(['variant','split','canonical',*index]).any():raise ValueError('Duplicate permutation observations')
    actual_keys=set(observations.groupby(['variant','split','canonical']).groups)
    if actual_keys!=keys:raise ValueError('Permutation observation coverage incomplete')
    for split in ['reader_val','test']:
        reference=None
        for variant in variants:
            for canonical in [False,True]:
                g=observations[(observations.variant==variant)&(observations.split==split)&(observations.canonical==canonical)]
                if set(g.seed)!={11,22,33}:raise ValueError('Permutation seeds incomplete')
                support=set(map(tuple,g[index].to_numpy()))
                if reference is None:reference=support
                if support!=reference:raise ValueError('Permutation panels have different support')
                if not np.array_equal(g.changed,(g.original_identity!=g.permuted_identity).astype(int)):raise ValueError('Permutation identity comparison incorrect')
                if not g.original_identity.between(0,3).all() or not g.permuted_identity.between(0,3).all():raise ValueError('Invalid option identity')
                r=summary[(summary.variant==variant)&(summary.split==split)&(summary.canonical==canonical)].iloc[0]
                mean=g.groupby(['seed','t']).changed.mean().mean()
                if r['n']!=len(g) or r['changes']!=g.changed.sum() or not np.isclose(r['mean'],mean,rtol=0,atol=1e-12):raise ValueError('Permutation summary differs from observations')
                if not np.isfinite([r['mean'],r['lo'],r['hi']]).all() or not 0<=r['lo']<=r['hi']<=1:raise ValueError('Invalid permutation interval')
    return {'summary_cells':len(summary),'observations':len(observations),'identity_aligned':True,'seeds':[11,22,33],'passed':True}

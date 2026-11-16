"""Cost-only admission for the optional LLaMA dataset shift before C2/C4."""
from pathlib import Path
import json
import pyarrow.parquet as pq


def estimate(policy, source_plan, inference_rates):
    input_plan=json.loads(Path(policy['input_plan']).read_text())
    counts={s:len(input_plan['splits'][s]) for s in policy['splits']}
    rows={s:pq.read_table(Path(policy['reference_qa'])/f'{s}.parquet',
        columns=['is_heldout_template'],filters=[('is_heldout_template','=',False)]).num_rows for s in counts}
    generation=sum(counts.values())*source_plan['cost_estimate']['held_seconds_per_problem']*policy['generation_safety_factor']/3600
    inference=sum((inference_rates[kind]['seconds_per_example']*sum(rows.values())
        +policy['inference_load_seconds_per_model'])*policy['inference_safety_factor']/3600
        for kind in policy['reader_kinds'] for _ in policy['reader_seeds'])
    return {'problem_counts':counts,'problem_count':sum(counts.values()),'qa_rows':rows,
        'source_generation_estimate_gpu_hours':generation,'six_reader_inference_estimate_gpu_hours':inference,
        'estimated_additional_gpu_hours':generation+inference}


def input_rows(input_plan, origin_members):
    """Require complete original cohorts, then retain frozen ID halves."""
    from collections import defaultdict
    members=defaultdict(set)
    for rows in input_plan['splits'].values():
        for r in rows:members[r['cohort_id']].add(r['id'])
    if dict(members)!=origin_members:raise ValueError('Unseen input plan omits original cohort members')
    result={}
    for split,rows in input_plan['splits'].items():
        result[split]=[{**r,'cot':r.get('gold_cot') or r.get('equation') or '',
            'source_distribution':'unseen_pools'} for r in rows]
    ids=[r['id'] for rows in result.values() for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Unseen input split leakage')
    return result

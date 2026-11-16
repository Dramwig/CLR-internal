"""Fresh, expanded group splits using the frozen synthetic-v2 implementation."""
from dataclasses import replace
import hashlib
from pathlib import Path

from clr.sources.kv_relay import verify_idea_a


def examples(config, split, limit=None):
    verify_idea_a(config['idea_a'])
    from latdef.data.synthetic_v2 import load_v2_config, generate_v2_split, solve_v2_text
    counts=config['counts']
    offsets={'reader_train':0,'reader_val':0,'cal_A':counts['reader_val'],
             'cal_B':counts['reader_val']+counts['cal_A'],'test':0}
    source='train' if split=='reader_train' else 'test' if split=='test' else 'val'
    cfg=replace(load_v2_config(Path(config['idea_a'])/'configs/synthetic/v2.yaml'),
                seed=config['data_seed'], train_groups=counts['reader_train'],
                val_groups=counts['reader_val']+counts['cal_A']+counts['cal_B'],
                test_groups=counts['test'])
    count=counts[split] if limit is None else min(counts[split],limit)
    offset=offsets[split]
    for e in generate_v2_split(cfg,source,group_limit=offset+count):
        index=e.meta['group_index']
        if index<offset or e.meta['alpha']!=1 or e.meta['d_pub']!=(index//12)%3:
            continue
        if solve_v2_text(e)!=e.label:
            raise ValueError('Frozen generator produced a label/text discrepancy')
        yield e


def scan_partition(group_ids,fit_count):
    order=sorted(range(len(group_ids)),key=lambda i:hashlib.sha256(group_ids[i].encode()).hexdigest())
    if len(set(group_ids))!=len(group_ids) or not 0<fit_count<len(group_ids):
        raise ValueError('Invalid group-level layer scan partition')
    return set(order[:fit_count]),set(order[fit_count:])


def selected_layers(anchor,total=36,count=4):
    if total%count:
        raise ValueError('Layer count must divide backbone depth for equal spacing')
    return sorted((anchor+j*(total//count))%total for j in range(count))


def fixed_hash_subset(group_ids,count):
    """Protocol-fixed receiver/natural subset; no outcome-dependent selection."""
    if count<0 or count>len(group_ids):raise ValueError('subset larger than split')
    order=sorted(group_ids,key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    return order[:count]

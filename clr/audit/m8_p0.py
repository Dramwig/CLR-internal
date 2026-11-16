"""Public-only expectations and explicit, diagnostic feature interventions."""
import hashlib
import re
import numpy as np
import torch

from clr.data.m7 import TEAMS, commitment_route


def public_constraints(x_pub):
    """Parse/solve only literal receiver-visible public text; no Example/meta."""
    explicit = re.search(r'verified public result is z = ([01])', x_pub)
    if explicit:
        z = int(explicit[1])
    elif 'Public routing puzzle' in x_pub:
        start, steps = re.search(r'Start at (node_\d+) and follow exactly (\d+) links', x_pub).groups()
        edges = dict(re.findall(r'(node_\d+) leads to (node_\d+)', x_pub))
        flags = dict(re.findall(r'terminal node (node_\d+) has flag ([01])', x_pub))
        for _ in range(int(steps)): start = edges[start]
        z = int(flags[start])
    else:
        value = int(re.search(r'(?:Start with|Initially the warehouse has) (\d+)', x_pub)[1])
        for action, n in re.findall(r'Step \d+: (add|multiply by|subtract|divide by) (\d+)', x_pub):
            n = int(n)
            if action == 'add': value += n
            elif action == 'multiply by': value *= n
            elif action == 'subtract': value -= n
            elif action == 'divide by':
                # The literal public instruction specifies rounding down.
                value //= n
        bound = int(re.search(r'final number is at least (-?\d+)', x_pub)[1])
        z = int(value >= bound)
    fields = re.findall(r'If z = ([01]) and e = (\w+), choose option ([A-D])', x_pub)
    table = [[None]*4 for _ in range(2)]
    for zz, team, letter in fields: table[int(zz)][TEAMS.index(team)] = ord(letter)-65
    if any(sorted(row) != list(range(4)) for row in table):
        raise ValueError('Public decision table is not an invertible permutation')
    return {'z_pub': z, 'T': table}


def expected_team(x_pub, commitment):
    public = public_constraints(x_pub)
    return public['T'][public['z_pub']].index(commitment_route(commitment))


def donor_map(entries):
    order = sorted(entries, key=lambda e: hashlib.sha256(e['group_id'].encode()).hexdigest())
    if len(order) < 2: raise ValueError('Need at least two groups for shuffled-z')
    mapping = {}
    for e, donor in zip(order, order[1:]+order[:1]):
        if e['group_id'] == donor['group_id']: raise ValueError('Shuffled-z group leakage')
        mapping[e['id']] = donor['id']; mapping[e['cf_id']] = donor['cf_id']
    return mapping


def value_records(example):
    """Oracle location is used only in the explicitly requested localization diagnostic."""
    ticket = re.search(r'Ticket: (\w+)', example.x_pub)[1]
    team = TEAMS[example.meta['e_priv']]
    return [i+1 for i, line in enumerate(example.x_priv.splitlines())
            if re.search(rf'ticket {re.escape(ticket)}(?=[: ])', line)
            and re.search(r'\bteam '+re.escape(team)+r'\b', line)]


def record_intervention(source, record_ids, mode):
    """Remove entire feature vectors, including tail vectors overlapping removed records."""
    record_ids = torch.as_tensor(record_ids, device=source['segments'].device)
    hit = (source['segments'][..., None] == record_ids).any(-1) & (source['kinds'] == 0)
    if not hit.any(): raise ValueError('No matching record features')
    if mode == 'only': keep = hit
    elif mode == 'remove':
        bounds = source['token_ranges'][hit]
        ranges = source['token_ranges']
        overlap = ((ranges[..., 0, None] < bounds[:, 1]) &
                   (ranges[..., 1, None] > bounds[:, 0])).any(-1)
        keep = ~overlap
    else: raise ValueError(mode)
    result = {k: v.clone() for k, v in source.items()}
    result['mask'] &= keep
    if not result['mask'].any(): raise ValueError('Empty record intervention')
    return result


def summarize_groups(rows, predictions):
    out = {}
    for r, p in zip(rows, predictions):
        key = (r['group_id'], r['task'], r['category'])
        out.setdefault(key, []).append(int(int(np.argmax(p)) == r['label']))
    return [{'group_id': g, 'task': t, 'category': c, 'accuracy': float(np.mean(v)), 'n': len(v)}
            for (g, t, c), v in out.items()]

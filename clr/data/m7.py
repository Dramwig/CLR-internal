"""Generator-derived semantic labels and group-safe M7 examples.

The intervention changes the one private semantic variable e. Its supporting
records can involve several team fields; irrelevant ticket records, public T
and z remain unchanged. This is not a claim of a single raw-event edit.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import numpy as np

TEAMS=('Amber','Birch','Cedar','Dune')
COMMITMENT_TEMPLATES=(
    'COMMITMENT: choose Route {route}.',
    'COMMITMENT: select Route {route}.',
    'COMMITMENT: the conclusion is Route {route}.')

def format_commitment(episode_id,route):
    variant=int(hashlib.sha256(episode_id.encode()).hexdigest(),16)%len(COMMITMENT_TEMPLATES)
    return COMMITMENT_TEMPLATES[variant].format(route=chr(65+route))
TEMPLATES={
 'A1': ['Does this message encode private team {fact}?', 'Is {fact} the private team represented in the message?',
        'Does the latent message contain the fact e = {fact}?', 'According to the message, is private variable e equal to {fact}?',
        'Is the private fact in this message that the team is {fact}?', 'Does the transmitted state indicate private team {fact}?'],
 'A2': ['Does the message support conclusion {fact}?', 'Is {fact} the route supported by this latent message?',
        'According to the message, is the final route {fact}?', 'Does the latent evidence justify choosing {fact}?',
        'Is conclusion {fact} consistent with the message content?', 'Does this transmission imply final route {fact}?'],
 'A3': ['Does the message agree with the visible statement: {commitment}', 'Is the message consistent with this commitment: {commitment}',
        'Does this visible conclusion match the latent content: {commitment}', 'Is this commitment supported by the message: {commitment}',
        'Can this latent transmission justify the commitment: {commitment}', 'Do the message and this public conclusion agree: {commitment}'],
 'A4': ['Which team is private variable e in this message?', 'What is the private team encoded by the latent message?',
        'Read private variable e from this message: which team is it?', 'Which team does the message represent as its private fact?',
        'Identify the private team carried by the transmitted state.', 'According to the latent evidence, what team is e?']}


def counterfactual(example):
    from latdef.data.synthetic_v2 import solve_v2_text, oracle_fact_text
    old=example.meta['e_priv']; new=(old+1)%4
    ticket=re.search(r'Ticket: (\w+)',example.x_pub).group(1)
    lines=[]
    for line in example.x_priv.splitlines():
        if re.search(rf'ticket {re.escape(ticket)}(?=[: ])',line):
            # One semantic e intervention, preserving event counts/times/points and nuisance logs.
            line=re.sub(r'(?<=team )(Amber|Birch|Cedar|Dune)',
                        lambda m:TEAMS[new] if m[0]==TEAMS[old] else TEAMS[old] if m[0]==TEAMS[new] else m[0],line)
        lines.append(line)
    e=replace(example,id=example.id+'-cf',x_priv='\n'.join(lines),
              label=example.meta['T'][example.meta['z_pub']][new],
              meta={**example.meta,'e_priv':new,'intervention':'e_priv_only_team_swap_on_target_records',
                    'parent_id':example.id,'oracle_message':oracle_fact_text(example.question_type,ticket,TEAMS[new])})
    if solve_v2_text(e)!=e.label:raise AssertionError('Counterfactual symbolic label mismatch')
    if e.x_pub!=example.x_pub or e.meta['group_id']!=example.meta['group_id']:raise AssertionError('Invalid counterfactual group')
    return e


def split_examples(config,split,count):
    """Fresh M7 seed; unchanged frozen generator recipe; never called for test before final lock."""
    from latdef.data.synthetic_v2 import load_v2_config, generate_v2_split, solve_v2_text
    cfg=replace(load_v2_config(Path(config['idea_a'])/'configs/synthetic/v2.yaml'),seed=config['data_seed'])
    source='train' if split=='reader_train' else 'test' if split=='test' else 'val'
    offsets={'reader_train':0,'reader_val':0,'cal_A':config['counts']['reader_val'],
             'cal_B':config['counts']['reader_val']+config['counts']['cal_A'],'test':0}
    offset=offsets[split]
    for e in generate_v2_split(cfg,source,group_limit=offset+count):
        i=e.meta['group_index']
        if i<offset or e.meta['alpha']!=1 or e.meta['d_pub']!=(i//12)%3:continue
        if solve_v2_text(e)!=e.label:raise AssertionError('Base symbolic label mismatch')
        yield e


def commitment_route(text):
    m=re.fullmatch(r'COMMITMENT: (?:choose|select|the conclusion is) Route ([ABCD])\.',text)
    if not m:raise ValueError('Invalid fixed commitment format')
    return ord(m[1])-65

def matched_false_values(values):
    """A marginal-preserving derangement, provided no value occupies > half."""
    from collections import Counter
    counts=Counter(values);shift=max(counts.values())
    if shift*2>len(values):raise ValueError('Cannot align false-fact marginals without matching positives')
    order=sorted(range(len(values)),key=lambda i:(values[i],i))
    sorted_values=[values[i] for i in order]
    rotated=sorted_values[shift:]+sorted_values[:shift]
    result=[None]*len(values)
    for i,v in zip(order,rotated):result[i]=v
    if any(a==b for a,b in zip(values,result)):raise AssertionError('False-fact derangement failed')
    return result


def qa_rows(base,source,commitment,category,template_ids=(0,1,2,3)):
    """Only explicit typed questions/options/commitment enter models. No raw source text."""
    result=[]
    private=source['e']; other=source['paired_e']; route=source['route']; other_route=source['paired_route']
    for tid in template_ids:
        specs=[('A1',TEAMS[private],1),('A1',TEAMS[other],0),
               ('A2',f'Route {chr(65+route)}',1),('A2',f'Route {chr(65+other_route)}',0),
               ('A3','',int(route==commitment_route(commitment))),('A4','',private)]
        for task,fact,label in specs:
            options=list(TEAMS) if task=='A4' else ['No','Yes']
            q=TEMPLATES[task][tid].format(fact=fact,commitment=commitment)
            result.append({'episode_id':base['id'],'group_id':base['group_id'],'message_id':source['id'],
                           'category':category,'task':task,'template_id':tid,'question_text':q,
                           'options':options,'label':label})
    return result


def corrupt(z,kind,seed,relative_noise=.1):
    """Virtual controls preserve the actual z interface; no substituted input vectors."""
    import torch
    out=z.clone()
    if kind=='truncate':return out[:max(1,len(out)//2)]
    if kind=='zero_layer':return torch.zeros_like(out) # One saved layer; transmitted full-layer zero is handled by source.
    if kind=='noise':
        g=torch.Generator().manual_seed(seed)
        eps=torch.randn(out.shape,generator=g,dtype=torch.float32)
        eps*=relative_noise*out.float().norm()/eps.norm().clamp_min(1e-12)
        return (out.float()+eps).half()
    raise ValueError(kind)


def threshold(clean_scores,alpha):
    """Reject scores > q. Conservative finite-sample upper quantile with infinity boundary."""
    import math
    rank=math.ceil((len(clean_scores)+1)*(1-alpha))
    return float('inf') if rank>len(clean_scores) else float(np.sort(clean_scores)[rank-1])

"""M8c direct-lookup protocol on top of the frozen synthetic-v2 episodes."""
from dataclasses import replace
import re
from clr.data.m7 import TEAMS

def direct_example(example):
    # The direct task retains the ticket and definition, with Team candidates.
    ticket = re.search(r'^Ticket: (\w+)', example.x_pub).group(1)
    definition = re.search(r'(?m)^Private fact definition: (.+)$', example.x_pub).group(1)
    public = f'Ticket: {ticket}\nPrivate fact definition: {definition}\nQuestion: Which team is the private variable e?'
    public = public + '\nOptions:\n' + '\n'.join(
        f'{chr(65+i)}. {team}' for i, team in enumerate(TEAMS))
    return replace(example, x_pub=public,
                   options=list(TEAMS), label=int(example.meta['e_priv']),
                   meta={**example.meta, 'm8c_task':'direct-lookup', 'm8c_parent_id':example.id})

def short_example(example):
    """Remove nuisance-ticket records; retain every target record verbatim."""
    ticket = re.search(r'^Ticket: (\w+)',example.x_pub).group(1)
    lines=example.x_priv.splitlines()
    keep=[i for i,line in enumerate(lines) if re.search(rf'ticket {re.escape(ticket)}(?=[: ])',line)]
    if not keep:raise ValueError('No target-ticket records')
    return replace(example,x_priv='\n'.join(lines[i] for i in keep),
        meta={**example.meta,'critical_record_indices':list(range(len(keep))),
              'm8c_log_view':'target_records_only','m8c_kept_indices':keep})

def direct_prepared(relay, example, prefix):
    from latdef.handoff.base import S2Input
    from latdef.inject.cache import left_pad
    import torch
    prompts, letters, _, _ = relay.s2._prepare([S2Input(example.x_pub, tuple(example.options))], relay.arm, prefix)
    ids, mask = left_pad([prompts[0] + relay.s2.direct_suffix], relay.s2.tokenizer.pad_token_id, relay.s2.device)
    return ids, mask, torch.tensor(letters, device=ids.device)

def direct_receiver(relay, prefix, prepared, cost):
    from latdef.inject.cache import prefix_forward
    ids, mask, letters = prepared
    out = prefix_forward(relay.s2.model, ids, mask, prefix=prefix, logits_to_keep=1)
    cost.forward_calls += 1
    return out.logits[:, -1].float().gather(1, letters).log_softmax(-1)[0]

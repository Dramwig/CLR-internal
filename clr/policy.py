"""Shared CODI continuation policy for M1 labels and M5 search.

z_0 is the last-layer hidden state at BOT (token index pad_to); z_t for
1<=t<=T is the output at continuous-input position pad_to+t. The transition
z_t -> z_(t+1) consumes prj(noise(z_t)). Thus from_step=t in M0.1's zero-based
consumed-input convention perturbs transitions t+1..T, never the saved z_t.
Outputs contain unmodified observed states AND actually consumed noisy feedback;
these are distinct objects. Final z_T is never consumed by another latent step.
K lanes replicate the original batch1 KV; no new padding/positions are created.
"""
from dataclasses import dataclass
import math
import torch
from clr.sources.common import RelativeNoise

@dataclass
class PrefixStates:
    source: object
    problem_id: str
    pad_to: int
    states: list
    layers: list
    caches: list
    feedback: list


def _repeat(cache,n):
    # GPT2 legacy cache is immutable; cloning prevents later callers aliasing it.
    return tuple(tuple(x.expand(n,*x.shape[1:]).clone() for x in pair) for pair in cache)


def _check(problem,prefix,t):
    if problem['id']!=prefix.problem_id or problem['pad_to']!=prefix.pad_to:
        raise ValueError('Prefix belongs to a different problem or cohort context')
    if not isinstance(t,int) or not 0<=t<len(prefix.states):raise ValueError('Invalid step')


@torch.inference_mode()
def prepare(problem,source):
    """Run clean batch1 fp32 context once; retain all seven states and exact KV."""
    c=source.config;m=source.model
    if c['dtype']!='float32' or c['position_ids']!='implicit_absolute' or c['carry_attention_mask']:
        raise ValueError('Only frozen CODI GPT2 fp32 protocol is supported')
    row=source.tokenizer.encode(problem['question'].strip().replace('  ',' '))
    width=problem['pad_to'];pad=width-len(row)
    if pad<0 or width+1+c['num_latent']+1+c['max_new_tokens']>m.codi.config.max_position_embeddings:
        raise ValueError('Invalid context length; no truncation allowed')
    ids=torch.tensor([[m.pad_token_id]*pad+row+[m.bot_id]],device=source.device)
    mask=torch.tensor([[0]*pad+[1]*(len(row)+1)],device=source.device)
    out=source._forward(input_ids=ids,attention_mask=mask,use_cache=True,output_hidden_states=True)
    states=[];layers=[];caches=[];feedback=[]
    for t in range(c['num_latent']+1):
        z=out.hidden_states[-1][:,-1,:].unsqueeze(1)
        states.append(z.clone());layers.append(torch.stack([h[0,-1].detach().cpu() for h in out.hidden_states]))
        caches.append(tuple(tuple(x for x in pair) for pair in out.past_key_values))
        if t<c['num_latent']:
            feedback.append(z.clone())
            out=source._forward(inputs_embeds=m.prj(z),past_key_values=out.past_key_values,use_cache=True,output_hidden_states=True)
    return PrefixStates(source,problem['id'],width,states,layers,caches,feedback)


@torch.inference_mode()
def _decode(source,cache,temperature,seeds):
    if not math.isfinite(temperature) or temperature<0:raise ValueError('Invalid temperature')
    m=source.model;n=len(seeds);emb=m.get_embd(m.codi,m.model_name)
    inp=emb(torch.tensor([m.eot_id],device=source.device)).unsqueeze(0).expand(n,-1,-1)
    generators=[torch.Generator(device=source.device).manual_seed(s) for s in seeds]
    tokens=[[] for _ in seeds];finished=[False]*n
    for _ in range(source.config['max_new_tokens']):
        out=source._forward(inputs_embeds=inp,past_key_values=cache,use_cache=True)
        cache=out.past_key_values;logits=out.logits[:,-1,:m.codi.config.vocab_size-1]
        if temperature==0:tok=logits.argmax(-1)
        else:
            probs=torch.softmax(logits.float()/temperature,dim=-1)
            tok=torch.cat([torch.multinomial(p,1,generator=g) for p,g in zip(probs,generators)])
        for i,k in enumerate(tok.tolist()):
            if not finished[i]:tokens[i].append(k);finished[i]=k==source.tokenizer.eos_token_id
        if all(finished):break
        inp=emb(tok[:,None])
    return [{'answer':source.tokenizer.decode(x,skip_special_tokens=True),'hit_token_limit':not done} for x,done in zip(tokens,finished)]


@torch.inference_mode()
def early_exit(problem,prefix_states,t):
    _check(problem,prefix_states,t)
    return _decode(prefix_states.source,prefix_states.caches[t],0,[0])[0]


@torch.inference_mode()
def rollout(problem,prefix_states,t,sigma,temperature,seed):
    """Unique continuation engine; integer seed -> one result, list -> K lanes.

    PrefixStates retains batch1 KV and fp32 states; storage quantization is not
    silently used for label generation. For explicit persisted-state audits a
    caller may substitute fp16-roundtripped states while retaining the same KV.
    Returned states are z_t..z_T (post-forward observations); feedback[j] is
    noise(z_(t+j)) actually fed through prj. Metadata preserves this distinction.
    """
    _check(problem,prefix_states,t)
    seeds=[seed] if isinstance(seed,int) else list(seed)
    if not seeds or any(type(s)!=int for s in seeds):raise ValueError('Require integer seed(s)')
    s=prefix_states.source;m=s.model;n=len(seeds)
    noise=[RelativeNoise({'from_step':t,'sigma':sigma,'seed':r},s.device) for r in seeds]
    cache=_repeat(prefix_states.caches[t],n)
    z=prefix_states.states[t].expand(n,-1,-1).clone()
    states=[z.detach().cpu()];feedback=[]
    for j in range(t,s.config['num_latent']):
        consumed=z if sigma==0 else torch.cat([f(z[i:i+1],j) for i,f in enumerate(noise)])
        feedback.append(consumed.detach().cpu())
        out=s._forward(inputs_embeds=m.prj(consumed),past_key_values=cache,use_cache=True,output_hidden_states=True)
        cache=out.past_key_values;z=out.hidden_states[-1][:,-1,:].unsqueeze(1)
        states.append(z.detach().cpu())
    answers=_decode(s,cache,temperature,seeds)
    for i,a in enumerate(answers):
        a.update(states=torch.stack([x[i,0] for x in states]),feedback=torch.stack([x[i,0] for x in feedback]) if feedback else torch.empty(0,z.shape[-1]),
                 t0=t,seed=seeds[i],sigma=sigma,temperature=temperature,physical_batch=n)
    return answers[0] if isinstance(seed,int) else answers

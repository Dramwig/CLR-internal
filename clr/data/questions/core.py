"""Behavior-grounded questions; source problem text never enters reader inputs."""
from functools import lru_cache
from decimal import Decimal, InvalidOperation
import hashlib,random
import numpy as np

def seed(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:15],16)

@lru_cache(maxsize=200000)
def number(x):
    from clr.data.labels import numeric_answer
    try:
        d=Decimal(str(x))
        if d.is_finite():return str(d.normalize())
    except InvalidOperation:pass
    v=numeric_answer(str(x))
    if v is None:return '<invalid>'
    return str(Decimal(str(v)).normalize())

def choices(target,candidates,rng):
    target=number(target);pool=list(dict.fromkeys(number(v) for v in candidates))
    pool=[v for v in pool if v!=target];rng.shuffle(pool)
    if len(pool)<3:raise ValueError('Insufficient distinct observed distractors')
    options=[target]+pool[:3];rng.shuffle(options)
    return options,options.index(target)

def matched_values(problem_values,split_seed):
    """Reciprocal cross-problem pairing preserves exact positive/negative marginals.

    Unpairable problems are explicitly counted, never assigned a false negative.
    One randomly selected positive per parsable problem; paired donors must not
    be in the recipient's complete intermediate-value set.
    """
    rng=random.Random(split_seed);ids=sorted(problem_values);rng.shuffle(ids)
    chosen={p:rng.choice(sorted(problem_values[p])) for p in ids}
    pending=list(ids);result={}
    while pending:
        a=pending.pop();match=None
        for j,b in enumerate(pending):
            if chosen[a] not in problem_values[b] and chosen[b] not in problem_values[a]:match=j;break
        if match is None:continue
        b=pending.pop(match);result[a]=(chosen[a],chosen[b]);result[b]=(chosen[b],chosen[a])
    return result,sorted(set(ids)-set(result))

def numeric_features(values):
    return _numeric_features(tuple(values))

@lru_cache(maxsize=200000)
def _numeric_features(values):
    out=[]
    for v in values:
        try:
            x=float(Decimal(str(v)));valid=np.isfinite(x)
        except (ValueError,InvalidOperation):x=0.;valid=False
        if not valid:x=0.
        l=np.log1p(abs(x));out.append([1.,float(valid),np.sign(x),min(l,30)/10,float(x==0),float(x.is_integer()),np.sin(l),np.cos(l)])
    return np.asarray(out,dtype=np.float32)

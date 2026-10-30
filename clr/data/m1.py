"""M1 data identities, equation parsing and quantization diagnostics."""
import hashlib,re
from decimal import Decimal,InvalidOperation
import torch


def rng_seed(seed,pid,kind,t=0,k=0):
    return int.from_bytes(hashlib.sha256(f'{seed}:{pid}:{kind}:{t}:{k}'.encode()).digest()[:8],'big')%(2**63-1)


def intermediate_values(cot):
    """Extract explicit equation RHS only; no inference of hidden reasoning.

    A missing/partially unparsed equation is marked failed, never drops a row.
    Scientific notation, signed decimal, comma-separated numbers are supported.
    """
    equations=re.findall(r'<<([^<>]+)>>',cot or '')
    values=[];errors=[]
    for equation in equations:
        rhs=equation.rsplit('=',1)[-1].strip().replace(',','')
        try:
            if '=' not in equation:raise InvalidOperation
            value=Decimal(rhs)
            if not value.is_finite():raise InvalidOperation
            values.append(str(value.normalize()))
        except InvalidOperation:errors.append(equation)
    return {'intermediate_values':list(dict.fromkeys(values)), 'equation_count':len(equations),
            'cot_parse_failed':not equations or bool(errors),'unparsed_equations':errors}


def quantization(x):
    original=x.float();back=original.half().float();delta=back-original
    if not torch.isfinite(back).all():raise ValueError('Nonfinite fp16 state')
    return {'max_abs':delta.abs().max().item(),'relative_l2':(delta.norm()/original.norm().clamp_min(1e-30)).item(),
            'max_vector_relative_l2':(delta.norm(dim=-1)/original.norm(dim=-1).clamp_min(1e-30)).max().item(),
            'squared_error':delta.square().sum().item(),'elements':original.numel()}

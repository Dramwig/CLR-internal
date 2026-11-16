"""Documented full-prefill adaptations of published KV perturbation families."""
import torch
from clr.audit.m8a_perturb import Perturbation,to_vectors,from_vectors


@torch.no_grad()
def baseline(prefix,kind,epsilon,metadata,layers,seed,direction=None):
    from latdef.inject.cache import PrefixKV
    generator=torch.Generator(device=prefix.mask.device).manual_seed(seed)
    if kind in ['E0_scale8','E0_sign','E0_randomnorm']:
        pairs=[]
        for pair in prefix.layers:
            tensors=[]
            for original in pair:
                if kind=='E0_scale8':value=original.float()*8
                elif kind=='E0_sign':value=-original.float()
                else:
                    value=torch.randn(original.shape,device=original.device,generator=generator)
                    value*=prefix.mask[:,None,:,None]
                    value*=original.float().norm()/value.norm().clamp_min(1e-20)
                tensors.append(value.to(original.dtype))
            pairs.append(tuple(tensors))
        return PrefixKV(tuple(pairs),prefix.mask,prefix.active_layers,prefix.gates)
    attack=Perturbation(prefix,metadata,'E0',layers)
    radius=1. if epsilon is None else epsilon
    for layer in attack.layers:
        if kind=='E0_noise':d=torch.randn(attack.base[layer].shape,device=prefix.mask.device,generator=generator)
        elif kind=='E0_diffmean':
            if direction is None:raise ValueError('DiffMean extraction has no retained val pairs')
            d=direction[layer][None,None,:].expand_as(attack.base[layer]).clone()
        else:raise ValueError(kind)
        d*=prefix.mask[...,None]
        d*=radius*attack.norm[layer]/d.norm().clamp_min(1e-20)
        attack.delta[layer]=d
    return attack.build(False)


@torch.no_grad()
def ratios(clean,changed):
    return {str(l):float((to_vectors(pair).float()-to_vectors(clean.layers[l]).float()).norm()/to_vectors(clean.layers[l]).float().norm().clamp_min(1e-20))
            for l,pair in enumerate(changed.layers)}

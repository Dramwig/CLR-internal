"""Source contracts and relative noise; no implicit model/device substitutions."""
import importlib.util
import math
from pathlib import Path
import torch

class RelativeNoise:
    """Zero-based inclusive from_step: 0 perturbs every consumed latent step.

    Draw eps ~ N(0,I) independently per step, normalize by sqrt(d), NOT by
    the realized eps norm. Calculate norms/noise in fp32, cast back once.
    A private device generator never changes the source's global RNG.
    """
    def __init__(self, perturb, device):
        self.spec = perturb
        if perturb is None: return
        if set(perturb) != {'from_step', 'sigma', 'seed'}: raise ValueError('Invalid perturb keys')
        if type(perturb['from_step']) is not int or perturb['from_step'] < 0: raise ValueError('from_step must be nonnegative int')
        if not math.isfinite(perturb['sigma']) or perturb['sigma'] < 0: raise ValueError('Invalid sigma')
        if type(perturb['seed']) is not int: raise ValueError('seed must be int')
        self.generator = torch.Generator(device=device).manual_seed(perturb['seed'])
    def __call__(self, z, step):
        if self.spec is None or step < self.spec['from_step'] or self.spec['sigma'] == 0: return z
        x=z.float()
        eps=torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self.generator)
        return (x+self.spec['sigma']*x.norm(dim=-1,keepdim=True)*eps/math.sqrt(x.shape[-1])).to(z.dtype)


def import_file(path, name):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def check_device(config):
    device=torch.device(config['device'])
    if device.type=='cuda':
        torch.cuda.set_device(device)
        if config['dtype']=='bfloat16' and not torch.cuda.is_bf16_supported(): raise RuntimeError('bf16 unsupported')
        free,_=torch.cuda.mem_get_info(device)
        if free < config['min_free_gib']*1024**3: raise RuntimeError(f'GPU free {free/1024**3:.2f} GiB < required {config["min_free_gib"]} GiB')
    return device


def question(problem):
    if isinstance(problem,str): return problem
    # Gold labels, steps, and metadata never enter the source.
    return problem['question']


def max_latent_difference(a,b):
    if len(a)!=len(b): raise ValueError('Latent step count differs')
    if not a: raise ValueError('No latent states captured')
    if any(x.shape!=y.shape for x,y in zip(a,b)): raise ValueError('Latent shapes differ')
    return max((x.float()-y.float()).abs().max().item() for x,y in zip(a,b))

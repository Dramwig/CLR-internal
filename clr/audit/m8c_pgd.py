"""PGD with one shared receiver forward for success checking and gradients."""
import torch
from clr.audit.m8a_perturb import Perturbation
from clr.audit.m8c_core import direct_receiver
from clr.audit.m8c_reader import rng_seed

def optimize(base,metadata,prepared,entry,target,kind,epsilon,design,seed,lam,
             relay,transform,observe,audits,cost,config,clean_feature=None,namespace=''):
    attack=Perturbation(base,metadata,kind,config['layers']);selection=None
    auditor=audits[design,seed] if design else None
    maximum=config['steps']
    for updates in range(maximum+1):
        cost.check();attack.leaf();prefix=attack.build()
        if design=='R4':lp,state=observe(prefix,prepared,cost)
        else:lp=direct_receiver(relay,prefix,prepared,cost);state=None
        accepted=False
        if updates:
            with torch.no_grad():
                accepted=int(lp.argmax())==target
                if auditor and (accepted or updates==maximum):
                    seq=state if design=='R4' else transform(prefix,prefix.mask,metadata)
                    selection=sum(float(auditor.loss(seq,auditor.prepare(entry,rng_seed(entry['id'],design,seed,namespace+'selection',j),16))) for j in range(config['selection_panels']))/config['selection_panels']
                    accepted=accepted and selection<=auditor.cal['thresholds']['16']['0.05']
            if (accepted and config['early_success']) or updates==maximum:
                if state is not None:state.values=state.values.detach()
                return attack,updates,selection,accepted and config['early_success'],lp.detach(),state
        objective=lp[target]
        if auditor:
            seq=state if design=='R4' else transform(prefix,prefix.mask,metadata)
            objective=objective-lam*auditor.loss(seq,auditor.prepare(entry,rng_seed(entry['id'],design,seed,namespace+'EOT',updates+1),config['eot_k']))
        gradients=torch.autograd.grad(objective,list(attack.delta.values()))
        attack.step(gradients,epsilon,maximum,config['step_multiplier'])
        del gradients,objective,lp,state,prefix
        if kind=='E2':attack.enforce_fp16_identity(transform,metadata,clean_feature)
    raise AssertionError('Unreachable PGD termination')

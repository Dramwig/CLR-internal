"""M5b controlled inputs. Frozen baseline/CLR definitions are imported unchanged."""
import torch
from torch import nn
from clr.baselines.probes import Probe
from clr.models.clr_v1 import Lite

class PrefixProbe(Probe):
    def __init__(self,qtype,hidden=128):
        super().__init__('B3',qtype,hidden)
        self.encoder=nn.Linear(1536+64+8,hidden)


def current_only(prefix,mask):
    ii=mask.long().sum(1)-1
    z=prefix[torch.arange(len(prefix),device=prefix.device),ii,None]
    return z,torch.ones((len(prefix),1),dtype=torch.bool,device=prefix.device)


def prefix_features(bank,mean,scale,device):
    """Concat prefix mean and current observation, both in the original z norm."""
    z=bank.z.to(device);ii=bank.prefix_indices.to(device);mask=bank.prefix_mask.to(device)
    current=(z-mean)/scale
    # Avoid allocating all [num_states,7,768] observations at once.
    result=[]
    for start in range(0,len(z),4096):
        zz=(z[ii[start:start+4096]]-mean)/scale
        m=mask[start:start+4096].float().unsqueeze(-1)
        result.append(torch.cat([(zz*m).sum(1)/m.sum(1),current[start:start+4096]],1))
    return torch.cat(result)

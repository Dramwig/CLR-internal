"""Native-dimensional M12 ablations; import frozen components unchanged."""
import torch
from torch import nn
from clr.models.m12 import AblationLite
from clr.models.m6 import NativeLite
from clr.data.m6 import Bank, rows

class NativeAblation(AblationLite):
    def __init__(self,ablation,dim=2048):
        # Identical native-dimension initialization order to NativeLite.
        super().__init__('main')
        if dim!=768:
            self.adapter[0]=nn.Linear(dim,256)
            self.heads=nn.ModuleDict({k:nn.Sequential(nn.Linear(256+dim+8,256),nn.GELU(),nn.Linear(256,1)) for k in ['score','yesno','choice']})
            self.state_mean=torch.zeros(dim);self.state_scale=torch.ones(dim)
        self.ablation=ablation
        if ablation=='A2':self.heads=nn.ParameterDict({k:nn.Parameter(torch.zeros(())) for k in ['score','yesno','choice']})
        if ablation not in ['main','A1','A2','A4']:raise ValueError(ablation)

def make_model(c):return NativeAblation(c['ablation'],c['dim'])

def PrefixBank(c,split,phase,ids=None):return Bank(c['root'],split,phase,ids)

def rows_for(c,split,phase,qtypes=None,heldout=False,ids=None):
    if heldout:raise ValueError('This task reports seen templates only')
    return rows(c['qa_root'],split,phase,ids=ids,qtypes=qtypes)

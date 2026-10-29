"""CODI-LLaMA v2: retain known unused checkpoint tensors without relaxing loading.

The pinned upstream test.py calls only codi/prj and ignores four dynamic_cls
keys via strict=False. We retain those exact tensors as inert buffers; no
classifier architecture or forward behavior is invented. Every other key must
match the official model class, and final load_state_dict is strict=True.
"""
from pathlib import Path
from types import SimpleNamespace
import torch
from peft import LoraConfig,TaskType
from .common import check_device,import_file
from .codi_v2 import CODISourceV2

AUXILIARY_KEYS=frozenset({'dynamic_cls.1.bias','dynamic_cls.1.weight','dynamic_cls.3.bias','dynamic_cls.3.weight'})

def load_preserving_auxiliary(model,state,allowed):
    if set(allowed)!=AUXILIARY_KEYS:raise ValueError('Explicit four-key checkpoint allowlist required')
    expected=set(model.state_dict());actual=set(state)
    missing=expected-actual;extra=actual-expected
    if missing or extra!=AUXILIARY_KEYS:raise RuntimeError(f'Unrecognized checkpoint schema: missing={sorted(missing)}, extra={sorted(extra)}')
    if hasattr(model,'dynamic_cls'):raise RuntimeError('Official architecture now defines dynamic_cls; re-audit required')
    outer=torch.nn.Module()
    for index in ['1','3']:
        module=torch.nn.Module()
        for name in ['weight','bias']:
            value=state[f'dynamic_cls.{index}.{name}']
            module.register_buffer(name,torch.empty_like(value))
        outer.add_module(index,module)
    model.add_module('dynamic_cls',outer)
    return model.load_state_dict(state,strict=True)

class CODILlamaSourceV2(CODISourceV2):
    def __init__(self,config):
        if config['family']!='llama' or config['dtype']!='bfloat16':raise ValueError('Audited loader is LLaMA bf16 only')
        self.config=dict(config);self.device=check_device(config)
        mod=import_file(Path(config['code_path'])/'src/model.py','clr_official_codi_llama_v2')
        args=mod.ModelArguments(model_name_or_path=config['base_path'],train=False,lora_init=True)
        train=SimpleNamespace(bf16=True,num_latent=config['num_latent'],use_lora=True,use_prj=True,prj_no_ln=False,prj_dropout=0.,prj_dim=config['prj_dim'],print_loss=False,ref_loss_factor=1.,distill_loss_div_std=True,distill_loss_type='l2',distill_loss_factor=1.,fix_attn_mask=False)
        targets=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj']
        self.model=mod.CODI(args,train,LoraConfig(task_type=TaskType.CAUSAL_LM,r=128,lora_alpha=32,lora_dropout=.1,target_modules=targets))
        state=torch.load(config['checkpoint'],map_location='cpu',weights_only=True)
        load_preserving_auxiliary(self.model,state,config['preserved_unused_checkpoint_keys'])
        self.model.codi.tie_weights();self.model.to(device=self.device,dtype=torch.bfloat16).eval().requires_grad_(False)
        self.tokenizer=self.model.tokenizer;self.tokenizer.padding_side='left';self.forward_calls=0

def load(config):return CODILlamaSourceV2(config)

"""Versioned LLaMA continuation: immutable legacy KV, observed output states.

M1 transition/noise/decode functions are imported. Only clean prefix capture is
generalized to audited bf16 LLaMA, with explicit legacy-cache conversion so
DynamicCache mutation cannot change previously captured prefixes.
"""
import torch
from transformers.cache_utils import DynamicCache
from clr.sources.codi_llama_v2 import CODILlamaSourceV2
from clr.policy import PrefixStates, rollout, early_exit

class LlamaM6(CODILlamaSourceV2):
    def _forward(self,**kw):
        if isinstance(kw.get('past_key_values'),tuple):
            kw['past_key_values']=DynamicCache.from_legacy_cache(kw['past_key_values'])
        out=super()._forward(**kw)
        if hasattr(out.past_key_values,'to_legacy_cache'):
            out.past_key_values=out.past_key_values.to_legacy_cache()
        return out

@torch.inference_mode()
def prepare(problem,source):
    c=source.config; m=source.model
    if c['family']!='llama' or c['dtype']!='bfloat16' or c['position_ids']!='implicit_absolute' or c['carry_attention_mask']:
        raise ValueError('M6 LLaMA requires the audited bf16/absolute/no-carry protocol')
    row=source.tokenizer.encode(problem['question'].strip().replace('  ',' ')); width=problem['pad_to']; pad=width-len(row)
    if pad<0 or width+8+c['max_new_tokens']>m.codi.config.max_position_embeddings: raise ValueError('Invalid frozen context; no truncation')
    ids=torch.tensor([[m.pad_token_id]*pad+row+[m.bot_id]],device=source.device)
    mask=torch.tensor([[0]*pad+[1]*(len(row)+1)],device=source.device)
    out=source._forward(input_ids=ids,attention_mask=mask,use_cache=True,output_hidden_states=True)
    states=[]; caches=[]; layers=[]; feedback=[]
    for t in range(c['num_latent']+1):
        z=out.hidden_states[-1][:,-1:,:]
        states.append(z.clone()); caches.append(tuple(tuple(x for x in pair) for pair in out.past_key_values))
        layers.append(torch.stack([h[0,-1].cpu() for h in out.hidden_states]))
        if t<c['num_latent']:
            feedback.append(z.clone()); out=source._forward(inputs_embeds=m.prj(z),past_key_values=out.past_key_values,use_cache=True,output_hidden_states=True)
    return PrefixStates(source,problem['id'],width,states,layers,caches,feedback)

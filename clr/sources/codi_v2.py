"""Versioned CODI protocol. Frozen v1 is imported for weight loading only.

Padding width is input context, never inferred from a runtime microbatch.
Official replay contexts derive from canonical upstream cohorts (128, original
order), using question token lengths only, before seeing predictions or labels.
No claim is made that this reproduces a new unseen problem without that context.
"""
from pathlib import Path
import torch
from .codi import CODISource
from .common import RelativeNoise,question

class CODISourceV2(CODISource):
    def __init__(self,config):
        initial={**config,'dtype':'bfloat16'}
        super().__init__(initial)
        if config['dtype']=='float32':
            self.model.float()
            self.model.load_state_dict(torch.load(Path(config['checkpoint']),map_location='cpu',weights_only=True),strict=True)
            self.model.codi.tie_weights()
        elif config['dtype']!='bfloat16':raise ValueError('v2 permits explicitly audited bf16 or fp32 only')
        self.config=dict(config)
        self.tokenizer.padding_side='left'

    @torch.inference_mode()
    def run_batch(self,problems,perturbs=None,decode=None,capture_latents=True):
        c=self.config;m=self.model;n=len(problems)
        if not n:raise ValueError('Empty batch')
        rows=[self.tokenizer.encode(question(p).strip().replace('  ',' ')) for p in problems]
        widths=[p.get('pad_to',len(r)) if isinstance(p,dict) else len(r) for p,r in zip(problems,rows)]
        if len(set(widths))!=1:raise ValueError('Different canonical padding contexts may not be merged')
        width=widths[0]
        if any(len(r)>width for r in rows):raise ValueError('Truncation forbidden')
        if width+1+c['num_latent']+1+c['max_new_tokens']>m.codi.config.max_position_embeddings:raise ValueError('Context budget exceeded')
        ids=torch.tensor([[m.pad_token_id]*(width-len(r))+r+[m.bot_id] for r in rows],device=self.device)
        mask=torch.tensor([[0]*(width-len(r))+[1]*(len(r)+1) for r in rows],device=self.device)
        perturbs=perturbs if perturbs is not None else [None]*n
        if len(perturbs)!=n:raise ValueError('Perturb count mismatch')
        noises=[RelativeNoise(p,self.device) for p in perturbs]
        decode=decode or {'temperature':0,'seeds':[0]*n}
        temp=decode['temperature']
        if temp<0:raise ValueError('Negative temperature')
        gens=[torch.Generator(device=self.device).manual_seed(s) for s in decode['seeds']]
        if len(gens)!=n:raise ValueError('Decode seed count mismatch')
        steps=[[] for _ in rows];start=self.forward_calls
        kwargs={}
        if c['position_ids']=='masked_relative':kwargs['position_ids']=(mask.cumsum(-1)-1).clamp_min(0)
        elif c['position_ids']!='implicit_absolute':raise ValueError('Unknown position policy')
        out=self._forward(input_ids=ids,attention_mask=mask if c['prefill_attention_mask'] else None,use_cache=True,output_hidden_states=True,**kwargs)
        for t in range(c['num_latent']):
            # Preserve upstream tensor strides when noise is inactive. A cat()
            # here changes the bf16 linear kernel even for sigma=0.
            z=out.hidden_states[-1][:,-1,:].unsqueeze(1)
            if any(p and p['sigma']>0 and t>=p['from_step'] for p in perturbs):
                z=torch.cat([noise(z[i:i+1],t) for i,noise in enumerate(noises)])
            if capture_latents:
                obs=(z if c['layer']==-1 else out.hidden_states[c['layer']][:,-1:,:]).detach().cpu()
                for i in range(n):steps[i].append(obs[i,0].clone())
            mask=torch.cat([mask,torch.ones((n,1),device=self.device,dtype=mask.dtype)],dim=1)
            kwargs={}
            if c['position_ids']=='masked_relative':kwargs['position_ids']=mask.sum(-1,keepdim=True)-1
            out=self._forward(inputs_embeds=m.prj(z),past_key_values=out.past_key_values,use_cache=True,output_hidden_states=True,
                attention_mask=mask if c['carry_attention_mask'] else None,**kwargs)
        cache=out.past_key_values;emb=m.get_embd(m.codi,m.model_name)
        # Upstream expands a single EOT embedding across the batch (stride 0).
        inp=emb(ids.new_tensor([m.eot_id])).unsqueeze(0).expand(n,-1,-1)
        tokens=[[] for _ in rows];finished=[False]*n
        for _ in range(c['max_new_tokens']):
            mask=torch.cat([mask,torch.ones((n,1),device=self.device,dtype=mask.dtype)],dim=1)
            kwargs={}
            if c['position_ids']=='masked_relative':kwargs['position_ids']=mask.sum(-1,keepdim=True)-1
            out=self._forward(inputs_embeds=inp,past_key_values=cache,use_cache=True,attention_mask=mask if c['carry_attention_mask'] else None,**kwargs)
            cache=out.past_key_values;logits=out.logits[:,-1,:m.codi.config.vocab_size-1]
            if temp==0:tok=logits.argmax(-1)
            else:
                probs=torch.softmax(logits.float()/temp,dim=-1)
                tok=torch.cat([torch.multinomial(p,1,generator=g) for p,g in zip(probs,gens)])
            for i,token in enumerate(tok.tolist()):
                if not finished[i]:tokens[i].append(token);finished[i]=token==self.tokenizer.eos_token_id
            if all(finished):break
            inp=emb(tok[:,None])
        return [{'answer':self.tokenizer.decode(tokens[i],skip_special_tokens=True),'latent_steps':steps[i],
          'meta':{'protocol_version':'codi_v2','dtype':c['dtype'],'physical_batch_size':n,'pad_to':width,
                  'pad_count':width-len(rows[i]),'batch_forward_calls':self.forward_calls-start,
                  'generated_tokens':len(tokens[i]),'hit_token_limit':not finished[i],'perturb':perturbs[i],
                  'temperature':temp,'sampling_filter':'full_vocab_no_top_k_or_top_p'}} for i in range(n)]

    def run(self,problem,perturb=None,decode=None,capture_latents=True):
        return self.run_batch([problem],[perturb],decode,capture_latents)[0]

def load(config):return CODISourceV2(config)

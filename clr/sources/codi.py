"""CODI using the official architecture and checkpoint, greedy official decoding."""
from pathlib import Path
from types import SimpleNamespace
import torch
from peft import LoraConfig, TaskType
from .common import RelativeNoise, check_device, import_file, question

class CODISource:
    """latent_steps[t] is the hidden vector producing continuous input t (0..N-1).

    Step 0 is BOT's hidden state; steps 1..N-1 are the outputs of preceding
    continuous passes. Default layer=-1 is the final normalized layer, BEFORE
    the learned CODI projection. Other hidden_states indices are observations
    only: feedback always uses the official final layer + projection. Noise
    is applied to the final-layer vector before projection and consumption.
    Returned final-layer states include noise. No unused N+1 state is returned.
    """
    def __init__(self, config):
        self.config=dict(config);self.device=check_device(config)
        path=Path(config['checkpoint'])
        if not path.is_file(): raise FileNotFoundError(path)
        mod=import_file(Path(config['code_path'])/'src/model.py','clr_official_codi')
        args=mod.ModelArguments(model_name_or_path=config['base_path'],train=False,lora_init=True)
        train=SimpleNamespace(bf16=config['dtype']=='bfloat16',num_latent=config['num_latent'],use_lora=True,
          use_prj=True,prj_no_ln=False,prj_dropout=0.,prj_dim=config['prj_dim'],print_loss=False,
          ref_loss_factor=1.,distill_loss_div_std=True,distill_loss_type='l2',distill_loss_factor=1.,fix_attn_mask=False)
        targets=['c_attn','c_proj','c_fc'] if config['family']=='gpt2' else ['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj','gate_proj']
        self.model=mod.CODI(args,train,LoraConfig(task_type=TaskType.CAUSAL_LM,r=128,lora_alpha=32,lora_dropout=.1,target_modules=targets))
        state=torch.load(path,map_location='cpu',weights_only=True)
        # Full official state only; reject partial/adapted checkpoints.
        self.model.load_state_dict(state,strict=True)
        self.model.codi.tie_weights()
        self.model.to(device=self.device,dtype=getattr(torch,config['dtype'])).eval().requires_grad_(False)
        self.tokenizer=self.model.tokenizer
        self.forward_calls=0

    def _forward(self,**kw):
        self.forward_calls+=1
        return self.model.codi(**kw)

    @torch.inference_mode()
    def run(self,problem,perturb=None):
        c=self.config;m=self.model;start=self.forward_calls
        ids=self.tokenizer.encode(question(problem).strip().replace('  ',' '),return_tensors='pt').to(self.device)
        ids=torch.cat([ids,ids.new_tensor([[m.bot_id]])],dim=1)
        if ids.shape[1]+c['num_latent']+1+c['max_new_tokens']>m.codi.config.max_position_embeddings:
            raise ValueError('Context budget exceeded; truncation forbidden')
        noise=RelativeNoise(perturb,self.device);steps=[]
        out=self._forward(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,output_hidden_states=True)
        for t in range(c['num_latent']):
            z=noise(out.hidden_states[-1][:,-1:,:],t)
            observed=z if c['layer']==-1 else out.hidden_states[c['layer']][:,-1:,:]
            steps.append(observed[0,0].detach().cpu().clone())
            out=self._forward(inputs_embeds=m.prj(z),past_key_values=out.past_key_values,use_cache=True,output_hidden_states=True)
        cache=out.past_key_values
        emb=m.get_embd(m.codi,m.model_name)
        inp=emb(ids.new_tensor([[m.eot_id]]));tokens=[]
        for _ in range(c['max_new_tokens']):
            out=self._forward(inputs_embeds=inp,past_key_values=cache,use_cache=True)
            cache=out.past_key_values
            token=out.logits[:,-1,:m.codi.config.vocab_size-1].argmax(-1).item()
            tokens.append(token)
            if token==self.tokenizer.eos_token_id: break
            inp=emb(ids.new_tensor([[token]]))
        return {'answer':self.tokenizer.decode(tokens,skip_special_tokens=True),'latent_steps':steps,
          'meta':{'source':'codi','layer':c['layer'],'forward_calls':self.forward_calls-start,'generated_tokens':len(tokens),
                  'hit_token_limit':tokens[-1]!=self.tokenizer.eos_token_id,'perturb':perturb}}

    @torch.inference_mode()
    def run_batch(self,problems,perturbs=None):
        """Equal-token-length batching only; no padding can enter latent KV attention.

        Used for K independent noise rollouts. Explicit batch kernels may differ
        numerically from batch=1; batch size is part of the experiment protocol.
        """
        c=self.config;m=self.model;n=len(problems)
        if not n:raise ValueError('Empty batch')
        rows=[self.tokenizer.encode(question(p).strip().replace('  ',' '))+[m.bot_id] for p in problems]
        if len(set(map(len,rows)))!=1:raise ValueError('Batch requires equal token lengths; padding forbidden')
        if len(rows[0])+c['num_latent']+1+c['max_new_tokens']>m.codi.config.max_position_embeddings:raise ValueError('Context budget exceeded')
        perturbs=perturbs or [None]*n
        if len(perturbs)!=n:raise ValueError('Perturb batch size mismatch')
        noises=[RelativeNoise(p,self.device) for p in perturbs];steps=[[] for _ in rows];start=self.forward_calls
        ids=torch.tensor(rows,device=self.device)
        out=self._forward(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,output_hidden_states=True)
        for t in range(c['num_latent']):
            z=torch.cat([noise(out.hidden_states[-1][i:i+1,-1:,:],t) for i,noise in enumerate(noises)])
            obs=(z if c['layer']==-1 else out.hidden_states[c['layer']][:,-1:,:]).detach().cpu()
            for i in range(n):steps[i].append(obs[i,0].clone())
            out=self._forward(inputs_embeds=m.prj(z),past_key_values=out.past_key_values,use_cache=True,output_hidden_states=True)
        cache=out.past_key_values;emb=m.get_embd(m.codi,m.model_name)
        inp=emb(ids.new_full((n,1),m.eot_id));tokens=[[] for _ in rows];finished=[False]*n
        for _ in range(c['max_new_tokens']):
            out=self._forward(inputs_embeds=inp,past_key_values=cache,use_cache=True);cache=out.past_key_values
            tok=out.logits[:,-1,:m.codi.config.vocab_size-1].argmax(-1)
            for i,token in enumerate(tok.tolist()):
                if not finished[i]:
                    tokens[i].append(token);finished[i]=token==self.tokenizer.eos_token_id
            if all(finished):break
            inp=emb(tok[:,None])
        return [{'answer':self.tokenizer.decode(tokens[i],skip_special_tokens=True),'latent_steps':steps[i],
          'meta':{'source':'codi','layer':c['layer'],'batch_size':n,'batch_forward_calls':self.forward_calls-start,
          'generated_tokens':len(tokens[i]),'hit_token_limit':not finished[i],'perturb':perturbs[i]}} for i in range(n)]

def load(config):
    return CODISource(config)

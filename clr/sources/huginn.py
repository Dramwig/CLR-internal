"""Huginn recurrent-depth states from the pinned official HF implementation."""
import types
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import RelativeNoise,check_device,question

class HuginnSource:
    """Capture each post-core-block recurrent state at the answer prediction position.

    On prefill this is the final prompt token, which predicts answer token 0.
    On each decode forward it is the last answer token predicting the next.
    Steps are flattened in (generated-token index, recurrence index) order;
    meta.step_positions maps every entry. States are before final ln_f/coda.
    Relative noise affects only that position, after each complete recurrence;
    from_step indexes this flattened sequence, inclusively. Earlier prompt
    positions are untouched. This is NOT a decoder-layer hidden-state trace.
    """
    def __init__(self,config):
        self.config=dict(config);self.device=check_device(config)
        self.tokenizer=AutoTokenizer.from_pretrained(config['checkpoint'],local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(config['checkpoint'],local_files_only=True,
            trust_remote_code=True,torch_dtype=getattr(torch,config['dtype']))
        self.model.to(self.device).eval().requires_grad_(False)
        if not hasattr(self.model,'core_block_forward'): raise RuntimeError('Unexpected Huginn implementation')
        self.forward_calls=0

    @torch.inference_mode()
    def run(self,problem,perturb=None):
        c=self.config;noise=RelativeNoise(perturb,self.device);states=[];positions=[];tokens=[]
        # Canonical lm-eval GSM8K zero-shot prompt without a system message.
        prompt='Question: '+question(problem)+'\nAnswer:'
        ids=self.tokenizer.encode(prompt,add_special_tokens=True,return_tensors='pt').to(self.device)
        if ids.shape[1]+c['max_new_tokens']>self.model.config.block_size: raise ValueError('Context budget exceeded')
        original=self.model.core_block_forward
        def core(this,*args,**kwargs):
            x,idx=original(*args,**kwargs)
            x=x.clone();x[:,-1:]=noise(x[:,-1:],len(states))
            states.append(x[0,-1].detach().cpu().clone())
            positions.append([len(tokens),(len(states)-1)%c['num_steps']])
            return x,idx
        self.model.core_block_forward=types.MethodType(core,self.model)
        cache=None;start=self.forward_calls;prompt_len=ids.shape[1]
        devices=[self.device.index] if self.device.type=='cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(c['seed'])
                for j in range(c['max_new_tokens']):
                    pos=torch.arange(prompt_len,device=self.device) if j==0 else torch.tensor([prompt_len+j-1],device=self.device)
                    self.forward_calls+=1
                    out=self.model(input_ids=ids,num_steps=c['num_steps'],past_key_values=cache,use_cache=True,cache_position=pos)
                    cache=out.past_key_values
                    tok=out.logits[0,-1].argmax().item();tokens.append(tok)
                    text=self.tokenizer.decode(tokens,skip_special_tokens=True)
                    if tok in c['eos_token_ids'] or any(stop in text for stop in c['stop_strings']): break
                    ids=torch.tensor([[tok]],device=self.device)
        finally: self.model.core_block_forward=original
        answer=self.tokenizer.decode(tokens,skip_special_tokens=True)
        for stop in c['stop_strings']: answer=answer.split(stop)[0]
        return {'answer':answer,'latent_steps':states,'meta':{'source':'huginn','step_positions':positions,
            'forward_calls':self.forward_calls-start,'recurrent_calls':len(states),'generated_tokens':len(tokens),'perturb':perturb}}

def load(config): return HuginnSource(config)

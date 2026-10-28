"""Instrument the frozen official Coconut implementation with a removable hook."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .common import RelativeNoise,check_device,import_file,question

class CoconutSource:
    """Each latent is the preceding hidden state fed into a continuous slot.

    layer=-1 captures the normalized final layer (after optional noise).
    Other layers are observed at the same position; official feedback stays
    at the final layer. Noise changes the actual input to the next step.
    """
    def __init__(self,config):
        self.config=dict(config);self.device=check_device(config)
        path=Path(config['checkpoint'])
        if not path.is_file(): raise FileNotFoundError(path)
        mod=import_file(Path(config['code_path'])/'coconut.py','clr_official_coconut')
        self.tokenizer=AutoTokenizer.from_pretrained(config['base_path'],local_files_only=True)
        for token in ['<|start-latent|>','<|end-latent|>','<|latent|>']: self.tokenizer.add_tokens(token)
        self.start,self.end,self.latent=[self.tokenizer.convert_tokens_to_ids(t) for t in ['<|start-latent|>','<|end-latent|>','<|latent|>']]
        base=AutoModelForCausalLM.from_pretrained(config['base_path'],local_files_only=True)
        base.resize_token_embeddings(len(self.tokenizer))
        self.model=mod.Coconut(base,self.latent,self.start,self.end,self.tokenizer.eos_token_id)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.model.to(device=self.device,dtype=getattr(torch,config['dtype']));self.model.eval()
        self.forward_calls=0

    @torch.inference_mode()
    def run(self,problem,perturb=None):
        c=self.config;steps=[];calls=0;noise=RelativeNoise(perturb,self.device)
        def hook(module,args,out):
            nonlocal calls
            calls+=1;self.forward_calls+=1
            if len(steps)<c['num_latent']:
                t=len(steps);states=list(out.hidden_states);last=states[-1].clone()
                last[:,-1:]=noise(last[:,-1:],t)
                obs=last if c['layer']==-1 else states[c['layer']]
                steps.append(obs[0,-1].detach().cpu().clone());states[-1]=last
                out.hidden_states=tuple(states)
            return out
        ids=self.tokenizer.encode(question(problem)+'\n')+[self.start]+[self.latent]*c['num_latent']+[self.end]
        if len(ids)+c['max_new_tokens']>self.model.base_causallm.config.max_position_embeddings: raise ValueError('Context budget exceeded')
        ids=torch.tensor([ids],device=self.device)
        handle=self.model.base_causallm.register_forward_hook(hook)
        try: out=self.model.generate(ids,torch.ones_like(ids),max_new_tokens=c['max_new_tokens'])
        finally: handle.remove()
        if len(steps)!=c['num_latent']: raise RuntimeError('Unexpected latent count')
        tokens=out[0,ids.shape[1]:].tolist()
        return {'answer':self.tokenizer.decode(tokens,skip_special_tokens=True),'latent_steps':steps,
                'meta':{'source':'coconut','layer':c['layer'],'forward_calls':calls,'generated_tokens':len(tokens),'perturb':perturb}}

def load(config): return CoconutSource(config)

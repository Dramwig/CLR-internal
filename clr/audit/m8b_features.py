"""Randomized source maps and differentiable receiver-state observation."""
import copy
import torch
from clr.audit.features import FeatureSequence,KVFeatures

KEYS=['values','mask','layers','segments','kinds']

def as_source(s):return {k:getattr(s,k)[None] for k in KEYS}

def cpu_sequence(s):return {k:getattr(s,k).detach().cpu().half() if k=='values' else getattr(s,k).detach().cpu() for k in s.__dataclass_fields__}

class ReceiverStates:
    def __init__(self,relay,projection,layers=(7,16,25,34),tail=8):
        self.relay,self.projection,self.layers,self.tail=relay,projection,tuple(layers),tail
    def __call__(self,prefix,prepared,cost):
        from latdef.inject.cache import prefix_forward
        ids,mask,letters=prepared;states={};hooks=[]
        for layer in self.layers:
            def capture(module,args,out,layer=layer):states[layer]=out[0] if isinstance(out,tuple) else out
            hooks.append(self.relay.s2.model.model.layers[layer].register_forward_hook(capture))
        try:out=prefix_forward(self.relay.s2.model,ids,mask,prefix=prefix,logits_to_keep=1)
        finally:
            for hook in hooks:hook.remove()
        cost.forward_calls+=1
        positions=mask[0].nonzero().flatten()[-self.tail:];values=[];ll=[];ss=[]
        for layer in self.layers:
            values.append(states[layer][0,positions].float()@self.projection);ll.extend([layer]*len(positions));ss.extend(range(len(positions)))
        v=torch.cat(values);device=v.device
        seq=FeatureSequence(v,torch.ones(len(v),device=device,dtype=torch.bool),torch.tensor(ll,device=device),torch.tensor(ss,device=device),positions.repeat(len(self.layers))[:,None].expand(-1,2),torch.ones(len(v),device=device,dtype=torch.long))
        logp=out.logits[:,-1].float().gather(1,letters).log_softmax(-1)[0]
        return logp,seq

class RandomKV:
    def __init__(self,projection):
        self.projection=projection
        self.full=KVFeatures(projection,range(36))
    def __call__(self,prefix,meta,seed,design):
        # Use the same fp16 sufficient statistics as training/calibration.
        # Casting retains gradients; the transmitted KV remains real BF16.
        if design=='R3':
            seq=self.full(prefix,prefix.mask,meta);seq.values=seq.values.half();return seq
        cache=cache_kv(prefix,meta,self.projection,detach=False)
        return cached_sequence(cache,seed,design)

def cache_kv(prefix,meta,projection,detach=True,layers=None):
    """Sufficient statistics for exact linear pooling with +-1 boundary jitter."""
    layers=list(range(36)) if layers is None else list(layers)
    base=KVFeatures(projection,layers)(prefix,prefix.mask,meta)
    offset=meta.get('physical_offset',0);edges=[b+offset for a,b in meta['record_spans'][:-1]]
    pairs=[]
    for layer in layers:
        k,v=prefix.layers[layer]
        vec=KVFeatures._vectors(k,v,0).float()
        pairs.append((vec[torch.tensor([[b-1,b] for b in edges],device=vec.device)]@projection).half())
    saved=cpu_sequence(base) if detach else {k:(getattr(base,k).half() if k=='values' else getattr(base,k)) for k in base.__dataclass_fields__}
    return {'layer_ids':layers,'base':saved,'boundary':torch.stack(pairs).cpu() if detach else torch.stack(pairs),'lengths':torch.tensor([b-a for a,b in meta['record_spans']]),'record_count':len(meta['record_spans'])}

def cached_sequence(cache,seed,design):
    s=FeatureSequence(**{k:v.clone() for k,v in cache['base'].items()})
    if design=='R3':return s
    g=torch.Generator().manual_seed(int(seed));layers=sorted(torch.randperm(36,generator=g)[:8].tolist())
    nr=cache['record_count'];width=nr+8
    available=cache.get('layer_ids',list(range(36)));depth=len(available)
    vals=s.values.reshape(depth,width,-1).float();bounds=s.token_ranges.reshape(depth,width,2)
    lens=cache['lengths'].clone().to(vals.device);j=torch.randint(-1,2,(nr-1,),generator=g)
    sums=vals[:,:nr]*lens.clone()[None,:,None]
    for i,delta in enumerate(j.tolist()):
        if delta:
            correction=cache['boundary'][:,i,1 if delta==1 else 0].float()*delta
            sums[:,i]+=correction;sums[:,i+1]-=correction;lens[i]+=delta;lens[i+1]-=delta
            bounds[:,i,1]+=delta;bounds[:,i+1,0]+=delta
    vals[:,:nr]=sums/lens[None,:,None]
    keep=((torch.rand(vals.shape[-1],generator=g)<.9)/.9).to(vals.device);vals*=keep
    select=torch.tensor([available.index(l) for l in layers],device=vals.device)[:,None]*width+torch.arange(width,device=vals.device)[None];select=select.flatten()
    for key in s.__dataclass_fields__:
        setattr(s,key,(vals.reshape(-1,vals.shape[-1]).half() if key=='values' else getattr(s,key))[select])
    return s


def cache_on_device(cache,device):
    return {**cache,'base':{k:v.to(device) for k,v in cache['base'].items()},'boundary':cache['boundary'].to(device)}

"""M5b observed-state sibling panels and searches with a clean child per parent."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from clr.policy import RelativeNoise
from clr.search.m5a import Engine, State, Values, seed_for, join
from clr.models.clr_v1 import make_model
from clr.models.clr_v0 import Serializer

@dataclass
class Branch:
    state: State
    history: torch.Tensor
    def select(self,indices):
        ii=list(indices)
        return Branch(self.state.select(ii),self.history.index_select(0,torch.tensor(ii,device=self.history.device,dtype=torch.long)))


def root_branch(prefix,t=0):
    return Branch(State(prefix.states[t],prefix.caches[t],t),torch.cat(prefix.states[:t+1],dim=1))


@torch.inference_mode()
def transition(engine,branch,seeds,clean_indices=()):
    if len(seeds)!=len(branch.state.z):raise ValueError('Lane count mismatch')
    clean=set(clean_indices);groups=[([i for i in range(len(seeds)) if i in clean],0.),([i for i in range(len(seeds)) if i not in clean],engine.sigma)]
    parts=[];order=[]
    for ii,sigma in groups:
        if ii:parts.append(engine.transition(branch.state.select(ii),[seeds[i] for i in ii],sigma=sigma));order+=ii
    state=join(parts).select(np.argsort(order).tolist())
    return Branch(state,torch.cat([branch.history,state.z],1))


@torch.inference_mode()
def children(engine,parent,N,seeds):
    """N total children per parent; index i*N is always its sigma=0 child."""
    if len(seeds)!=N*len(parent.state.z):raise ValueError('Wrong candidate count')
    expanded=parent.select([i for i in range(len(parent.state.z)) for _ in range(N)])
    return transition(engine,expanded,seeds,range(0,len(seeds),N))


@torch.inference_mode()
def continue_branch(engine,branch,seeds,stop=6,clean_indices=()):
    """Production Gaussian policy, with one persistent RNG per trajectory."""
    if len(seeds)!=len(branch.state.z):raise ValueError('Lane count mismatch')
    clean=set(clean_indices)
    noises=[RelativeNoise({'from_step':branch.state.t,'sigma':0. if i in clean else engine.sigma,'seed':s},engine.source.device) for i,s in enumerate(seeds)]
    while branch.state.t<stop:
        parts=[]
        for start in range(0,len(seeds),engine.batch):
            ii=list(range(start,min(start+engine.batch,len(seeds))));state=branch.state.select(ii)
            feedback=torch.cat([noises[i](state.z[j:j+1],state.t) for j,i in enumerate(ii)])
            out=engine.source._forward(inputs_embeds=engine.source.model.prj(feedback),past_key_values=state.cache,use_cache=True,output_hidden_states=True)
            parts.append(State(out.hidden_states[-1][:,-1:,:],out.past_key_values,state.t+1))
        state=join(parts);branch=Branch(state,torch.cat([branch.history,state.z],1))
    return branch


class ValueRunner:
    def __init__(self,name,seed,c,templates,device):
        self.name=name;self.device=device;self.temperature=1.;self.model=None
        if name in ['B3','B4']:
            self.values=Values(c['m3_baseline_run'],seed,templates,device)
        else:
            variant='v1' if name=='CLR-v1' else 'lite'
            ck=(Path(c['d_runs'][str(seed)])/'best.pt' if name=='D-lite' else Path(c['m41_runs'][variant][str(seed)])/'best.pt')
            self.model=make_model({**c,'variant':variant,'model_path':c['m41_model_path'],'gradient_checkpointing':False}).to(device).eval()
            self.model.restore_trainable(torch.load(ck,map_location='cpu',weights_only=True))
            self.serializer=Serializer(AutoTokenizer.from_pretrained(c['m41_model_path'],local_files_only=True),templates)
            self.row={'qtype':'Q1','template_id':0,'options':['0']}
        self.flops_by_length={}
        self.max_batch=c.get('value_batch',128)

    @torch.inference_mode()
    def __call__(self,branch):
        if self.name in ['B3','B4']:return self.values(self.name,branch.state).float().cpu().numpy()
        out=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(branch.history),self.max_batch):
                z=branch.history[start:start+self.max_batch];text=self.serializer.batch([self.row]*len(z),self.device)
                logits=self.model(text,z,torch.ones(z.shape[:2],dtype=torch.bool,device=self.device),'Q1')
                out.append((logits[:,1]-logits[:,0]).float().cpu().numpy())
        return np.concatenate(out)

    def flop_cost(self,length):
        """Measured PyTorch operator FLOPs for this value model at actual prefix length.

        Counters include linear matmul/attention FLOPs; embedding, normalization,
        activation and serializer CPU operations are recorded as excluded.
        """
        if self.name=='B3':return float(2*(840*128+128)+2*768)
        if self.name=='B4':return float(4*768)
        if length not in self.flops_by_length:
            from torch.utils.flop_counter import FlopCounterMode
            fastpath=torch.backends.mha.get_fastpath_enabled()
            try:
                # The fused inference-only Transformer op has no registered FLOP
                # formula. Expand that same layer into its matmul/SDPA operators
                # for profiling; production inference keeps its original setting.
                torch.backends.mha.set_fastpath_enabled(False)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    z=torch.zeros(1,length,768,device=self.device);text=self.serializer.batch([self.row],self.device)
                    with FlopCounterMode(display=False) as counter:self.model(text,z,torch.ones(1,length,dtype=torch.bool,device=self.device),'Q1')
                    count=float(counter.get_total_flops())
                    if count<=0:raise RuntimeError('Missing value-model FLOP profile')
                    self.flops_by_length[length]=count
            finally:torch.backends.mha.set_fastpath_enabled(fastpath)
        return self.flops_by_length[length]


@torch.inference_mode()
def oracle_scores(engine,branch,problem,seed,K,key,gold):
    before=engine.lanes;scores=[];hits=[]
    # Bound full KV copies to engine.batch lanes instead of expanding N*K caches.
    for start in range(0,len(branch.state.z),max(1,engine.batch//K)):
        ii=list(range(start,min(start+max(1,engine.batch//K),len(branch.state.z))))
        seeds=[seed_for(seed,problem['id'],'m5b-oracle',key,branch.state.t,i,k) for i in ii for k in range(K)]
        final=engine.continue_pi(branch.state.select([i for i in ii for _ in range(K)]),seeds)
        answers,_,_=engine.decode(final)
        hh=np.array([a==gold for a in answers],dtype=np.uint8).reshape(len(ii),K)
        scores.extend(hh.mean(1));hits.extend(hh.tolist())
    return np.asarray(scores),engine.lanes-before,hits


def group_metrics(scores,truth,delta=.25):
    scores=np.asarray(scores);truth=np.asarray(truth)
    rho=float(spearmanr(scores,truth).statistic) if np.ptp(scores)>0 and np.ptp(truth)>0 else None
    correct=[]
    for i in range(len(scores)):
        for j in range(i+1,len(scores)):
            if abs(truth[i]-truth[j])>=delta:
                product=(scores[i]-scores[j])*(truth[i]-truth[j]);correct.append(float(product>0)+.5*float(product==0))
    return {'spearman':rho,'pair_accuracy':float(np.mean(correct)) if correct else None,'pairs':len(correct),'pair_correct':float(sum(correct)),
            'gain_vs_sigma0':float(truth[int(np.argmax(scores))]-truth[0]),'chosen':int(np.argmax(scores))}

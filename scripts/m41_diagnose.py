"""Val-only checkpoint causal/gradient diagnostics; never loads test predictions."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m41_budget import Budget
from clr.models.clr_v0 import CLR,Serializer
from clr.data.m4.reader import PrefixBank,rows_for,targets

def module_stats(model):
    out={}
    for name,module in model.named_children():
        if name=='backbone':continue
        ps=list(module.parameters());gs=[p.grad for p in ps if p.grad is not None]
        out[name]={'param_norm':sum(float(p.detach().float().square().sum()) for p in ps)**.5,'grad_norm':sum(float(g.float().square().sum()) for g in gs)**.5,'parameters':sum(p.numel() for p in ps)}
    if hasattr(model,'source_token'):out['source_token']={'param_norm':float(model.source_token.norm()),'grad_norm':float(model.source_token.grad.norm()) if model.source_token.grad is not None else 0.}
    for name,p in model.named_parameters():
        if name.endswith("logit_scale"):out[name]={"gate_value":float(p.detach().sigmoid()),"gradient_norm":float(p.grad.norm()) if p.grad is not None else 0.}
    return out

def main(c):
 with recorded(c) as run:
    start=time.monotonic();budget=Budget(run,c,start);torch.set_num_threads(2);seed_everything(c['seed']);device=c['reader_device']
    try:
        templates=load_config(c['templates']);s=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates)
        allrows=rows_for(c,'reader_val','validate');rng=np.random.default_rng(c['seed']);groups={}
        for q in ['Q1','Q2','Q3','Q4']:
            rs=[r for r in allrows if r['qtype']==q];ii=rng.choice(len(rs),c['diagnostic_batches']*c['diagnostic_batch'],replace=False);groups[q]=[rs[i] for i in ii]
        bank=PrefixBank(c,'reader_val','validate',{r['problem_id'] for rs in groups.values() for r in rs});del allrows
        write_json(run.path/'panel.json',groups)
        norm=np.load(Path(c['m3_baseline_run'])/'normalization.npz');suite=json.loads(Path('data/M4_v0/training_suite.json').read_text())
        cases={'fresh':None,**{f'main_{seed}':suite[f'main_{seed}'] for seed in [11,22,33]},'transfer_10000_11':suite['transfer_10000_11']}
        results=[]
        for case,path in cases.items():
            seed_everything(c['seed']);m=CLR.pretrained(c['model_path'],k=4,gradient_checkpointing=False).to(device)
            m.state_mean.copy_(torch.from_numpy(norm['mean']).to(device));m.state_scale.copy_(torch.from_numpy(norm['scale']).to(device))
            if path:m.restore_trainable(torch.load(Path(path)/'best.pt',map_location='cpu',weights_only=True))
            m.eval()
            for q,rs in groups.items():
                for j in range(c['diagnostic_batches']):
                    budget.check();rows=rs[j*c['diagnostic_batch']:(j+1)*c['diagnostic_batch']];z,mask=bank.batch(rows,device);z=z.float().detach().requires_grad_(True);text=s.batch(rows,device)
                    # FP32 diagnostic resolves small derivatives, not bf16 rounded changes.
                    m.zero_grad(set_to_none=True);logits=m(text,z,mask,q)
                    decision=logits-logits.mean(-1,keepdim=True)
                    grad=torch.autograd.grad(decision[:,0].sum(),z,retain_graph=True)[0]
                    rawgrad=torch.autograd.grad(logits[:,-1].sum(),z,retain_graph=True)[0]
                    option_grads=[float(torch.autograd.grad(logits[:,i].sum(),z,retain_graph=True)[0].norm()) for i in range(logits.shape[1])];loss=-(targets(rows,device)*logits.log_softmax(-1)).sum(-1).mean();loss.backward()
                    with torch.no_grad():
                        zero=m(text,torch.zeros_like(z),mask,q);shuffle=m(text,z.roll(1,0),mask.roll(1,0),q)
                    results.append({'case':case,'qtype':q,'batch':j,'loss':float(loss),'z_gradient_norm':float(rawgrad.norm()),'contrast_gradient_norm':float(grad.norm()),'zero_logit_delta':float((zero-logits).abs().max()),'shuffle_logit_delta':float((shuffle-logits).abs().max()),'zero_contrast_delta':float(((zero-zero.mean(-1,keepdim=True))-decision).abs().max()),'shuffle_contrast_delta':float(((shuffle-shuffle.mean(-1,keepdim=True))-decision).abs().max()),'modules':module_stats(m),'parameters':{name:{'norm':float(p.detach().norm()),'gradient_norm':float(p.grad.norm()) if p.grad is not None else 0.} for name,p in m.named_parameters() if not name.startswith('backbone.')},'option_gradient_norms':option_grads})
                    write_json(run.path/'path_diagnostics.json',results)
            del m;gc.collect();torch.cuda.empty_cache()
        # Controlled diagnostic replay, not historical M4 gradient records.
        m=CLR.pretrained(c['model_path'],k=4,gradient_checkpointing=False).to(device);m.state_mean.copy_(torch.from_numpy(norm['mean']).to(device));m.state_scale.copy_(torch.from_numpy(norm['scale']).to(device));m.train()
        opt=torch.optim.AdamW([{'params':[p for n,p in m.named_parameters() if p.requires_grad and n.startswith('backbone.')],'lr':c['encoder_lr']},{'params':[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('backbone.')],'lr':c['head_lr']}],weight_decay=.01)
        traces=[]
        for step in range(c['diagnostic_steps']):
            budget.check();q=['Q1','Q2','Q3','Q4'][step%4];rs=groups[q][:c['diagnostic_batch']];z,mask=bank.batch(rs,device);opt.zero_grad(set_to_none=True)
            logits=m(s.batch(rs,device),z.float(),mask,q);loss=-(targets(rs,device)*logits.log_softmax(-1)).sum(-1).mean();loss.backward()
            traces.append({'step':step,'qtype':q,'loss':float(loss),'modules':module_stats(m),'gate':'none in CLR v0'});torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();write_json(run.path/'module_trace.json',traces)
        run.write_metrics(status='completed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=False,diagnostic_batches=len(results),gate='none')
        write_json(Path(c['output'])/'path_diagnostics_run.json',{'run':str(run.path)})
    except Exception:
        run.write_metrics(status='failed',gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

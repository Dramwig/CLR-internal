"""Parallel conditional linear probes for preregistered train-only layer scans."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def fit_probes(features,truth,negative,fit_mask,config,run,cost,label):
    """features[N,layer,dim], A4 linear logits and fact-conditioned A1 logits.

    The condition is the fact explicitly asked in A1. It is not an oracle input:
    the two A1 question instances select separate logits of the same probe.
    Each group and its CF are confined to the same scan partition.
    """
    device=config['device'];torch.manual_seed(config['seed'])
    x=features.float();train=torch.as_tensor(fit_mask,dtype=torch.bool)
    mean=x[train].mean(0);scale=x[train].std(0).clamp_min(1e-4)
    n,layers,dim=x.shape
    weight=torch.nn.Parameter(torch.zeros(layers,dim,8,device=device))
    bias=torch.nn.Parameter(torch.zeros(layers,8,device=device))
    optimizer=torch.optim.AdamW([weight,bias],lr=config['scan_lr'],weight_decay=.01)
    indices=torch.where(train)[0].numpy();rng=np.random.default_rng(config['seed'])
    target=torch.as_tensor(truth,dtype=torch.long);other=torch.as_tensor(negative,dtype=torch.long)
    ids=torch.arange(layers,device=device)[None]
    for step in range(config['scan_steps']):
        cost.check();ii=rng.choice(indices,size=min(config['scan_batch_size'],len(indices)),replace=False)
        z=((x[ii]-mean)/scale).to(device)
        logits=torch.einsum('bld,ldc->blc',z,weight)+bias
        a4=torch.nn.functional.cross_entropy(logits[:,:,:4].reshape(-1,4),
                         target[ii].to(device)[:,None].expand(-1,layers).flatten())
        row=torch.arange(len(ii),device=device)[:,None]
        pos=logits[row,ids,target[ii].to(device)[:,None]+4]
        neg=logits[row,ids,other[ii].to(device)[:,None]+4]
        a1=(torch.nn.functional.softplus(-pos).mean()+torch.nn.functional.softplus(neg).mean())/2
        loss=a4+a1
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%50==0:
            cost.update();print(label,step,config['scan_steps'],float(loss.detach()),flush=True)
    valid=torch.where(~train)[0];pred=[]
    with torch.no_grad():
        for start in range(0,len(valid),config['scan_batch_size']):
            ii=valid[start:start+config['scan_batch_size']]
            z=((x[ii]-mean)/scale).to(device)
            pred.append((torch.einsum('bld,ldc->blc',z,weight)+bias).cpu())
    logits=torch.cat(pred);yy=target[valid];nn=other[valid];curves=[]
    for layer in range(layers):
        a4=logits[:,layer,:4].argmax(-1)==yy
        positive=logits[torch.arange(len(valid)),layer,yy+4]
        negative_score=logits[torch.arange(len(valid)),layer,nn+4]
        y=np.r_[np.ones(len(valid)),np.zeros(len(valid))]
        score=torch.cat([positive,negative_score]).numpy()
        curves.append({'layer':layer,'A4_accuracy':float(a4.float().mean()),
                       'A1_accuracy':float(np.mean((score>0)==y)),
                       'A1_AUROC':float(roc_auc_score(y,score)),'validation_messages':len(valid)})
    torch.save({'weight':weight.detach().cpu(),'bias':bias.detach().cpu(),'mean':mean,'scale':scale,
                'valid_indices':valid,'logits':logits,'truth':yy,'negative':nn},run.path/(label+'_probe.pt'))
    return curves

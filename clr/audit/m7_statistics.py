"""Group bootstrap for nonlinear QA metrics, conditional on fixed trained seeds."""
import numpy as np


def nonlinear_qa_intervals(frame,weights):
    groups=sorted(frame.group_id.unique());seeds=sorted(frame.seed.unique())
    ece_draws=np.zeros(len(weights));ece_mean=0.;auc_matrix=None
    for seed in seeds:
        f=frame[frame.seed==seed]
        p=np.stack(f.probs.to_list());y=f.label.to_numpy(int)
        confidence=p.max(1);correct=p.argmax(1)==y
        bins=np.minimum((confidence*15).astype(int),14)
        gi=np.searchsorted(groups,f.group_id.to_numpy())
        sums=np.zeros((len(groups),15))
        np.add.at(sums,(gi,bins),correct.astype(float)-confidence)
        ece_mean+=np.abs(sums.sum(0)).sum()/len(f)/len(seeds)
        ece_draws+=np.abs(weights@sums).sum(1)/len(f)/len(seeds)
        if p.shape[1]==2:
            pos=[p[(gi==i)&(y==1),1] for i in range(len(groups))]
            neg=[p[(gi==i)&(y==0),1] for i in range(len(groups))]
            if min(map(len,pos))==0 or min(map(len,neg))==0:raise ValueError('Combined QA needs both binary labels per group')
            if len(set(map(len,pos)))!=1 or len(set(map(len,neg)))!=1:raise ValueError('QA group question weights differ')
            pos=np.stack(pos);neg=np.stack(neg)
            difference=pos[:,:,None,None]-neg[None,None,:,:]
            matrix=((difference>0)+.5*(difference==0)).mean(axis=(1,3))
            if auc_matrix is None:auc_matrix=np.zeros_like(matrix)
            auc_matrix+=matrix/len(seeds)
    def summary(mean,draws):
        return {'mean':float(mean),'lo':float(np.quantile(draws,.025)),'hi':float(np.quantile(draws,.975))}
    result={'ece':summary(ece_mean,ece_draws)}
    if auc_matrix is not None:
        draws=np.einsum('bi,ij,bj->b',weights,auc_matrix,weights,optimize=True)/len(groups)**2
        result['auroc']=summary(auc_matrix.mean(),draws)
    return result

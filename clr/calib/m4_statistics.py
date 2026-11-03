"""Exact weighted AUROC under problem-cluster bootstrap, including tied scores."""
import numpy as np

class WeightedAUC:
    def __init__(self,labels,scores):
        self.order=np.argsort(scores,kind='stable');ordered=np.asarray(scores)[self.order]
        self.labels=np.asarray(labels)[self.order];self.starts=np.r_[0,1+np.flatnonzero(np.diff(ordered))]
    def __call__(self,weights):
        w=np.asarray(weights)[:,self.order];positive=np.add.reduceat(w*self.labels,self.starts,axis=1);negative=np.add.reduceat(w*(1-self.labels),self.starts,axis=1)
        denominator=positive.sum(1)*negative.sum(1)
        if np.any(denominator==0):raise ValueError('Bootstrap replicate has only one outcome class')
        return (positive*(np.cumsum(negative,axis=1)-.5*negative)).sum(1)/denominator

"""Source-shift diagnostics; weighted finite-sample quantile includes test mass.

Estimated density ratios and distribution/label shift do not inherit an
unconditional coverage guarantee. Classifier fitting is restricted to cal_A;
cal_B labels alone set thresholds. Evaluation labels never enter either fit.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from clr.calib.m23 import conformal_threshold

def domain_features(logits,steps):
    p=softmax(np.asarray(logits),axis=1)
    return np.column_stack([np.sort(p,axis=1),np.asarray(steps)/6,-(p*np.log(p.clip(1e-12))).sum(1)])

def fit_ratio(source_cal,target_cal):
    x=np.concatenate([source_cal,target_cal]);y=np.r_[np.zeros(len(source_cal)),np.ones(len(target_cal))]
    model=LogisticRegression(C=1.,max_iter=1000,random_state=11).fit(x,y)
    prior=len(source_cal)/len(target_cal)
    def ratio(features):
        prob=model.predict_proba(features)[:,1].clip(1e-6,1-1e-6)
        return (prob/(1-prob)*prior).clip(.1,10.)
    return ratio,model

def weighted_threshold(scores,cal_weights,test_weights,alpha):
    scores=np.asarray(scores); weights=np.asarray(cal_weights); tw=np.atleast_1d(test_weights)
    if not len(scores) or len(weights)!=len(scores) or np.any(weights<=0) or np.any(tw<=0):raise ValueError('Positive density ratios and nonempty calibration required')
    order=np.argsort(scores,kind='stable');cumulative=np.cumsum(weights[order]);target=(1-alpha)*(weights.sum()+tw)
    indices=np.searchsorted(cumulative,target,side='left')
    result=np.full(len(tw),np.inf);valid=indices<len(scores);result[valid]=scores[order[indices[valid]]]
    return result

def coverage(probabilities,targets,threshold):
    sets=1-np.asarray(probabilities)<=np.asarray(threshold)[...,None]
    return (sets*targets).sum(1),sets.sum(1)

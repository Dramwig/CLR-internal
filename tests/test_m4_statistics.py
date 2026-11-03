import numpy as np
from sklearn.metrics import roc_auc_score
from clr.calib.m4_statistics import WeightedAUC

def test_weighted_auc_matches_sklearn_with_ties():
    y=np.array([0,1,0,1,1,0]);p=np.array([.2,.2,.8,.8,.9,.1]);w=np.array([[1,1,1,1,1,1],[2,0,1,3,1,2],[0,3,2,1,1,0]])
    expected=[roc_auc_score(y,p,sample_weight=x) for x in w]
    np.testing.assert_allclose(WeightedAUC(y,p)(w),expected,rtol=1e-14)

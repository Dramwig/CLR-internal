import numpy as np
from clr.calib.m4 import fit,apply
from clr.calib.m4_evaluation import thresholds,summarize

def test_mondrian_finite_sample_and_soft_outcomes():
    rows=[{'problem_id':str(i),'state_ref':{'t':0,'kind':'clean'},'options':[],'template_id':0,'label':i%2,'label_soft':.5,'bernoulli_outcomes':[0,1],'is_heldout_template':False} for i in range(20)]
    logits=np.zeros((20,2));cal={'temperature':1.,'bias':0.};th=thresholds(rows,logits,cal,'Q1',11)
    assert th['0:0.05']==.5
    stats,cp=summarize(rows,logits,'Q1',{'temperature':cal},{'temperature':th},11)
    assert all(x['coverage']==1 and x['size']==2 for x in cp)
    assert all(abs(x['brier']-.25)<1e-8 for x in stats)

def test_affine_float32_has_nonzero_gradient():
    x=np.zeros((100,2),np.float32);y=np.tile([.2,.8],(100,1)).astype(np.float32)
    cal=fit(x,y,'affine')
    assert abs(cal['bias']-np.log(4))<1e-3


def test_small_stratum_infinite_threshold_is_json_safe():
    import json
    rows=[{'problem_id':str(i),'state_ref':{'t':0,'kind':'clean'},'options':[],'template_id':0,'label':0,'label_soft':0.,'bernoulli_outcomes':[0],'is_heldout_template':False} for i in range(3)]
    x=np.zeros((3,2));cal={'temperature':1.,'bias':0.};th=thresholds(rows,x,cal,'Q1',11)
    _,cp=summarize(rows,x,'Q1',{'temperature':cal},{'temperature':th},11)
    assert all(r['threshold'] is None and r['threshold_is_infinite'] for r in cp)
    json.dumps(cp,allow_nan=False)

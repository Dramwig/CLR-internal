import hashlib
import json
import numpy as np
from scipy.special import softmax
from clr.export.qual import rank_ids, key


def test_selection_is_literal_hash_and_outcome_independent():
    ids=['a','b','c','d']
    wanted=sorted(ids,key=lambda p:hashlib.sha256((p+'qual').encode()).hexdigest())[:3]
    assert rank_ids(reversed(ids),3)==wanted
    assert rank_ids(ids+ids,3)==wanted
    # Outcomes are never an argument to the sampling function.
    changed_outcomes={p:False for p in ids}
    assert rank_ids(changed_outcomes,3)==wanted


def test_prediction_identity_preserves_choice_order():
    r=dict(problem_id='p',state_ref={'t':2},template_id=0,options=['1','2','3','4'])
    assert key(r)!=key({**r,'options':['2','1','3','4']})
    p=softmax(np.array([.1,2.,-.3,.5],np.float64))
    threshold=.7;perm=[2,0,3,1]
    np.testing.assert_array_equal((1-p<=threshold)[perm],1-p[perm]<=threshold)
    # Labels and conformal indices are in the original option order.
    assert p.argmax()==1


def test_json_probability_roundtrip_is_bitwise():
    p=softmax(np.array([-3.125,.2,2.1,-9.],np.float64))
    q=np.array(json.loads(json.dumps(p.tolist())),np.float64)
    np.testing.assert_array_equal(p.view(np.uint64),q.view(np.uint64))

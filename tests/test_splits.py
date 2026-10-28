import pytest
from clr.data.splits import freeze,problem_id

def test_problem_identity_and_frozen_splits(tmp_path):
    assert problem_id('a  b')==problem_id('a b')
    p=tmp_path/'split.json';s={'train':['p1'],'val':['p2'],'cal':['p3'],'test':['p4']}
    assert freeze(p,s)==freeze(p,s)
    with pytest.raises(ValueError): freeze(p,{'train':['p1'],'val':['p1']})
    with pytest.raises(ValueError): freeze(p,{'train':['p1','p1']})
    with pytest.raises(ValueError): freeze(p,{'train':['p5']})

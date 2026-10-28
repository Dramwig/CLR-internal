import pytest
from clr.data.dedup_v2 import normalize,audit
from clr.utils.stats_v2 import bootstrap_mean

def test_normalization_retains_numbers_and_signs():
    assert normalize('JOHN’S  1.5 apples!')==normalize("John's 1.5 apples")
    assert normalize('1.5')!=normalize('15')
    assert normalize('-5')!=normalize('5')

def test_thirteen_gram_boundary_and_exact():
    shared='one two three four five six seven eight nine ten eleven twelve thirteen'
    train=[{'question':shared+' end'},{'question':'Short question?'}]
    target=[{'id':'a','question':shared+' changed'},{'id':'b','question':'short question!'},{'id':'c','question':'two three four five six seven eight nine ten eleven twelve thirteen changed'}]
    r=audit(train,target)
    assert r['a']['ngram_training_row_count']==1 and not r['a']['exact_training_rows']
    assert r['b']['exact_training_rows']==[1]
    assert not r['c']['overlap']

def test_bootstrap_is_by_question_and_reproducible():
    a=bootstrap_mean([1,0,1,0]);b=bootstrap_mean([1,0,1,0])
    assert a==b and a['n_questions']==4 and a['bootstrap_resamples']==10000
    assert a['mean']==.5 and a['ci95'][0]<=.5<=a['ci95'][1]
    with pytest.raises(ValueError):bootstrap_mean([])

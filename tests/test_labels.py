import pytest
from clr.data.labels import score_answer,canonical_answer

def test_gsm_official_last_number_and_invalid():
    assert score_answer('2 plus 3 gives 5.','#### 5','gsm8k')
    assert score_answer('1,000','1000.0','gsm8k')
    assert not score_answer('unknown','5','gsm8k')
    assert canonical_answer('no answer','gsm8k') is None
    with pytest.raises(ValueError):score_answer('bad','bad','gsm8k')

def test_prosqa_exact_answer():
    assert score_answer('### Tom is a lempus.','Tom is a lempus.','prosqa')
    assert not score_answer('Tom is a lempus','Tom is a lempus.','prosqa')

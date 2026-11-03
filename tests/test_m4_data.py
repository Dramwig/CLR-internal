import pytest
from clr.data.m4.equations import equation_values

def test_numeric_equations_and_intermediates():
    assert set(equation_values('(2+3)*4=20','20'))=={'5','20'}
    assert equation_values('(76.0-25.0)','51')==['51']
    assert equation_values('7+2=9','9')==['9']
    assert equation_values('3/2=1.5','1.5')==['1.5']
    for formula in ['x+2=3','__import__("os")','2+3=6','1/0','1/3']:
        with pytest.raises((ValueError,ZeroDivisionError)):equation_values(formula)
    with pytest.raises(ValueError):equation_values('2+3=5','6')

def test_phase_guard_rejects_test_and_unseen_eval_in_debug():
    from clr.data.m4.reader import require_split
    for phase in ['train','validate','calibrate']:
        for split in ['test','unseen_eval']:
            with pytest.raises(ValueError):require_split(split,phase)
    require_split('reader_val','validate');require_split('test','final')

"""Run in .venv/huginn-eval; synthetic records test accounting, not accuracy."""
import torch
import pytest
pytest.importorskip('lm_eval')
from scripts.m0b_huginn_eval import summarize_samples,question_ci


def records():
    return [{'doc':{'id':p},'filter':f,'exact_match':v}
            for f in ['strict-match','flexible-extract']
            for p,v in [('p1',0),('p2',1)]]


def test_harness_filters_do_not_double_problem_count():
    result=summarize_samples(records(),['p1','p2'])
    assert set(result)=={'exact_match,strict-match','exact_match,flexible-extract'}
    assert all(m['n']==2 and m['mean']==.5 for m in result.values())
    assert all(m['bootstrap_replicates']==10000 for m in result.values())


def test_harness_duplicate_or_missing_problem_is_error():
    bad=records();bad[0]['doc']['id']='p2'
    with pytest.raises(ValueError,match='problem IDs'):
        summarize_samples(bad,['p1','p2'])
    with pytest.raises(ValueError,match='problem IDs'):
        summarize_samples(records()[:-1],['p1','p2'])


def test_question_bootstrap_extremes_and_reproducibility():
    assert question_ci([0]*3)['ci95']==[0.,0.]
    assert question_ci([1]*3)['ci95']==[1.,1.]
    assert question_ci([0,1,1,1])==question_ci([0,1,1,1])
    with pytest.raises(ValueError):question_ci([])


def test_utf8_startup_survives_c_locale_for_error_records(tmp_path):
    import subprocess,sys
    target=tmp_path/'error.txt'
    code="import locale,pathlib,sys;locale.setlocale(locale.LC_ALL,'C');pathlib.Path(sys.argv[1]).write_text('— 未回退模型')"
    subprocess.run([sys.executable,'-X','utf8','-c',code,str(target)],check=True)
    assert target.read_text(encoding='utf-8')=='— 未回退模型'

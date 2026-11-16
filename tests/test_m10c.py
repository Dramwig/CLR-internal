import pandas as pd
import pytest
from clr.paper.m10c_cost import paired_conformal, measured_median


def frame(model, value):
    return pd.DataFrame([dict(source='gpt2', seed=11, split='reader_val', qtype='Q3',
                              problem_id='one', t=2, instances=1, model=model, coverage=value, size=2)])


def test_paired_direction_and_missing_groups():
    a = frame('CLR-lite', 1); b = frame('B5c', 0)
    assert paired_conformal(a, b).coverage_difference.iloc[0] == -1
    b['problem_id'] = 'other'
    with pytest.raises(ValueError, match='Unpaired'): paired_conformal(a, b)


def test_duplicate_group_rejected():
    a = frame('CLR-lite', 1)
    with pytest.raises(pd.errors.MergeError): paired_conformal(pd.concat([a, a]), frame('B5c', 1))


def test_median_and_throughput_units():
    latency, throughput = measured_median([4., 1., 2.], 64)
    assert latency == 2 / 64
    assert throughput == 32
    with pytest.raises(ValueError): measured_median([1., 2.], 64)

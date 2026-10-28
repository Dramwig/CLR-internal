import pytest
from clr.utils.stats import mean_ci
from clr.utils.config import load_config

def test_ci_uses_seeds_not_rollouts():
    s=mean_ci([.1,.2,.3]);assert abs(s['mean']-.2)<1e-10
    assert .248<s['ci95_halfwidth']<.249
    with pytest.raises(ValueError):mean_ci([.1,.2])

def test_yaml_duplicates_and_unknown_override(tmp_path):
    p=tmp_path/'c.yaml';p.write_text('seed: 1\nseed: 2\n')
    with pytest.raises(Exception):load_config(p)
    p.write_text('source:\n  seed: 1\n')
    assert load_config(p,['source.seed=3'])['source']['seed']==3
    with pytest.raises(KeyError):load_config(p,['source.seed_typo=3'])

from clr.models.clr_v1 import Lite
from scripts.m41_diagnose import module_stats

def test_module_logger_handles_lite_without_source_parameter():
    stats=module_stats(Lite(vocab_size=40,hidden=32,layers=4,heads=4))
    assert 'source_token' not in stats and stats['adapter']['parameters']>0

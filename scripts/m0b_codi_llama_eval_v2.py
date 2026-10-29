"""Reuse v2 evaluation/statistics with the explicitly audited LLaMA loader."""
import argparse
import scripts.m01_protocol_eval as runner
from clr.sources.codi_llama_v2 import load
from clr.utils.config import load_config
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
 runner.load=load
 runner.main(load_config(a.config,a.overrides))

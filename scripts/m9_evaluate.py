"""Reuse frozen M6 calibration/inference; all training must be hash-locked."""
import sys
sys.dont_write_bytecode=True
import argparse
from clr.utils.config import load_config
from clr.utils import m9_runtime
import scripts.m6_evaluate as evaluate

def main(c):
    evaluate.Budget=m9_runtime.Budget;evaluate.main(c)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

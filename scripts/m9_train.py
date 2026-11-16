"""Reuse frozen M6 trainer with an independent M9 runtime and output root."""
import sys
sys.dont_write_bytecode=True
import argparse
from clr.utils.config import load_config
from clr.utils import m9_runtime
import scripts.m6_train as train

def main(c):
    train.Budget=m9_runtime.Budget;train.live=m9_runtime.live
    train.main(c)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

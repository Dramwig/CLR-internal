"""Same deterministic runner, explicitly substituting only the audited loader."""
import runpy
import clr.sources.codi_v2 as original
from clr.sources.codi_llama_v2 import load
if __name__=='__main__':
 original.load=load
 runpy.run_module('scripts.m01_determinism',run_name='__main__')

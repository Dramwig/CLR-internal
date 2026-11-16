"""Source and reader smoke admission, then non-test HiddenBench training."""
import argparse,json,subprocess
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def execute(python,script,config,name):
 p=Path('reports/M10_evidence')/name
 if p.exists():raise FileExistsError(p)
 with p.open('x') as f:subprocess.run([python,script,config],stdout=f,stderr=subprocess.STDOUT,check=True)
def main(c):
 sm=json.loads(Path(c['source_smoke']).read_text())
 if not sm['passed'] or sm['estimated_all_65_GPUh_conservative']>2:raise ValueError('HiddenBench source admission failed')
 execute('.venv-comm/bin/python','scripts/m10_source_hiddenbench.py',c['source_full'],'hiddenbench_source_full.log')
 execute('.venv/bin/python','scripts/m10_train_hiddenbench.py',c['reader_smoke'],'hiddenbench_reader_smoke.log')
 sm=json.loads(Path(c['reader_smoke_ref']).read_text());hist=json.loads((Path(sm['run'])/'history.json').read_text());met=json.loads((Path(sm['run'])/'metrics.json').read_text())
 # Three seeds, two LRs, 16 epochs, 20x training-label expansion. Include
 # doubled validation/IO margin and source allowance. Selection uses no test.
 estimate=met['gpu_hours']*3*2*16*20*2+1
 write_json('data/M10_v1/hiddenbench/budget_admission.json',{'source_smoke':c['source_smoke'],'reader_smoke':sm['run'],'estimated_GPUh':estimate,'limit_GPUh':12,'admitted':estimate<12,'test_opened':False})
 if estimate>=12:raise ValueError('HiddenBench reader budget admission failed')
 execute('.venv/bin/python','scripts/m10_evaluate_hiddenbench.py',c['eval_smoke'],'hiddenbench_eval_smoke.log')
 execute('.venv/bin/python','scripts/m10_train_hiddenbench.py',c['reader_full'],'hiddenbench_reader_full.log')
 write_json('data/M10_v1/hiddenbench/non_test_completed.json',{'controller':c,'test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

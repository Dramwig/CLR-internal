"""Resume-aware CPU controller for completed M5b evidence; no new GPU jobs."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from clr.utils.config import load_config
from clr.utils.idea_a import write_json


def read(path):return json.loads(Path(path).read_text())


def main(c):
    root=Path(c['output']);journal=root/'finish_pipeline.json';state={'stage':'waiting_dependencies'}
    write_json(journal,state);started=set()
    def execute(config, script, logfile):
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        out=Path(logfile).open('w');p=subprocess.Popen([sys.executable,script,str(config)],stdout=out,stderr=subprocess.STDOUT,env=env)
        return p,out
    def done(name):
        p=root/name
        return p.exists() and read(p).get('stage')=='completed'
    procs={}
    base=load_config(c['bc_base'])
    try:
        while True:
            for name in ['A_pipeline.json','D_pipeline.json','queue_C.json']:
                p=root/name
                if p.exists() and read(p).get('stage')=='failed':raise RuntimeError(f'Prerequisite failed: {name}')
            for part,(p,out) in list(procs.items()):
                if p.poll() is not None:
                    out.close();del procs[part]
                    if p.returncode:raise RuntimeError(f'Analysis/report failed: {part}; see reports/M5b_finish_{part}.log')
                    state[part]='completed'
            if done('A_pipeline.json') and 'A' not in started:
                procs['A']=execute(c['a_config'],'scripts/m5b_analyze_a.py','reports/M5b_finish_A.log');started.add('A');state['A']='running'
            if (root/'D_B_rescore.json').exists() and 'B' not in started:
                cfg={**base,'case':'B_analysis','B_only':True,'D_B_run':read(root/'D_B_rescore.json')['run']}
                path=root/'pipeline_configs/analysis_B.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                procs['B']=execute(path,'scripts/m5b_analyze_bc.py','reports/M5b_finish_B.log');started.add('B');state['B']='running'
            if done('D_pipeline.json') and done('queue_C.json') and state.get('B')=='completed' and 'BC' not in started:
                cfg={**base,'D_B_run':read(root/'D_B_rescore.json')['run'],'C_runs':[Path(f'reports/M5b_C_{s}.log').read_text().splitlines()[0] for s in c['seeds']], 'D_C_runs':read(root/'D_pipeline.json')['C_runs']}
                Path(c['bc_config']).write_text(yaml.safe_dump(cfg,sort_keys=False))
                procs['BC']=execute(c['bc_config'],'scripts/m5b_analyze_bc.py','reports/M5b_finish_BC.log');started.add('BC');state['BC']='running'
            if state.get('A')=='completed' and state.get('BC')=='completed':break
            state.update(stage='analyzing_or_waiting',active=list(procs));write_json(journal,state);time.sleep(20)
        completion={'A':read(root/'A_evaluation_complete.json'),'A_training':read(root/'A_training_lock.json'),
                    'B_run':base['B_run'],'C_runs':cfg['C_runs'],'D':read(root/'D_pipeline.json'),
                    'D_generation':load_config('configs/M5b_d_pipeline_v2.yaml')['generation_run'],
                    'analysis_runs':{name:read(root/name) for name in ['A_analysis_run.json','B_analysis_run.json','BC_analysis_run.json']},
                    'milestone_complete':False,'stage':'reporting'}
        write_json(root/'completion.json',completion)
        state.update(stage='reporting');write_json(journal,state)
        p,out=execute(c['report_config'],'scripts/m5b_report.py','reports/M5b_finish_report.log');p.wait();out.close()
        if p.returncode:raise RuntimeError('Report failed; see reports/M5b_finish_report.log')
        state.update(stage='reports_completed_pending_final_audit');write_json(journal,state)
    except Exception as e:
        state.update(stage='failed',error=str(e));write_json(journal,state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

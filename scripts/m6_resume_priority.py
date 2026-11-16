"""Wait for existing GPU queues, then hand over the CPU controller."""
import argparse,json,os,subprocess,time
from pathlib import Path
from scripts.m6_pipeline import ROOT,PY
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import snapshot,live


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);snapshot(run,c)
    while any(Path(f'/proc/{pid}/cmdline').exists() and b'm6_' in Path(f'/proc/{pid}/cmdline').read_bytes() for pid in c['prior_pids']):time.sleep(10)
    path=Path(c['pipeline']);state=json.loads(path.read_text())
    required=c.get('required_models',[f'single_llama_{kind}_{sd}' for kind in ['lite','b3'] for sd in [11,22,33]])
    if not all(key in state['models'] for key in required):
        raise RuntimeError('Controller handover lacks required completed checkpoints; inspect before resuming')
    state.setdefault('controller_handovers',[]).append({'run':str(run.path),'reason':c.get('reason','D before C2/C4'),
        'prior_pids':c['prior_pids'],'GPU_children_interrupted':False,'training_config_changed':False})
    write_json(path,state)
    live(stage=c.get('handover_live_stage','C1_completed_controller_handover'),pipeline_log=c['log'],
         controller_resume_run=str(run.path))
    log=Path(c['log'])
    if log.exists():raise FileExistsError(log)
    env={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1',**c['runtime_env']}
    with log.open('w') as stream:
        proc=subprocess.run([str(PY),'scripts/m6_pipeline.py',c['config']],env=env,stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError('Resumed M6 controller failed; '+str(log))
    run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

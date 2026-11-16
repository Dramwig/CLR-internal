"""Budgeted extension smoke/full scan, fixed choice, then resume M6 production."""
import argparse,json,time
from pathlib import Path
import numpy as np
import yaml
from scripts.m6_pipeline import Controller
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import snapshot,hours,live


def main(c):
 with recorded(c) as run:
    print(run.path,flush=True);snapshot(run,c);ctl=Controller(c);ctl.state.update(status='running',controller_run=str(run.path));ctl.save()
    try:
        smoke=ctl.execute('noise_extension_smoke_v2','scripts/m6_source.py',load_config(c['smoke_config']))
        times=json.loads(Path(smoke,'timing.json').read_text())['noise'];used=sum(r['gpu_hours'] for r in hours() if r.get('budget_group')=='noise_extension')
        forecast=300*float(np.mean(times))/3600+.01
        write_json(ctl.output/'estimate.json',{'used_extension_gpu_hours':used,'full_forecast_gpu_hours':forecast,'limit':2,'unit':'GPU job wall time; same K16/all6t/all4sigma'})
        if used+forecast>=1.98:raise RuntimeError('Extension smoke projects beyond 2GPUh; no grid/K/precision reduction authorized')
        full=load_config(c['full_config']);full['smoke_run']=smoke
        extension=ctl.execute('noise_extension_full_v2','scripts/m6_source.py',full)
        choice=ctl.execute('noise_choice_v2','scripts/m6_choose_noise.py',{'milestone':'M6','stage':'noise_choice_v2','section':'B','output':c['production_output'],'original_run':c['original_run'],'extension_run':extension,'preregistration':c['preregistration']})
        completion={'smoke_run':smoke,'extension_run':extension,'selection_run':choice,'preregistration':c['preregistration']};write_json(Path(c['production_output'])/'noise_extension_completion.json',completion)
        production=load_config(c['production_config']);production.update(version='noise-v2',noise_selection_run=choice,pipeline_filename='pipeline_noise_v2.json',selected_sigma_smoke_config='configs/M6_source_smoke_cost.yaml')
        path=Path('configs/M6_pipeline_noise_v2.yaml');text=yaml.safe_dump(production,sort_keys=False,allow_unicode=True)
        if path.exists() and path.read_text()!=text:raise ValueError('Production v2 config already differs')
        path.write_text(text)
        ctl.execute('production_noise_v2','scripts/m6_pipeline.py',production)
        ctl.state['status']='completed';ctl.save();run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)
    except Exception as exc:
        ctl.state.update(status='blocked_or_failed',reason=str(exc));ctl.save();live(stage='blocked_or_failed',reason=str(exc),extension_pipeline=str(ctl.statepath),milestone_complete=False)
        run.write_metrics(status='failed',gpu_hours=0,source_forward_calls=0,error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

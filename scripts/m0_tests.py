import argparse,subprocess,sys,time
from clr.utils.records import recorded
from clr.utils.config import load_config
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
with recorded(load_config(a.config,a.overrides)) as run:
 t=time.perf_counter();result=subprocess.run([sys.executable,'-m','pytest','-q'],text=True,capture_output=True)
 (run.path/'pytest.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError('Unit tests failed; see pytest.log')
 run.write_metrics(status='completed',pytest_output=result.stdout,wall_seconds=time.perf_counter()-t,gpu_hours=0,source_forward_calls=0)
 print(run.path,result.stdout)

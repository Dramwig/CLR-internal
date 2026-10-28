"""Detached full-run supervisor; configs remain the only command-line inputs."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from clr.utils.config import load_config
p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();c=load_config(a.config)
status_path=Path(c['status']);status={'pid':os.getpid(),'started':datetime.now(timezone.utc).isoformat(),'status':'running','smoke_run':c['smoke_run']}
def save():status_path.write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
save()
try:
    env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUTF8='1')
    with open(c['log'],'x',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'scripts/m01_noise.py',c['full_config'],'smoke_run='+c['smoke_run']],stdout=log,stderr=subprocess.STDOUT,env=env)
        status['child_pid']=child.pid;save();code=child.wait()
    if code:raise RuntimeError(f'Full noise exited {code}; see {c["log"]}')
    run=Path(c['log']).read_text(encoding='utf-8').strip().splitlines()[-1]
    status['run']=run;status['status']='reporting';save()
    subprocess.run([sys.executable,'scripts/m01_noise_report.py',c['report_config'],'run='+run],env=env,check=True)
    status['status']='completed';save()
except BaseException:
    status['status']='failed';status['error']=traceback.format_exc();save()
    with open('docs/BLOCKERS.md','a',encoding='utf-8') as f:f.write('\n## Full noise job failure\n'+status['error']+'\n')
    raise

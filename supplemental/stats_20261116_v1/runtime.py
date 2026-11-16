"""New-only run ledger; never write the old milestone logs or results."""
import os, json, time, tarfile, hashlib, traceback
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from clr.utils.idea_a import Run, write_json

ROOT=Path('supplemental/stats_20261116_v1')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def cost(exclude=None):
    total=0.
    for p in (ROOT/'jobs').glob('*.json'):
        r=json.loads(p.read_text());path=Path(r['run'])
        if exclude and path.resolve()==Path(exclude).resolve():continue
        m=json.loads((path/'metrics.json').read_text())
        value=m.get('gpu_hours',0.)
        if m['status']=='running':
            try:os.kill(r['pid'],0);value=max(value,(time.time()-r['started'])/3600)
            except ProcessLookupError:pass
        total+=value
    return total

@contextmanager
def recorded(c):
    start=time.monotonic()
    with Run(c,root=Path.cwd(),run_root=ROOT/'runs') as run:
        (ROOT/'jobs').mkdir(exist_ok=True)
        write_json(ROOT/'jobs'/f'{run.path.name}.json',{'run':str(run.path),'started':time.time(),'pid':os.getpid()})
        files=sorted([*Path('clr').rglob('*.py'),*Path('scripts').glob('*.py'),*ROOT.glob('*.py'),*ROOT.glob('*.yaml')])
        write_json(run.path/'code_sha256.json',{str(p):digest(p) for p in files})
        with tarfile.open(run.path/'startup_source.tar.gz','w:gz') as tar:
            for p in files:tar.add(p)
        try:yield run
        except BaseException:
            (run.path/'traceback.txt').write_text(traceback.format_exc())
            with (ROOT/'BLOCKERS.md').open('a') as f:f.write(f'\n{datetime.now().isoformat()} {run.path}:\n{traceback.format_exc()}\n')
            raise
        finally:run.write_metrics(gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)

class Budget:
    def __init__(self,run,c,start):self.path=run.path;self.c=c;self.start=start
    def check(self):
        own=(time.monotonic()-self.start)/3600;total=cost(self.path)+own
        write_json(self.path/'budget.json',{'own_gpu_hours':own,'total_gpu_hours':total,'limit':25})
        if total>=24.95:raise TimeoutError('Supplement aggregate 25 GPUh hard limit')
        if datetime.now().astimezone()>=datetime.fromisoformat(self.c['deadline']):raise TimeoutError('User deadline')
        if self.c['mode']=='smoke' and own*3600>=900:raise TimeoutError('Smoke 900 second hard limit')

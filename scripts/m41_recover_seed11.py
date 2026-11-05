"""Explicit seed11 retry, while preserving the two original active workers."""
import argparse,json,time,subprocess,os
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m41_status import update

def main(c):
    smoke=Path(c['memory_smoke_receipt']);start=time.monotonic()
    while not smoke.exists():
        if time.monotonic()-start>900:raise TimeoutError('Activation checkpointing memory smoke missing')
        time.sleep(5)
    sm=json.loads(smoke.read_text());metrics=json.loads(Path(sm['run'],'metrics.json').read_text())
    if metrics['status']!='completed' or metrics.get('test_opened'):raise ValueError('Memory smoke failed')
    update('training_recovery','seed11启用activation checkpointing，等价性及192-token压力smoke通过；seed22/33原训练继续，模型/batch/精度/曝光不变',test_opened=False)
    with Path('reports/M41_main_v1_11_checkpointing_retry.log').open('w') as log:subprocess.run(['.venv/bin/python','scripts/m41_train.py',c['retry_config']],check=True,stdout=log,stderr=subprocess.STDOUT)
    # Old controller must finish its remaining owned children before takeover.
    while Path(f"/proc/{c['old_pipeline_pid']}").exists():time.sleep(20)
    root=Path(c['output'])
    required=[root/f'trained_main_{v}_{s}.json' for s in [22,33] for v in ['v1','lite','b3_multi']]
    if not all(p.exists() for p in required):raise RuntimeError('Another original seed failed; explicit investigation required')
    subprocess.run(['.venv/bin/python','scripts/m41_pipeline.py',c['continuation_config']],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
 try:main(load_config(a.config,a.overrides))
 except Exception as e:
  update('blocked_error',f'M4.1恢复失败：{type(e).__name__}: {e}',test_opened=False)
  Path('docs/BLOCKERS.md').open('a').write(f'\n## M4.1 seed11恢复错误\n{type(e).__name__}: {e}；见reports/M41_recovery.log。未跳过关卡或缩小模型/曝光。\n');raise

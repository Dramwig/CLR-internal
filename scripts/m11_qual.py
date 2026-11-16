"""YAML-only entry point for frozen-weight qualitative data export."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from clr.export.qual import Export, json_write
from clr.utils.config import load_config
from clr.utils.records import recorded


def main(c):
    torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.set_device(c['device'])
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'freeze.json').exists():raise ValueError('M11 export frozen; use new version')
    with recorded(c) as run:
        started=time.monotonic();print(run.path,flush=True)
        (run.path/'gpu_start.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        status='failed'
        try:
            export=Export(c,run);counts,size=export.execute();status='completed'
            result=dict(passed=True,run=str(run.path),counts=counts,bytes=size,wall_seconds=time.monotonic()-started,
                        test_opened=c['stage']!='smoke',gpu_hours=(time.monotonic()-started)/3600,source_forward_calls=0)
            if c['stage']=='smoke':
                assert result['wall_seconds']<=900
                json_write(root/'smoke_passed.json',result)
            else:json_write(root/'export_complete.json',result)
            print(json.dumps(result),flush=True)
        except BaseException as error:
            with Path('docs/BLOCKERS.md').open('a') as f:
                f.write(f'\n## M11-qual 导出失败（{c["stage"]}）\n- run `{run.path}`；{type(error).__name__}: {error}。未训练/重校准/改冻结；同配置修复后重试。\n')
            raise
        finally:
            elapsed=(time.monotonic()-started)/3600
            run.write_metrics(status=status,gpu_hours=elapsed,source_forward_calls=0,test_opened=c['stage']!='smoke')
            json_write(root/('cost_'+run.path.name+'.json'),dict(run=str(run.path),stage=c['stage'],status=status,gpu_hours=elapsed,source_forward_calls=0,device=c['device'],gpu='NVIDIA GeForce RTX 4090',scope='conservative whole process elapsed incl CPU/loading; includes smoke and failures'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

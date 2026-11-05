"""Wait for existing workers; recover only unfinished checkpointed jobs."""
import argparse,json,subprocess,time
from pathlib import Path
import yaml
from clr.utils.config import load_config
from clr.utils.m41_status import update

def main(c):
    for pid in c['wait_pids']:
        while Path(f'/proc/{pid}').exists():
            time.sleep(20)
    root=Path(c['output'])
    if (root/'freeze.json').exists():return
    for old in c['recover_runs']:
        run=Path(old);cfg=json.loads((run/'config.json').read_text())
        receipt=root/f"trained_{cfg['case']}_{cfg['variant']}_{cfg['seed']}.json"
        if receipt.exists():continue
        metrics=json.loads((run/'metrics.json').read_text())
        if metrics['status']!='failed' or not (run/'last.pt').exists():
            raise RuntimeError('Recovery requires failed run and epoch checkpoint: '+old)
        cfg.update(resume_run=old,total_budget_hours=c['total_budget_hours'],run_budget_hours=c['run_budget_hours'],budget_authorization=c['authorization'])
        target=root/'train_configs'/'main_v1_11_budget50_resume.yaml'
        target.write_text(yaml.safe_dump(cfg,sort_keys=False))
        update('training_recovery','按用户授权50 GPUh预算，从epoch检查点恢复未完成seed；模型/数据/曝光不变',test_opened=False)
        subprocess.run(['.venv/bin/python','scripts/m41_train.py',str(target)],check=True)
    subprocess.run(['.venv/bin/python','scripts/m41_pipeline.py',c['pipeline']],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    try:main(load_config(a.config,a.overrides))
    except Exception as e:
        update('blocked_error',f'预算接续错误：{type(e).__name__}: {e}',test_opened=False)
        with Path('docs/BLOCKERS.md').open('a') as f:f.write(f'\n## M4.1 预算接续错误\n{type(e).__name__}: {e}；见reports/M41_budget50_resume.log。\n')
        raise

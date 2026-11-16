"""Independent CPU watchdog: enforce 22 GPUh resident wall budget, never touch test."""
import argparse,json,os,signal,time
from pathlib import Path
from datetime import datetime,timezone

def main(config):
 import yaml
 c=yaml.safe_load(Path(config).read_text());root=Path(c['output'])
 while not (root/'freeze.json').exists():
  rows=[];active=[]
  for p in Path('runs').glob('*/config.json'):
   cfg=json.loads(p.read_text())
   if cfg.get('milestone')!='M10b':continue
   mf=p.parent/'metrics.json';bf=p.parent/'budget.json';m=json.loads(mf.read_text()) if mf.exists() else {};b=json.loads(bf.read_text()) if bf.exists() else {}
   charge=max(m.get('gpu_hours',0),b.get('own_gpu_hours',0))
   if m.get('status') not in ['completed','failed']:
    lease=p.parent/'gpu_lease.json';pid=json.loads(lease.read_text())['pid'] if lease.exists() else None
    if pid and Path(f'/proc/{pid}').exists():
     cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
     if 'm10b_' not in cmd:raise RuntimeError('GPU lease pid reused; no process killed')
     charge=(b.get('own_gpu_hours',0)+(time.time()-bf.stat().st_mtime)/3600) if bf.exists() else (time.time()-p.stat().st_mtime)/3600
     active.append((pid,p.parent,charge))
   elif m.get('status')=='failed' and not charge:charge=.001
   rows.append({'run':str(p.parent),'gpu_hours_live':charge,'status':m.get('status')})
  total=sum(r['gpu_hours_live'] for r in rows)
  d={'GPUh_live':total,'hard_limit_GPUh':22,'stop_guard_GPUh':21.97,'active_pids':[p for p,r,h in active],'runs':rows,'updated_utc':datetime.now(timezone.utc).isoformat()};(root/'watchdog.json').write_text(json.dumps(d,indent=2)+'\n')
  if total>=21.97:
   for pid,run,charge in active:
    os.kill(pid,signal.SIGTERM);mf=run/'metrics.json';m=json.loads(mf.read_text());m.update(status='failed',gpu_hours=charge,stop_reason='M10b hard-budget watchdog guard');mf.write_text(json.dumps(m,indent=2)+'\n')
   (root/'budget_stopped.json').write_text(json.dumps(d,indent=2)+'\n')
   Path('reports/M10b_REVIEW.md').write_text('# M10b 审查\n\n### (a) 状态一句话\n未完成：22GPUh硬预算保护触发，实验停止；不减少seed或标签。\n\n### (b) 关卡清单\n通过val-only smoke；未通过完整三seed/最终配对比较。\n\n### (c) 核心主表\nCLR−B5c Q3/Q4 test差值及95%CI为NA（完整实验未完成）。\n\n### (d) 本里程碑新增的偏离与阻塞\n成本达到保护线，所有已运行数据/失败保留；详见data/M10b_v1/watchdog.json、budget_stopped.json。LLaMA和审计任务未执行。\n\n### (e) 需要审查者决定的问题\n无。不追加预算，不以少seed/少标签替代。\n\n### (f) 证据路径\ndata/M10b_v1/，docs/M10b_TASK.md，configs/M10b/。\n')
   break
  time.sleep(c['interval_seconds'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(a.config)

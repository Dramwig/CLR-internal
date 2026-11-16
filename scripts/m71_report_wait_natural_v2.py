"""Compute locked primary statistics while fixed natural diagnostics finish."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import scripts.m71_report as report
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

original=report.natural_statistics
def await_natural(root,rep,c,out,cache):
    while True:
        pointers=[root/rep/'natural'/f'CLR_{seed}.json' for seed in c['seeds']]
        if all(p.exists() for p in pointers):
            if all(json.loads((Path(json.loads(p.read_text())['run'])/'metrics.json').read_text())['status']=='completed' for p in pointers):
                break
        time.sleep(10)
    return original(root,rep,c,out,cache)
report.natural_statistics=await_natural

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    c=load_config(a.config,a.overrides)
    report.main(c)
    note=('收尾调度：natural test按原batch2配对分3个GPU分片；各设备先复放固定val配对，概率/token一致后才运行test。'
          '原进程完成val后由已有文件保护退出（FileExistsError），未中断前向；完整保留失败run与成本。'
          '统计与诊断并行，仅调度变更，未改子集/模型/解码/阈值。')
    review=Path('reports/M7.1_REVIEW.md')
    review.write_text(review.read_text().replace('### (e) 需要审查者决定的问题',note+'\n\n### (e) 需要审查者决定的问题'))
    if review.stat().st_size>8000:raise RuntimeError('Review exceeds 8000 UTF8 bytes after execution note')
    details=Path('reports/M7.1_details.md');details.write_text(details.read_text()+'\n\n'+note+'\n')
    write_json(Path(c['output'])/'report_wait_natural_v2_completed.json',{'review_bytes':review.stat().st_size,'execution_note':note})

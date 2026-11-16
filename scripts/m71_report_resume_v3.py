"""Resume rendering from completed, hashed statistics; keep the failed run."""
import sys
sys.dont_write_bytecode=True
import argparse,json,shutil
from pathlib import Path
import pandas as pd
import scripts.m71_report_v3 as report
from clr.audit.features import sha256
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    cached=Path(c['cached_statistics'])
    write_json(Path(c['output'])/'statistics_cache_resume_v3.json',{
        'original_report_run':'runs/20261107T121722207176Z_7dd6f8e381b6_5e441d',
        'reason':'canonicalize current-run path during cost closure; no statistical protocol change',
        'sha256':{str(p):sha256(p) for p in sorted(cached.iterdir()) if p.is_file()}})
    def detection_rows(*args):return pd.read_csv(cached/'detection_scores.csv.gz')
    def detection_summary(frame,config,cache,panel):
        result=pd.read_csv(cached/'detection_group_bootstrap.csv')
        rows=result[result.panel==panel].to_dict('records');rates={}
        for key,f in frame.groupby(['split','kind','k','alpha','category']):
            rates[key]=f.groupby('group_id').rejected.mean().sort_index()
        return rows,rates
    def qa(root,rep,config,out,cache):
        for name in ['qa_group_bootstrap.csv','qa_category_group_bootstrap.csv','qa_per_seed.csv','qa_CLR_minus_B3.csv']:
            shutil.copyfile(cached/name,out/name)
        return pd.read_csv(cached/'qa_group_bootstrap.csv'),pd.read_csv(cached/'qa_CLR_minus_B3.csv')
    def receiver(root,rep,config,out,cache):
        shutil.copyfile(cached/'receiver_causal.json',out/'receiver_causal.json')
        return json.loads((cached/'receiver_causal.json').read_text())
    def natural(root,rep,config,out,cache):
        shutil.copyfile(cached/'natural_commitment_distribution.csv',out/'natural_commitment_distribution.csv')
        return pd.read_csv(cached/'natural_commitment_distribution.csv')
    def plot(det,out,**kwargs):
        for ext in ['png','pdf']:shutil.copyfile(cached/('counterfactual_detection.'+ext),out/('counterfactual_detection.'+ext))
    gates=iter(json.loads((cached/'C1_group_bootstrap.json').read_text()).values())
    report.detection_rows=detection_rows;report.summarize_detection=detection_summary
    report.qa_statistics=qa;report.receiver_statistics=receiver;report.natural_statistics=natural
    report.plot_detection=plot;report.gate_intervals=lambda *args,**kwargs:next(gates)
    report.main(c)
    note=('收尾调度：natural test按原batch2配对分3个GPU分片；各设备先复放固定val配对，概率/token一致后才运行test。'
          '原进程完成val后由已有文件保护退出（FileExistsError），未中断前向；完整保留失败run与成本。'
          '报告成本闭合曾因相对/绝对路径比较误计自身为open，保留失败run；新版本复用已落盘统计，未改子集/模型/解码/阈值。')
    path=Path('reports/M7.1_REVIEW.md')
    text=path.read_text().replace('### (e) 需要审查者决定的问题',note+'\n\n### (e) 需要审查者决定的问题')
    if len(text.encode())>8000:raise ValueError('Review exceeds 8000 bytes')
    path.write_text(text)
    details=Path('reports/M7.1_details.md');details.write_text(details.read_text()+'\n\n'+note+'\n')
    write_json(Path(c['output'])/'report_wait_natural_v2_completed.json',{'review_bytes':len(text.encode()),'execution_note':note,'statistics':'analysis_v3'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

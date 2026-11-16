"""Clarify evidence limits in the review before the final immutable freeze."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import scripts.m71_natural_finish_v2 as finish
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

original_execute=finish.execute
def execute(script,config,label):
    if script=='scripts/m71_freeze.py':
        review=Path('reports/M7.1_REVIEW.md');text=review.read_text()
        text=text.replace('历史raw逐张量对照仍缺原artifact，不能宣称完全通过。','历史raw逐张量对照：未通过（缺原artifact）。')
        note=('文本基线是可见统计特征的logistic；0事件的非参数bootstrap区间退化，不代表总体率为0。'
              '监督参照A3未学会，不能充当oracle上界。收尾控制器整数YAML键的入口错误已以字符串键修复，未产生模型调用。')
        text=text.replace('### (e) 需要审查者决定的问题',note+'\n\n### (e) 需要审查者决定的问题')
        if len(text.encode())>8000:raise ValueError('Final review exceeds 8000 UTF8 bytes')
        review.write_text(text)
        details=Path('reports/M7.1_details.md');details.write_text(details.read_text()+'\n\n'+note+'\n')
        write_json(Path('data/M71_v1/review_final_annotations_v3.json'),{'review_bytes':len(text.encode()),'notes':note})
    return original_execute(script,config,label)
finish.execute=execute

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    finish.main(load_config(a.config,a.overrides))

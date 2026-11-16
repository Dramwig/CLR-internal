"""Freeze independent HiddenBench groups and fact-panel audit before training."""
import argparse,json
from pathlib import Path
from clr.audit.m10_hiddenbench import read_raw,public_group,digest,make_panels
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 items=read_raw(c);groups=sorted({public_group(e) for e in items},key=lambda g:digest(str(c['seed'])+g));splits={};pos=0
 for split,n in c['group_counts'].items():splits[split]=groups[pos:pos+n];pos+=n
 if pos!=len(groups):raise ValueError('All independent source groups must be used')
 result={'raw_sha256':c['raw_sha256'],'splits':splits,'panels':{},'counterfactual':False,'commitment_source':'programmatic','expected_answers':'parse audited fact from visible commitment; never private label','adaptation':'explicit audited-fact assertion plus original conclusion; four free-text fact candidates','template_train':[0,1,2,3],'template_heldout':[4,5],'strict_1pct_threshold':'positive infinity: fewer than 99 independent clean cal_B groups'}
 for split,gs in splits.items():result['panels'][split]=make_panels([e for e in items if public_group(e) in gs],c['seed'])
 root=Path(c['hidden_root']);root.mkdir(exist_ok=True,parents=True);p=root/'protocol.json'
 if p.exists():raise FileExistsError('Protocol already frozen')
 write_json(p,result);print({s:len(g) for s,g in splits.items()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

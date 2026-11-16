"""M10b final trained-language comparator; only frozen CSVs are consumed."""
from clr.paper.render import Context,SOURCE_NAMES,escape,interval,one
ORDER=['B0','B3','B3-prefix','B6','B5c','CLR-lite-single','CLR-lite','CLR-lite-multi']
def t1(c):
 f=c.read('main');split=c.config.get('split','test');f=f[f.split==split];rows=[]
 for source in ['gpt2','llama']:
  for model in ORDER:
   row=[SOURCE_NAMES[source],escape(model)]
   for q,metric,scale in [('Q1','nll',1),('Q2','auroc',1),('Q3','accuracy',100),('Q4','accuracy',100),('Q3','Q3_minus_B0',100)]:
    row.append('NA' if model=='B5c' and (q in ['Q1','Q2'] or source=='llama') else interval(one(f,source=source,model=model,qtype=q,metric=metric),scale))
   rows.append(row)
 c.write_data(f);c.table(['Source','Reader',r'Q1 NLL $\downarrow$',r'Q2 AUROC $\uparrow$',r'Q3 acc. (\%)',r'Q4 acc. (\%)','Q3$-$B0 (pp)'],rows,f'{split}; clean states, seen templates, micro within step and macro over steps/three fixed seeds, 10000 problem-bootstrap intervals. B5c: frozen Qwen3-1.7B + two-layer mapping + rank-16 LoRA, one full training pass; training exposure is lower than the eight-pass CLR/B6. B5c Q1/Q2 and LLaMA are NA because excluded by M10b scope. Calibration uses cal_A temperature and cal_B Mondrian thresholds. Q1 continuation policies differ across sources.')
def main():
 import argparse,yaml
 from pathlib import Path
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();c=Context(yaml.safe_load(Path(a.config).read_text()));t1(c);c.finish()

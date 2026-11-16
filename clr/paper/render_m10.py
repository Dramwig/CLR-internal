"""M10 paper supplements; frozen M9 renderer remains read-only."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from clr.paper.render import Context,SOURCE_NAMES,escape,interval,one,errorline,COLORS

ORDER=['B0','B3','B3-prefix','B6','B5c','CLR-lite-single','CLR-lite','CLR-lite-multi']
def t1(c):
 f=c.read('main');split=c.config.get('split','test');f=f[f.split==split];rows=[]
 for source in ['gpt2','llama']:
  for model in ORDER:
   row=[SOURCE_NAMES[source],escape(model)]
   for q,metric,scale in [('Q1','nll',1),('Q2','auroc',1),('Q3','accuracy',100),('Q4','accuracy',100),('Q3','Q3_minus_B0',100)]:
    row.append(('--' if q in ['Q1','Q2'] else 'NR') if model=='B5c' else interval(one(f,source=source,model=model,qtype=q,metric=metric),scale))
   rows.append(row)
 c.write_data(f);c.table(['Source','Reader',r'Q1 NLL $\downarrow$',r'Q2 AUROC $\uparrow$',r'Q3 acc. (\%)',r'Q4 acc. (\%)','Q3$-$B0 (pp)'],rows,f'{split}; clean states and seen templates. Micro within step, macro over steps and three fixed seeds; 10000 problem-bootstrap intervals. B6 uses separately trained single attention probes per type. B5c NR: its fastest measured GPT2-only one-epoch/three-seed runtime estimate exceeds the 12 GPUh allocation, before other tasks and evaluation. Q1 continuation policies differ across sources.')

def t4(c):
 f=c.read('detection');rows=[]
 for split in ['test','reader_val']:
  for kind in ['CLR','B3','B6','B5c','linear','norm','text']:
   k=16 if kind in ['CLR','B3','B6','linear'] else 0
   base=dict(panel='full',split=split,kind=kind,k=k,category='counterfactual');a=one(f,**base,alpha=.05);b=one(f,**base,alpha=.01)
   rows.append([escape(split),escape(kind),*(['NR']*4 if kind=='B5c' else [interval(a,100),interval(b,100),interval(a,mean='AUROC',lo='AUROC_lo',hi='AUROC_hi'),interval(a,100,mean='FPR',lo='FPR_lo',hi='FPR_hi')])])
 c.write_data(f);c.table(['Split','Auditor',r'CF detect @5\%',r'CF detect @1\%','AUROC',r'Clean FPR @5\%'],rows,'Synthetic v2, fixed-seed group-bootstrap intervals; k=16 for QA readers. Expectations use visible commitment and public constraints only. Text is restricted to its original authorized 300 test groups. Commitments are programmatic for all categories. B5c NR under the preassigned budget; it is not a defeated comparator.')

def t4_qa(c):
 f=c.read('qa');rows=[]
 for split in ['test','reader_val']:
  for kind in ['CLR','B3','B6','B5c','linear','B0']:
   rows.append([escape(split),escape(kind),*[('--' if t in ['A2','A3'] else 'NR') if kind=='B5c' else interval(one(f,split=split,kind=kind,task=t,templates='seen',metric='correct'),100) for t in ['A1','A2','A3','A4']]])
 c.write_data(f);c.table(['Split','Auditor',r'A1 acc. (\%)',r'A2 acc. (\%)',r'A3 acc. (\%)',r'A4 acc. (\%)'],rows,'Synthetic v2, seen templates; three fixed seeds and 10000 episode-group bootstrap draws. A2/A3 failures prevent claiming a general consistency-QA capability. B5c not run in full under its budget.')

def hiddenbench(c):
 f=c.read('hiddenbench');rows=[]
 for split in ['test','reader_val']:
  for kind in ['CLR','B3','B6','norm','text']:
   base=dict(split=split,kind=kind)
   cells=[escape(split),escape(kind)]
   for metric,scale in [('detect_0.05',100),('detect_0.01',100),('AUROC',1),('A1_accuracy',100),('A4_accuracy',100)]:cells.append(interval(one(f,**base,metric=metric),scale))
   rows.append(cells)
 c.write_data(f);c.table(['Split','Auditor',r'Mismatch @5\%',r'Mismatch @1\%','AUROC',r'A1 acc. (\%)',r'A4 acc. (\%)'],rows,'Exploratory HiddenBench fact-panel adaptation: 65 cases / 63 independent public-description groups; 13 train, 20 val, 4 cal_A, 20 cal_B, 6 test. Commitment explicitly states the audited fact and original conclusion. No counterfactual construction. Thresholds calibrate group-mean scores; 1% is +infinity and rejects nothing because fewer than 99 clean calibration groups exist. Six test groups give very weak uncertainty resolution.')

def pertype(c):
 f=c.read('main');rows=[]
 for split in ['test','reader_val']:
  for source in ['gpt2','llama']:
   for q,metric,scale in [('Q1','nll',1),('Q2','auroc',1),('Q3','accuracy',100),('Q4','accuracy',100)]:
    cells=[SOURCE_NAMES[source],escape(split),q,escape(metric)]
    cells.extend(interval(one(f,source=source,split=split,qtype=q,metric=metric,model=m),scale) for m in ['CLR-lite','CLR-per-type'])
    rows.append(cells)
 c.write_data(f[f.model.isin(['CLR-lite','CLR-per-type'])]);c.table(['Source','Split','Type','Metric','Multitype CLR','Separate-type CLR'],rows,'Identical CLR architecture; each separate-type checkpoint receives exactly the corresponding type exposure of the frozen multitype checkpoint (eight complete epochs). Accuracy is percent; NLL/AUROC are unscaled. Fixed three seeds, 10000 problem-cluster bootstrap intervals.',long=True)

def transfer(c):
 f=c.read('transfer');fig,axes=plt.subplots(1,2,figsize=(7.5,3.1));full=int(f.N_value.max())
 for ax,task in zip(axes,['A1','A4']):
  for init,color in [('scratch',COLORS['scratch']),('M9',COLORS['CLR'])]:
   z=f[(f.split=='test')&(f.task==task)&(f.initialization==init)&(f.metric=='accuracy')].sort_values('N_value')
   errorline(ax,z.N_value,100*z['mean'],100*z.lo,100*z.hi,color=color,marker='o',label='M9 body initialization' if init=='M9' else 'Scratch')
  ax.set(xscale='log',xlabel='Audit QA labels N (A1 + A4)',ylabel='Test accuracy (%)',title=task,ylim=(0,100));ax.set_xticks([100,300,1000,full],['100','300','1k','Full']);ax.legend(fontsize=7)
 fig.tight_layout();c.figure(fig);c.write_data(f)

BUILDERS={'t1':t1,'t4':t4,'t4_qa':t4_qa,'hiddenbench':hiddenbench,'pertype':pertype,'transfer':transfer}
def main(builder):
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();c=Context(yaml.safe_load(Path(a.config).read_text()));BUILDERS[builder](c);c.finish()

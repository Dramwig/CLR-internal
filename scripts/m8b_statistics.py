"""Frozen-setting summary and scientific k/epsilon plots."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.audit.m8b_statistics import select_candidate,group_ci
from clr.audit.m8b_runtime import snapshot,Cost
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.config import load_config

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);folder=root/c['analysis_name'];folder.mkdir(exist_ok=False)
        raw=[]
        for split in ['reader_val','test']:
            for p in sorted(root.glob(f'{c["evaluation_prefix"]}{split}_part*.json')):
                ref=json.loads(p.read_text());rs=[json.loads(l) for l in (Path(ref['run'])/'results.jsonl').open()]
                raw.extend({**r,'split':split} for r in rs)
        grouped=defaultdict(list);clean={};qa={}
        for r in raw:
            key=(r['split'],r['group_id']);clean[key]=r['clean_audit'];qa[key]=r['qa']
            grouped[(r['split'],r['group_id'],r['kind'],r['epsilon'],r['step_budget'],r['attack_design'],r['attack_seed'])].append(r)
        records=[]
        for (split,gid,kind,eps,steps,ad,aseed),rs in grouped.items():
            selected=select_candidate(rs)
            for design in (c['designs'] if kind!='E3' else [ad]):
                for seed in (c['reader_seeds'] if kind!='E3' else [aseed]):
                    for k in ([1,4,16,64] if design=='R1' else [16]):
                        a=selected['audit'][f'{design}:{seed}:{k}'];cl=selected['clean_audit'][f'{design}:{seed}:{k}']
                        for alpha in [.01,.05]:
                            detected=a[str(alpha)];fpr=cl[str(alpha)]
                            records.append({'split':split,'group_id':gid,'kind':kind,'epsilon':eps,'steps':steps,'design':design,'seed':seed,'k':k,'alpha':alpha,'failure':selected['changed_to_target'] and not detected,'target_accepted':selected['target_achieved'] and not detected,'target_achieved':selected['target_achieved'],'changed_to_target':selected['changed_to_target'],'detected':detected,'clean_FPR':fpr,'clean_correct':selected['clean_correct'],'actual_steps':selected['steps'],'lambda':selected['lambda'],'max_relative_norm':selected['max_relative_norm']})
        df=pd.DataFrame(records);df.to_csv(folder/'group_metrics.csv.gz',index=False)
        keys=['split','kind','epsilon','steps','design','k','alpha'];summary=[]
        for key,part in df.groupby(keys,dropna=False):
            row=dict(zip(keys,key));row['groups']=part.group_id.nunique();row['seeds']=part.seed.nunique()
            for metric in ['failure','target_accepted','target_achieved','changed_to_target','detected','clean_FPR']:
                pergroup=part.groupby('group_id')[metric].mean().values;mean,lo,hi=group_ci(pergroup);row[metric]=mean;row[metric+'_lo']=lo;row[metric+'_hi']=hi
            conditional=part[part.clean_correct]
            row['clean_correct_failure']=float(conditional.failure.mean()) if len(conditional) else None;row['mean_actual_steps']=float(part.actual_steps.mean());summary.append(row)
        out=pd.DataFrame(summary);out.to_csv(folder/'robustness_summary.csv',index=False)
        qrows=[]
        for (split,gid),models in qa.items():
            for ds,result in models.items():
                design,seed=ds.split(':')
                for task,values in result.items():qrows.append({'split':split,'group_id':gid,'design':design,'seed':int(seed),'task':task,'accuracy':np.mean(values)})
        qdf=pd.DataFrame(qrows);qdf.to_csv(folder/'QA_groups.csv',index=False);qsummary=[]
        for key,p in qdf.groupby(['split','design','task']):
            mean,lo,hi=group_ci(p.groupby('group_id').accuracy.mean());qsummary.append(dict(zip(['split','design','task'],key))|{'accuracy':mean,'lo':lo,'hi':hi})
        pd.DataFrame(qsummary).to_csv(folder/'QA_summary.csv',index=False)
        # Paired design differences versus R1 k16, same group and seed.
        paired=[]
        ref=df[(df.design=='R1')&(df.k==16)]
        for design in ['R4','R2','R3']:
            p=df[df.design==design].merge(ref,on=['split','group_id','seed','kind','epsilon','steps','alpha'],suffixes=('_design','_R1'))
            p['reduction']=p.failure_R1.astype(float)-p.failure_design.astype(float)
            for key,part in p.groupby(['split','kind','epsilon','steps','alpha'],dropna=False):
                mean,lo,hi=group_ci(part.groupby('group_id').reduction.mean());paired.append(dict(zip(['split','kind','epsilon','steps','alpha'],key))|{'design':design,'reduction':mean,'lo':lo,'hi':hi})
        pd.DataFrame(paired).to_csv(folder/'paired_reductions.csv',index=False)
        for alpha in [.05,.01]:
            fig,ax=plt.subplots(figsize=(6,4))
            for design in c['designs']:
                p=out[(out.split=='test')&(out.kind=='E3')&(out.steps==max(c['steps']))&(out.alpha==alpha)&(out.design==design)&(out.k==16)].sort_values('epsilon')
                ax.plot(p.epsilon,p.failure,'o-',label=design)
            ax.set(xlabel='Per-layer relative KV norm ε',ylabel='Robust failure rate',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(folder/f'epsilon_failure_{alpha}.png',dpi=180);fig.savefig(folder/f'epsilon_failure_{alpha}.pdf');plt.close(fig)
        p=out[(out.split=='test')&(out.kind=='E3')&(out.steps==max(c['steps']))&(out.alpha==.05)&(out.design=='R1')&(out.epsilon==.03)].sort_values('k')
        if len(p):
            fig,ax=plt.subplots(figsize=(6,4));ax.plot(p.k,p.failure,'o-',label='R1 mean NLL + conformal')
            base=float(p[p.k==1].failure.iloc[0]);changed=float(p[p.k==1].changed_to_target.iloc[0]);ks=np.array([1,4,16,64]);refcurve=changed*(base/max(changed,1e-12))**ks
            ax.plot(ks,refcurve,'--',label='Independent veto reference (different rule)');ax.set(xscale='log',xlabel='Number of random questions k',ylabel='Robust failure rate',ylim=(0,1));ax.set_xticks(ks,ks);ax.legend();fig.tight_layout();fig.savefig(folder/'k_failure.png',dpi=180);fig.savefig(folder/'k_failure.pdf');plt.close(fig)
            write_json(folder/'independence_reference.json',{'rule':'independent per-question veto, calibrated mean-NLL is not this rule','k':ks.tolist(),'curve':refcurve.tolist(),'single_question_failure':base,'changed_fraction':changed})
        write_json(folder/'completion.json',{'raw_rows':len(raw),'group_metric_rows':len(df),'summaries':len(out),'run':str(run.path),'test_groups':len([key for key in clean if key[0]=='test']),'CI':'10000 group bootstrap; fixed training seeds averaged per group; does not estimate unseen seed variance'})
        cost.update(status='completed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

"""Locked group/seed aggregation; no attack or checkpoint selection on test."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np
from clr.audit.m8c_search import select_candidate,failure,ratio_interval
from clr.audit.m8c_runtime import snapshot,Cost,total_cost
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def ci(values,seed=20261109):
    x=np.asarray(values,float);rng=np.random.default_rng(seed)
    draws=np.concatenate([x[rng.integers(len(x),size=(min(500,10000-i),len(x)))].mean(1) for i in range(0,10000,500)])
    return {'mean':float(x.mean()),'low':float(np.quantile(draws,.025)),'high':float(np.quantile(draws,.975)),'groups':len(x)}

def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);dest=root/c['analysis_folder'];dest.mkdir(exist_ok=True)
        raw=[];seen=set();matrix={};panel_group_ids={}
        for spec in c['matrix_specs']:
            refs=sorted(root.glob(spec['pattern']));assert len(refs)==spec['parts'],(spec,refs)
            groupkeys={}
            for refpath in refs:
                ref=json.loads(refpath.read_text());rpath=Path(ref['run']);met=json.loads((rpath/'metrics.json').read_text())
                assert met['status']=='completed' or spec.get('allow_failed_complete_prefix',False)
                complete_ids=set(ref.get('group_ids',[])) if spec.get('allow_failed_complete_prefix',False) else None
                with (rpath/'results.jsonl').open() as f:
                    for line in f:
                        r=json.loads(line);r['split']=spec['split'];r['panel']=spec['panel']
                        if complete_ids is not None and r['group_id'] not in complete_ids:continue
                        if r['epsilon'] not in spec.get('epsilons',c['epsilons']):continue
                        key=(spec['panel'],spec['split'],r['group_id'],r['kind'],r['epsilon'],r['attack_design'],r['attack_seed'],r['lambda'])
                        assert key not in seen;seen.add(key);raw.append(r)
                        groupkeys.setdefault(r['group_id'],set()).add(key[3:])
                        assert r['max_relative_norm']<=r['epsilon']*(1+1e-5)
                        if r['kind']=='E2':assert r['original_f_fp16_equal']
            assert len(groupkeys)==spec['groups'],(spec,len(groupkeys))
            panel_group_ids[spec['panel'],spec['split']]=set(groupkeys)
            if spec['split']=='reader_val':
                for (panel,split),ids in panel_group_ids.items():
                    if split=='reader_val' and panel!=spec['panel']:assert ids==set(groupkeys),'Val attack panels must use same frozen IDs'
            expected={(kind,eps,d,s,l) for kind in spec['kinds'] for eps in spec.get('epsilons',c['epsilons'])
                      for d in (c['designs'] if kind=='E3' else [None])
                      for s in (spec.get('reader_seeds',c['reader_seeds']) if kind=='E3' else [None])
                      for l in (spec['lambdas'] if kind=='E3' else [0.])}
            assert all(v==expected for v in groupkeys.values()),'Incomplete M8c matrix'
            if spec['split']=='test' and spec['panel']=='main':
                locked=json.loads((root/'test_ids.json').read_text())['group_ids']
                assert len(locked)==spec['groups'] and set(groupkeys)==set(locked)
            matrix[spec['split']+':'+spec['panel']]={'groups':len(groupkeys),'rows_per_group':len(expected),'passed':True}
        # Retain every lambda; the selected attack uses attack-side quantities only.
        buckets={}
        for r in raw:
            k=tuple(r[t] for t in ['panel','split','group_id','kind','epsilon','attack_design','attack_seed']);buckets.setdefault(k,[]).append(r)
        chosen=[select_candidate(rows) for rows in buckets.values()]
        groupmetrics=[]
        for rule,rows in [('selected',chosen),('lambda_rows',raw)]:
            for r in rows:
                for key,a in r['audit'].items():
                    d,s=key.split(':')
                    for alpha in [.01,.05]:
                        groupmetrics.append({'panel':r['panel'],'split':r['split'],'group_id':r['group_id'],'rule':rule,'kind':r['kind'],'epsilon':r['epsilon'],
                            'design':d,'seed':int(s),'lambda':r['lambda'] if rule=='lambda_rows' else 'selected','alpha':alpha,
                            'failure':int(failure(r,key,alpha)),'target':int(r['target_achieved']),'changed_to_target':int(r['changed_to_target']),
                            'detected':int(a[str(alpha)]),'clean_fpr':int(r['clean_audit'][key][str(alpha)]),'steps':r['steps'],
                            'clean_correct':int(r['clean_correct']),'clean_already_target':int(r['clean_pred']==r['target'])})
        agg={}
        for r in groupmetrics:
            key=tuple(r[k] for k in ['panel','split','rule','kind','epsilon','design','lambda','alpha']);agg.setdefault(key,{}).setdefault(r['group_id'],[]).append(r)
        summary=[]
        fields=['panel','split','rule','kind','epsilon','design','lambda','alpha']
        for key,groups in agg.items():
            result=dict(zip(fields,key))
            for metric in ['failure','target','changed_to_target','detected','clean_fpr','steps','clean_correct','clean_already_target']:
                values=[np.mean([r[metric] for r in rs]) for rs in groups.values()]
                estimate=ci(values);result.update({metric+'_'+k:v for k,v in estimate.items() if k!='groups'})
            result['groups']=len(groups);summary.append(result)
        for result in summary:
            key=tuple(result[f] for f in fields);groups=agg[key]
            target_mass=sum(np.mean([r['target'] for r in rs]) for rs in groups.values())
            detected_targets=sum(np.mean([r['target']*r['detected'] for r in rs]) for rs in groups.values())
            result['detected_given_target']=detected_targets/target_mass if target_mass else None
        write_csv(dest/'robustness_summary.csv',summary);write_csv(dest/'group_metrics.csv.gz',groupmetrics)
        # QA uses all val200 and the one clean evaluation per test group/seed.
        qa=[]
        for d in c['designs']:
            for s in c['reader_seeds']:
                rs=json.loads((root/c['calibration_folder']/f'{d}_{s}_valQA.json').read_text())['rows']
                qa.extend({**r,'design':d,'seed':s,'split':'reader_val'} for r in rs)
        dedup={}
        for r in raw:
            if r['split']!='test' or r['panel']!='main':continue
            for key,tasks in r['qa'].items():
                d,s=key.split(':')
                for t,correct in tasks.items():dedup[r['group_id'],d,int(s),t]={'group_id':r['group_id'],'design':d,'seed':int(s),'task':t,'accuracy':float(np.mean(correct)),'split':'test'}
        qa+=list(dedup.values());qs=[]
        for split in ['reader_val','test']:
            for d in c['designs']:
                for t in ['A1','A4']:
                    groups={}
                    for r in qa:
                        if (r['split'],r['design'],r['task'])==(split,d,t):groups.setdefault(r['group_id'],[]).append(r['accuracy'])
                    if groups:qs.append({'split':split,'design':d,'task':t,**ci([np.mean(x) for x in groups.values()])})
        write_csv(dest/'qa_summary.csv',qs);write_csv(dest/'qa_groups.csv.gz',qa)
        minimum=[];ratios=[]
        for spec in c.get('minimum_specs',[]):
            refs=sorted(root.glob(spec['pattern']))
            if not refs and spec.get('optional',False):continue
            assert len(refs)==spec['parts']
            groups={}
            for p in refs:
                ref=json.loads(p.read_text());met=json.loads((Path(ref['run'])/'metrics.json').read_text())
                assert met['status']=='completed' or spec.get('allow_partial',False)
                for r in map(json.loads,(Path(ref['run'])/'results.jsonl').read_text().splitlines()):
                    r['split']=spec['split'];minimum.append(r);assert (r['design'],r['seed']) not in groups.setdefault(r['group_id'],{})
                    groups[r['group_id']][r['design'],r['seed']]=r
            if not spec.get('allow_partial',False):assert len(groups)==spec['groups']
            else:assert len(groups)<=spec['groups']
            for gid,rs in groups.items():
                expected={('receiver',None)}|{(d,s) for d in c['designs'] for s in c['reader_seeds']}
                if set(rs)!=expected and spec.get('allow_partial',False):continue
                assert set(rs)==expected
                for (d,s),r in rs.items():
                    if d=='receiver':continue
                    interval=ratio_interval(rs['receiver',None],r)
                    ratios.append({'split':spec['split'],'group_id':gid,'design':d,'seed':s,**(interval or {'status':'already_target'}),
                                   'independent_panel_accepts':None if not r.get('best') else not r['best']['report_detected_5']})
        write_json(dest/'minimum_rows.json',minimum);write_csv(dest/'minimum_ratios.csv',ratios)
        ms=[]
        for split in sorted({r['split'] for r in ratios}):
            for d in c['designs']:
                rr=[r for r in ratios if r['split']==split and r['design']==d];finite=[r['estimate'] for r in rr if r['status']=='finite_search']
                ms.append({'split':split,'design':d,'pairs':len(rr),'finite_pairs':len(finite),'reader_censored':sum(r['status']=='reader_censored' for r in rr),
                    'receiver_censored':sum(r['status']=='receiver_censored' for r in rr),'already_target':sum(r['status']=='already_target' for r in rr),
                    **{f'q{q}':float(np.quantile(finite,q/100)) if finite else None for q in [10,25,50,75,90]},
                    'finite_independent_acceptance':float(np.mean([r['independent_panel_accepts'] for r in rr if r['status']=='finite_search'])) if finite else None,
                    'planned_groups':100,'complete_groups':len({r['group_id'] for r in rr})})
        write_csv(dest/'minimum_summary.csv',ms)
        costs=[]
        for p in Path('runs').glob('*/config.json'):
            cfg=json.loads(p.read_text())
            if cfg.get('milestone')!='M8c':continue
            q=p.parent/'metrics.json'
            if q.exists():costs.append({'run':str(p.parent),**json.loads(q.read_text())})
        write_csv(dest/'costs.csv',costs);write_json(dest/'matrix_verification.json',matrix)
        # Standard standalone scientific figures.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5,3.3))
        for d in c['designs']:
            rr=sorted([r for r in summary if r['panel']=='main' and r['split']=='test' and r['kind']=='E3' and r['alpha']==.05 and r['rule']=='selected' and r['design']==d],key=lambda r:r['epsilon'])
            if rr:ax.errorbar([r['epsilon'] for r in rr],[100*r['failure_mean'] for r in rr],yerr=[[100*(r['failure_mean']-r['failure_low']) for r in rr],[100*(r['failure_high']-r['failure_mean']) for r in rr]],marker='o',capsize=3,label=d)
        ax.set(xscale='log',xlabel='Per-layer relative KV radius ε',ylabel='Robust failure (%)',ylim=(0,100));ax.legend();fig.tight_layout();fig.savefig(dest/'epsilon_failure.pdf');fig.savefig(dest/'epsilon_failure.png',dpi=170);plt.close(fig)
        secondary=json.loads((root/'secondary_status.json').read_text()) if (root/'secondary_status.json').exists() else {}
        write_json(dest/'summary.json',{'run':str(run.path),'gpu_hours':total_cost(),'robustness':summary,'QA':qs,'minimum':ms,'matrix':matrix,'secondary_status':secondary})
        cost.update(status='completed');print(dest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

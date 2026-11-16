"""YAML-driven artificial val-proxy statistics fixture, with zero model calls."""
import sys
sys.dont_write_bytecode=True
import argparse,time
import numpy as np
import pandas as pd
from clr.utils.records import recorded
from clr.utils.config import load_config
from clr.audit.m71_runtime import snapshot
from scripts.m71_report import summarize_detection,plot_detection


def main(c):
    with recorded(c) as run:
        snapshot(run);start=time.monotonic();rng=np.random.default_rng(c['seed']);rows=[]
        for kind in ['CLR','B3','norm','text','supervised']:
            for seed in ([11,22,33] if kind in ['CLR','B3'] else [0]):
                for k in ([1,4,16] if kind in ['CLR','B3'] else [0]):
                    for alpha in [.01,.05]:
                        for i in range(c['proxy_groups']):
                            for category in ['clean','counterfactual']:
                                score=rng.random()
                                rows.append(dict(split='val_proxy',kind=kind,seed=seed,k=k,alpha=alpha,
                                    group_id=f'val-proxy-{i}',category=category,score=score,rejected=float(score>.9)))
        summary,_=summarize_detection(pd.DataFrame(rows),c,{},'val_proxy')
        frame=pd.DataFrame(summary);frame.to_csv(run.path/'statistics.csv',index=False)
        plot_detection(frame,run.path,split='val_proxy',panel='val_proxy')
        seconds=time.monotonic()-start
        if seconds>900:raise RuntimeError('CPU statistics smoke exceeded 15 minutes')
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=False,
                          seconds=seconds,artificial_val_proxy=True,groups=c['proxy_groups'])
        print('artificial val proxy only',len(rows),len(summary),seconds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*')
    a=p.parse_args();main(load_config(a.config,a.overrides))

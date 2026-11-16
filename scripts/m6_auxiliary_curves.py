"""Complete C3 learning curves for Q1/Q2 from locked final statistics."""
import argparse
import time
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.m6_runtime import snapshot


def main(c):
    with recorded(c) as run:
        print(run.path,flush=True);start=time.monotonic();snapshot(run,c)
        pipeline=load_config(c['pipeline_config']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
        data=pd.read_csv(c['metrics'],dtype={'n_labels':str})
        counts={s:pq.read_table(Path(spec['qa'])/'reader_train.parquet',columns=['is_heldout_template'],filters=[('is_heldout_template','=',False)]).num_rows for s,spec in pipeline['sources'].items()}
        figure_index=[]
        for split in ['reader_val','test']:
            fig,axes=plt.subplots(2,2,figsize=(11,7))
            for row,(donor,target) in enumerate([('gpt2','llama'),('llama','gpt2')]):
                for col,(q,metric) in enumerate([('Q1','brier'),('Q2','accuracy')]):
                    ax=axes[row,col]
                    for family,label in [('transfer','frozen-body CLR'),('lite','scratch CLR'),('b3','scratch B3'),('b5b','frozen Qwen mapping')]:
                        g=data[(data.source==target)&(data.split==split)&(data.qtype==q)&(data.kind=='clean')&(data.metric==metric)&(data.family==family)]
                        if family=='transfer':g=g[g.donor==donor]
                        elif family in ['lite','b3']:g=g[(g.donor==donor)|(g['case']==f'single_{target}_{family}')]
                        if g.empty:continue
                        g=g.copy();g['labels']=g.n_labels.map(lambda n:counts[target] if n=='full' else int(n));g=g.sort_values('labels')
                        ax.errorbar(g['labels'],g['mean'],yerr=[g['mean']-g.ci_low,g.ci_high-g['mean']],marker='o',label=label)
                    ax.set(xscale='log',xlabel='Target training QA labels N',ylabel='Q1 Brier (lower is better)' if q=='Q1' else 'Q2 accuracy',title=f'{donor} → {target} / {split}')
                    ax.set_xticks([100,300,1000,3000,counts[target]],['100','300','1k','3k','full']);ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle('Q1: target-source V (GPT2 sigma=1.5; LLaMA sigma=2, high label degeneracy)',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.96))
            stem=f'learning_curves_Q1_Q2_{split}'
            for suffix in ['png','pdf']:fig.savefig(out/f'{stem}.{suffix}',dpi=160)
            plt.close(fig)
            figure_index.append({'split':split,'qtypes':'Q1,Q2','png':str(out/f'{stem}.png'),'pdf':str(out/f'{stem}.pdf'),'source_metrics':c['metrics'],'CI':'10000 problem cluster bootstrap; fixed seed mean','Q1_target':'source-specific V; LLaMA high label degeneracy; no common cross-source V'})
        pd.DataFrame(figure_index).to_csv(out/'auxiliary_learning_curve_index.csv',index=False)
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start,figures=len(figure_index))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

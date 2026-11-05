"""Synthetic files only: exercise report formatting without benchmark test data."""
import json
from pathlib import Path
from contextlib import contextmanager
import torch  # Load the project C++ runtime before pyarrow/Pillow.
import pandas as pd
from scripts import m41_report as report

def test_review_generation_on_synthetic_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);Path('reports').mkdir();root=Path('data');root.mkdir();suite={}
    cases=['main_v1','main_lite','main_b3_multi','B0','B3']+[f'{p}_{n}'+('_v1' if p=='transfer' else '') for p in ['transfer','scratch_b3'] for n in [100,1000,10000]]
    for case in cases:
        for seed in [11,22,33]:
            p=tmp_path/f'{case}_{seed}';p.mkdir();suite[f'{case}_{seed}']=str(p)
            (p/'metrics.json').write_text(json.dumps({'status':'completed'}));stats=[];cp=[]
            for split in ['reader_val','test']:
                for q in ['Q1','Q2','Q3','Q4']:
                    for t in range(7):
                        for cal in ['temperature','affine']:
                            common=dict(case=case,seed=seed,split=split,qtype=q,t=t,heldout=False,calibration=cal,subset='all')
                            stats.append({**common,'kind':'clean','nll':.5,'brier':.2,'mae':.2,'accuracy':.7,'auroc':.7,'ece15':.03})
                            cp.append({**common,'alpha':.1,'coverage':.91,'size':1.5})
            (p/'statistics.json').write_text(json.dumps(stats));(p/'conformal.json').write_text(json.dumps(cp))
    (root/'training_complete.json').write_text(json.dumps({'runs':{}}))
    (root/'evaluation_complete.json').write_text(json.dumps({'runs':suite}));(root/'variant_selection.json').write_text(json.dumps({'selected':'v1'}));(root/'root_cause.md').write_text('仅为格式测试的合成根因占位内容。'*30)
    pd.DataFrame([{'step':0,'module':m,'grad_norm':1.} for m in ['adapter','cross','numeric','heads']]).to_csv('reports/M41_module_trace.csv',index=False)
    @contextmanager
    def recorded(c):
        class Run:
            path=tmp_path/'report_run'
            def write_metrics(self,**values):self.values=values
        r=Run();r.path.mkdir();yield r
    monkeypatch.setattr(report,'recorded',recorded);monkeypatch.setattr(report,'hours_used',lambda:1.)
    def paired(suite,a,b,seeds,split,q,metric,rng):return dict(a=a,b=b,split=split,qtype=q,metric=metric,delta=0.,ci_low=-.01,ci_high=.01)
    def cp(c,suite,a,b,split,q,rng):return [dict(a=a,b=b,split=split,qtype=q,metric=m,delta=0.,ci_low=-.01,ci_high=.01) for m in ['coverage','size']]
    monkeypatch.setattr(report,'paired',paired);monkeypatch.setattr(report,'cp_paired',cp)
    report.main({'output':str(root),'seeds':[11,22,33],'total_budget_hours':50})
    text=Path('reports/M4.1_REVIEW.md').read_text();assert len(text.encode())<=8000
    assert all(f'## ({x})' in text for x in 'abcdef')
    assert Path('reports/M41_figures/transfer.png').exists()

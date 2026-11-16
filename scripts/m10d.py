"""YAML-only M10d smoke/train/lock/final pipeline. No source model calls."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from clr.baselines.data_m23 import Questions
from clr.baselines.m10d import scalar_model, equal_step_statistics
from clr.calib.m23 import metrics, temperature
from clr.data.m6 import Bank
from clr.data.questions.core import seed
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything, write_json
from clr.utils.records import recorded
from scripts.m23_baselines import batches, predict


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(c):
    root = Path(c['output']); root.mkdir(parents=True, exist_ok=True)
    with recorded(c) as run:
        start = time.monotonic(); torch.set_num_threads(2)
        print(run.path, flush=True)
        templates = load_config(c['templates'])
        def check():
            elapsed = (time.monotonic()-start)/3600
            old = sum(json.loads(p.read_text())['gpu_hours'] for p in root.glob('cost_*.json'))
            if old + elapsed > c['gpu_budget_hours']:
                raise TimeoutError('M10d hard total GPU budget')
            if c['stage'] == 'smoke' and elapsed > .25:
                raise TimeoutError('M10d smoke exceeds 15min')
        def load(split, phase, subset=False):
            path = Path(c['qa']) / f'{split}.parquet'
            ids = set(pq.read_table(path, columns=['problem_id']).column(0).to_pylist())
            if subset:
                ids = set(sorted(ids)[:c['smoke_problems']])
            bank = Bank(c['root'], split, phase, ids)
            return bank, ids
        def ds(split, q, bank, ids, part=None):
            return Questions(Path(c['qa'])/f'{split}.parquet', q, bank,
                             templates, problem_ids=ids, cal_part=part)
        try:
            if c['stage'] in ['smoke', 'train']:
                small = c['stage'] == 'smoke'
                if not small:
                    gate=json.loads((root/'smoke.json').read_text())
                    assert json.loads(Path(gate['run'],'metrics.json').read_text())['status']=='completed'
                tb, ti = load('reader_train','train',small)
                vb, vi = load('reader_val','validate',small)
                assert ti.isdisjoint(vi)
                mean=tb.z.mean(0); scale=tb.z.std(0,correction=0).clamp_min(.01)
                np.savez(run.path/'normalization.npz',mean=mean.numpy(),scale=scale.numpy())
                tb.z=((tb.z-mean)/scale).to(c['baseline_device'])
                vb.z=((vb.z-mean)/scale).to(c['baseline_device'])
                tr=ds('reader_train','Q1',tb,ti); va=ds('reader_val','Q1',vb,vi)
                assert all(r['label_soft'] in [0.,1.] for r in tr.rows)
                assert len(ti)==(c['smoke_problems'] if small else 6817)
                write_json(run.path/'data_audit.json',dict(train_problems=len(ti),val_problems=len(vi),train_rows=len(tr.rows),val_rows=len(va.rows),test_opened=False,labels='unaveraged trajectory Bernoulli; frozen M6 start/observed-state rules'))
                selection=[]; locked=[]
                for rs in ([11] if small else c['seeds']):
                    best=None; bestloss=float('inf'); selected=None
                    for lr in c['learning_rates']:
                        seed_everything(rs); model=scalar_model(tb.dim).to(c['baseline_device'])
                        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=c['weight_decay'])
                        for epoch in range(1 if small else c['epochs']):
                            model.train(); rng=np.random.default_rng(seed(rs,epoch,'order'))
                            for _,(z,q,o,y) in batches(tr,tb,c['baseline_device'],c['batch_size'],rng.permutation(len(tr.rows))):
                                loss=-(y*F.log_softmax(model(z,q,o),-1)).sum(-1).mean()
                                opt.zero_grad(); loss.backward(); opt.step(); check()
                            vl=metrics(predict(model,va,vb,c),va.y)['nll']
                            if vl<bestloss:
                                bestloss=vl;best=copy.deepcopy(model.state_dict()); selected=dict(seed=rs,lr=lr,epoch=epoch+1,val_nll=vl)
                            print(rs,lr,epoch+1,vl,flush=True)
                    p=run.path/f'B4_Q1_{rs}.pt';torch.save(best,p)
                    locked.append(dict(seed=rs,path=str(p),sha256=digest(p)))
                    selection.append(selected);check()
                write_json(run.path/'selection.json',selection)
                result=dict(run=str(run.path),checkpoints=locked,normalization=str(run.path/'normalization.npz'),normalization_sha256=digest(run.path/'normalization.npz'),test_opened=False)
                if not small:
                    # Lock ALL three seeds together before any cal/test loading.
                    assert {a['seed'] for a in locked}==set(c['seeds'])
                    result['locked_before_test']=True
                write_json(root/('smoke.json' if small else 'training_lock.json'),result)
                if small:
                    model.load_state_dict(best)
                    for q in ['Q1','Q2']:
                        vq=ds('reader_val',q,vb,vi);x=predict(model,vq,vb,c)
                        assert np.isfinite(x).all()
                        assert equal_step_statistics(vq.rows,x,1.,q)
            elif c['stage']=='final':
                lock=json.loads((root/'training_lock.json').read_text())
                assert lock['locked_before_test'] and not lock['test_opened']
                assert len(lock['checkpoints'])==3
                for a in lock['checkpoints']:assert digest(a['path'])==a['sha256']
                assert digest(lock['normalization'])==lock['normalization_sha256']
                norm=np.load(lock['normalization']);allstats=[];calibration={}
                # cal_A only; Q1 and Q2 have separate fixed temperatures.
                cb,ci=load('cal','calibrate');cb.z=((cb.z-torch.from_numpy(norm['mean']))/torch.from_numpy(norm['scale'])).to(c['baseline_device'])
                for a in lock['checkpoints']:
                    model=scalar_model(cb.dim).to(c['baseline_device']);model.load_state_dict(torch.load(a['path'],weights_only=True,map_location=c['baseline_device']))
                    calibration[a['seed']]={}
                    for q in ['Q1','Q2']:
                        cq=ds('cal',q,cb,ci,'cal_A');calibration[a['seed']][q]=temperature(predict(model,cq,cb,c),cq.y);check()
                write_json(run.path/'calibration.json',calibration)
                # All training/checkpoint/temperature choices now fixed.
                for split in ['reader_val','test']:
                    bank,ids=load(split,'final');bank.z=((bank.z-torch.from_numpy(norm['mean']))/torch.from_numpy(norm['scale'])).to(c['baseline_device'])
                    for a in lock['checkpoints']:
                        model=scalar_model(bank.dim).to(c['baseline_device']);model.load_state_dict(torch.load(a['path'],weights_only=True,map_location=c['baseline_device']))
                        for q in ['Q1','Q2']:
                            data=ds(split,q,bank,ids);x=predict(model,data,bank,c);temp=calibration[a['seed']][q]
                            pq.write_table(pa.Table.from_pylist([dict(problem_id=r['problem_id'],state_ref=r['state_ref'],template_id=r['template_id'],replicate=r['replicate'],label=r['label'],label_soft=r['label_soft'],logits=x[i].tolist(),temperature=temp) for i,r in enumerate(data.rows)]),run.path/f'B4_{q}_{a["seed"]}_{split}.parquet',compression='zstd')
                            allstats.extend(dict(seed=a['seed'],split=split,**r) for r in equal_step_statistics(data.rows,x,temp,q));check()
                    del bank
                write_json(run.path/'statistics.json',allstats)
                write_json(root/'final.json',dict(run=str(run.path),statistics=str(run.path/'statistics.json')))
            else:raise ValueError(c['stage'])
            run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=(time.monotonic()-start)/3600)
        finally:
            elapsed=(time.monotonic()-start)/3600
            write_json(root/f'cost_{run.path.name}.json',dict(run=str(run.path),stage=c['stage'],gpu_hours=elapsed,source_forward_calls=0,gpu='RTX4090',device=c['baseline_device']))
            run.write_metrics(gpu_hours=elapsed,source_forward_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

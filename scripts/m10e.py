"""YAML-driven M10e development and hash-gated final evaluation."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from clr.baselines.data_m23 import Questions
from clr.baselines.m10e import HSRM, linear_probe, delta, centroid_score, fit_logistic, calibrated_logits
from clr.calib.m23 import metrics, temperature
from clr.data.m4.reader import PrefixBank, require_split
from clr.data.m6 import Bank
from clr.data.questions.core import seed
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything, write_json
from clr.utils.records import recorded
from scripts.m10d import digest


class Data:
    def __init__(self, c, split, phase, small=False):
        require_split(split, phase)
        self.c = c; self.split = split
        sc = c['sources'][c['source']]
        self.path = Path(sc['qa']) / f'{split}.parquet'
        ids = set(pq.read_table(self.path, columns=['problem_id']).column(0).to_pylist())
        self.total_problems = len(ids)
        if small:
            ids = set(sorted(ids)[:c['smoke_problems']])
            assert len(ids) / self.total_problems <= .05
        self.ids = ids
        if c['source'] == 'llama':
            self.bank = Bank(sc['root'], split, phase, ids)
        else:
            b = PrefixBank(dict(root=sc['root'], qa_root=sc['qa'], output=c['output']), split, phase, ids)
            b.pi = b.prefix_indices; b.mask = b.prefix_mask; b.dim = b.z.shape[1]
            self.bank = b
        self.templates = load_config(c['templates'])

    def questions(self, q, part=None):
        return Questions(self.path, q, self.bank, self.templates, cal_part=part, problem_ids=self.ids)


def logits(model, ds, bank, c, method):
    out = []; model.eval()
    with torch.no_grad():
        for _, args in batches(ds, bank, c, method):
            out.append(model(*args).cpu().numpy())
    return np.concatenate(out)


def batches(ds, bank, c, method, order=None):
    device = c['baseline_device']
    if order is None: order = np.arange(len(ds.rows))
    bs = c['batch_size'] if method == 'HSRM' else c['probe_batch_size']
    if method == 'B2' and not hasattr(ds, '_cache'):
        ds._cache = [torch.as_tensor(x, device=device) for x in [ds.q, ds.options]]
    for first in range(0, len(order), bs):
        ii = order[first:first+bs]; ix = torch.as_tensor(ds.state[ii], device=device)
        if method == 'HSRM':
            args = [bank.z[bank.pi[ix]], bank.mask[ix]]
        else:
            jj = torch.as_tensor(ii, device=device)
            args = [bank.z[ix]] + [x[jj] for x in ds._cache]
        yield ii, args


def to_device(bank, c):
    for name in ['z', 'pi', 'mask']:
        setattr(bank, name, getattr(bank, name).to(c['baseline_device']))


def clone_model(method, dim, q):
    return HSRM(dim) if method == 'HSRM' else linear_probe(dim, q)


def stats(rows, x, q):
    out = []
    for kind in ['clean', 'traj']:
        for t in sorted({r['state_ref']['t'] for r in rows}):
            ii = [i for i,r in enumerate(rows) if r['state_ref']['kind'] == kind and r['state_ref']['t'] == t]
            if not ii: continue
            if q == 'Q3': y = np.eye(4)[[rows[i]['label'] for i in ii]]
            else:
                y = np.array([[1-rows[i]['label_soft'],rows[i]['label_soft']] for i in ii])
            out.append(dict(kind=kind, t=t, n=len(ii), **metrics(x[ii], y, q=='Q3')))
    return out


def main(c):
    root = Path(c['output']); root.mkdir(parents=True, exist_ok=True)
    assert not (root/'freeze.json').exists(), 'M10e frozen; use a new authorized version'
    key = c['source']+'_'+c['method']
    stage = c['stage']
    with recorded(c) as run:
        print(run.path, flush=True)
        started = time.monotonic(); torch.set_num_threads(2)
        used_gpu = False
        def check():
            elapsed = (time.monotonic()-started)/3600
            if stage == 'smoke' and elapsed > .25: raise TimeoutError('M10e smoke >15min')
            old = sum(json.loads(p.read_text())['gpu_hours'] for p in root.glob('cost_*.json'))
            if old + (elapsed if used_gpu else 0) > c['gpu_budget_hours']:
                raise TimeoutError('M10e total GPU budget reached; explicit config extension required')
        def persist_cost():
            if used_gpu: torch.cuda.synchronize(c['baseline_device'])
            elapsed = (time.monotonic()-started)/3600
            cost = dict(run=str(run.path), key=key, stage=stage, wall_hours=elapsed,
                        gpu_hours=elapsed if used_gpu else 0., source_forward_calls=0,
                        device=c['baseline_device'] if used_gpu else 'cpu', gpu='RTX4090' if used_gpu else None)
            write_json(root/f'cost_{run.path.name}.json',cost)
            run.write_metrics(**cost)
        try:
            if stage in ['smoke','train']:
                small = stage == 'smoke'
                if not small:
                    gate = json.loads((root/f'smoke_{key}.json').read_text())
                    assert json.loads(Path(gate['run'],'metrics.json').read_text())['status'] == 'completed'
                trd = Data(c,'reader_train','train',small); vad = Data(c,'reader_val','validate',small)
                assert trd.ids.isdisjoint(vad.ids)
                trb,vab = trd.bank,vad.bank
                write_json(run.path/'data_audit.json',dict(train_problems=len(trd.ids),val_problems=len(vad.ids),
                    train_total=trd.total_problems,val_total=vad.total_problems,test_opened=False))
                selected=[]; checkpoints=[]; history=[]
                if c['method'] == 'CLUE':
                    ds = trd.questions('Q1'); endpoints={}
                    for r in ds.rows:
                        if r['state_ref']['t'] != 6: continue
                        k=(r['problem_id'],r['state_ref']['traj_id'])
                        y=r['label_soft']; assert y in (0.,1.)
                        if k in endpoints: assert endpoints[k][1] == y
                        endpoints[k]=(trb.index[k[0],k[1],6],y)
                    centroids=[];counts=[]
                    for y in [0,1]:
                        ix=[v[0] for v in endpoints.values() if v[1]==y]
                        assert ix
                        counts.append(len(ix)); sums=torch.zeros(trb.dim,dtype=torch.float64)
                        for first in range(0,len(ix),4096): sums += delta(trb,ix[first:first+4096]).double().sum(0)
                        centroids.append(sums/len(ix))
                    p=run.path/'centroids.pt';torch.save(torch.stack(centroids).float(),p)
                    checkpoints=[dict(seed=None,qtype='Q1',path=str(p),sha256=digest(p),counts=counts)]
                    # Validate inference on a val proxy, including t=0 zero deltas.
                    vd=vad.questions('Q1'); score=centroid_score(delta(vab,vd.state),torch.stack(centroids).float()).numpy()
                    assert np.isfinite(score).all(); assert np.isfinite(fit_logistic(score,vd.y[:,1])['slope'])
                else:
                    method=c['method'];used_gpu=True
                    if method=='B2':
                        mean=trb.z.mean(0);scale=trb.z.std(0,correction=0).clamp_min(.01)
                        np.savez(run.path/'normalization.npz',mean=mean.numpy(),scale=scale.numpy())
                        trb.z=(trb.z-mean)/scale;vab.z=(vab.z-mean)/scale
                    to_device(trb,c);to_device(vab,c)
                    for q in (['Q1'] if method=='HSRM' else ['Q1','Q2','Q3','Q4']):
                        tr=trd.questions(q);va=vad.questions(q)
                        if q=='Q1': assert all(r['label_soft'] in [0.,1.] for r in tr.rows)
                        for rs in ([11] if small else c['seeds']):
                            best=None; bestloss=float('inf'); choice=None
                            for lr in (c['learning_rates'] if method=='HSRM' else c['probe_learning_rates']):
                                seed_everything(rs);model=clone_model(method,trb.dim,q).to(c['baseline_device'])
                                opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=c['weight_decay'])
                                params=sum(p.numel() for p in model.parameters())
                                for epoch in range(1 if small else c['epochs']):
                                    model.train();begin=time.monotonic();n=0
                                    rng=np.random.default_rng(seed(rs,epoch,'order'))
                                    for ii,args in batches(tr,trb,c,method,rng.permutation(len(tr.rows))):
                                        y=torch.as_tensor(tr.y[ii],device=c['baseline_device'])
                                        loss=-(y*F.log_softmax(model(*args),-1)).sum(-1).mean()
                                        opt.zero_grad();loss.backward();opt.step();n+=len(ii);check()
                                    torch.cuda.synchronize(c['baseline_device']);seconds=time.monotonic()-begin
                                    vl=metrics(logits(model,va,vab,c,method),va.y,q=='Q3')['nll']
                                    record=dict(qtype=q,seed=rs,lr=lr,epoch=epoch+1,val_nll=vl,train_rows=n,train_seconds=seconds,parameters=params)
                                    history.append(record);write_json(run.path/'history.json',history)
                                    print(record,flush=True)
                                    if vl<bestloss:
                                        bestloss=vl;best=copy.deepcopy(model.state_dict());choice=record
                                    write_json(run.path/'progress.json',dict(key=key,**record,test_opened=False));check()
                            p=run.path/f'{method}_{q}_{rs}.pt';torch.save(best,p)
                            checkpoints.append(dict(seed=rs,qtype=q,path=str(p),sha256=digest(p)))
                            selected.append(choice)
                            # HSRM scalar also answers Q2; smoke must exercise that path.
                            if small and method=='HSRM':
                                model.load_state_dict(best);v2=vad.questions('Q2');xx=logits(model,v2,vab,c,method)
                                assert stats(v2.rows,xx,'Q2')
                        del tr,va
                    write_json(run.path/'selection.json',selected)
                result=dict(run=str(run.path),source=c['source'],method=c['method'],checkpoints=checkpoints,
                            normalization=str(run.path/'normalization.npz') if c['method']=='B2' else None,
                            config=str(run.path/'config.yaml'),test_opened=False)
                if result['normalization']: result['normalization_sha256']=digest(result['normalization'])
                write_json(root/f'{"smoke" if small else "trained"}_{key}.json',result)
            elif stage=='lock':
                expected=['gpt2_HSRM','llama_HSRM','gpt2_CLUE','llama_CLUE','llama_B2']
                models={}
                for k in expected:
                    d=json.loads((root/f'trained_{k}.json').read_text())
                    assert json.loads(Path(d['run'],'metrics.json').read_text())['status']=='completed'
                    assert not d['test_opened']
                    count=12 if d['method']=='B2' else (1 if d['method']=='CLUE' else 3)
                    assert len(d['checkpoints'])==count
                    for a in d['checkpoints']: assert digest(a['path'])==a['sha256']
                    d['config_sha256']=digest(d['config'])
                    d['code_sha256']={p:digest(p) for p in ['scripts/m10e.py','clr/baselines/m10e.py','configs/M10e/run.yaml',
                        c['templates'],'clr/baselines/probes.py','clr/baselines/data_m23.py','clr/data/m4/reader.py',
                        'clr/data/m6.py','clr/data/questions/core.py','clr/calib/m23.py','scripts/m10d.py']}
                    d['input_audit_sha256']=digest(root/'upstream_before.json')
                    models[k]=d
                write_json(root/'training_lock.json',dict(models=models,locked_before_test=True,test_opened=False,run=str(run.path)))
            elif stage=='final':
                lock=json.loads((root/'training_lock.json').read_text());assert lock['locked_before_test']
                d=lock['models'][key]
                for p,h in d['code_sha256'].items(): assert digest(p)==h,(p,'code changed after lock')
                assert digest(d['config'])==d['config_sha256']
                assert digest(root/'upstream_before.json')==d['input_audit_sha256']
                for a in d['checkpoints']: assert digest(a['path'])==a['sha256']
                method=c['method'];used_gpu=method!='CLUE';norm=None
                if d['normalization']:
                    assert digest(d['normalization'])==d['normalization_sha256']
                    norm=np.load(d['normalization'])
                def prepare(data):
                    if norm is not None:
                        data.bank.z=(data.bank.z-torch.from_numpy(norm['mean']))/torch.from_numpy(norm['scale'])
                    if method!='CLUE':to_device(data.bank,c)
                def predict(data,ds,a):
                    if method=='CLUE':
                        centers=torch.load(a['path'],map_location='cpu',weights_only=True);ss=[]
                        for first in range(0,len(ds.rows),4096):
                            ss.append(centroid_score(delta(data.bank,ds.state[first:first+4096]),centers).numpy())
                        return np.concatenate(ss)
                    model=clone_model(method,data.bank.dim,a['qtype']).to(c['baseline_device'])
                    model.load_state_dict(torch.load(a['path'],map_location=c['baseline_device'],weights_only=True))
                    return logits(model,ds,data.bank,c,method)
                calibration={};cal=Data(c,'cal','calibrate');prepare(cal)
                for a in d['checkpoints']:
                    for q in ([a['qtype']] if method=='B2' else ['Q1','Q2']):
                        ds=cal.questions(q,'cal_A');x=predict(cal,ds,a)
                        calibration[f"{a['seed']}_{q}"]=fit_logistic(x,ds.y[:,1]) if method=='CLUE' else temperature(x,ds.y)
                        check()
                write_json(run.path/'calibration.json',calibration)
                del cal
                allstats=[]
                write_json(run.path/'test_access.json',dict(phase='final',training_lock_sha256=digest(root/'training_lock.json'),temperatures_sha256=digest(run.path/'calibration.json')))
                for split in ['reader_val','test']:
                    data=Data(c,split,'final');prepare(data)
                    for a in d['checkpoints']:
                        for q in ([a['qtype']] if method=='B2' else ['Q1','Q2']):
                            ds=data.questions(q);x=predict(data,ds,a);fit=calibration[f"{a['seed']}_{q}"]
                            cx=calibrated_logits(x,fit) if method=='CLUE' else x/fit
                            pred=[dict(problem_id=r['problem_id'],state_ref=r['state_ref'],template_id=r['template_id'],replicate=r['replicate'],
                                       label=r['label'],label_soft=r['label_soft'],calibrated_logits=cx[i].tolist()) for i,r in enumerate(ds.rows)]
                            pq.write_table(pa.Table.from_pylist(pred),run.path/f"{method}_{q}_{a['seed']}_{split}.parquet",compression='zstd')
                            allstats.extend(dict(source=c['source'],method=method,seed=a['seed'],split=split,qtype=q,**r) for r in stats(ds.rows,cx,q))
                            check()
                    del data
                write_json(run.path/'statistics.json',allstats)
                write_json(root/f'final_{key}.json',dict(run=str(run.path),source=c['source'],method=method,statistics=str(run.path/'statistics.json')))
            else: raise ValueError(stage)
            run.write_metrics(status='completed',test_opened=stage=='final')
        finally:persist_cost()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

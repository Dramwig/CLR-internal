"""Read-only qualitative export. Selection never consults predictions.

Frozen scientific artifacts are hash-checked on every first read. New inference
is seed 11 only and uses the original calibration; it never calls fit/train/PGD.
"""
import gc
import hashlib
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from scipy.special import softmax
from safetensors import safe_open
from transformers import AutoTokenizer

from clr.calib.m4 import apply
from clr.data.m6 import Bank
from clr.data.m4.reader import probe_inputs
from clr.data.questions.core import number
from clr.models.clr_v0 import Serializer
from clr.models.m6 import NativeLite, NativeB3, MultiLite
from clr.models.m10 import AttentionProbe, OptionFeatures
from clr.baselines.probes import Probe
from clr.audit.m71_storage import load_bundle


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def rank_ids(ids, n):
    return sorted(set(ids), key=lambda p: (hashlib.sha256((p+'qual').encode()).hexdigest(), p))[:n]


def key(r):
    ref = r.get('state_ref', r)
    return (r['problem_id'], ref['t'], r['template_id'], tuple(r.get('options', [])))


def json_write(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


class Export:
    def __init__(self, c, run):
        self.c, self.run = c, run
        self.smoke = c['stage'] == 'smoke'
        self.dest = Path(c['output']) / ('smoke' if self.smoke else 'export')
        self.dest.mkdir(parents=True, exist_ok=True)
        self.split = 'reader_val' if self.smoke else 'test'
        self.frozen = {}; self.used = {}; self.anchors = {}; self.extra = []
        for p in sorted(Path('data').glob('*/freeze.json')):
            if 'M11_' in str(p): continue
            d = json.loads(p.read_text()); self.anchors[str(p)] = digest(p)
            for f, h in d.get('sha256', d.get('files', {})).items():
                if isinstance(h, str): self.frozen[str(Path(f).resolve())] = h
        m1 = json.loads(Path('reports/M1_completed_v3_manifest.json').read_text())
        self.anchors[m1['data_manifest']] = digest(m1['data_manifest'])
        for f, h in json.loads(Path(m1['data_manifest']).read_text()).items():
            self.frozen[str((Path('data/dumps/codi_gpt2_v3')/f).resolve())] = h
        self.registry = self.make_registry()
        self.templates = yaml.safe_load(self.read(c['templates']).read_text())
        self.serializer = Serializer(AutoTokenizer.from_pretrained(c['tokenizer'], local_files_only=True), self.templates)
        self.cache = {}; self.saved = []; self.selection = {}; self.inference_seconds = 0.
        self.sources = {'gpt2': ('data/dumps/codi_gpt2_v3', 'data/M23_v1a/full', 768),
                        'llama': ('data/dumps/codi_llama_M6_v1', 'data/M6_v1/qa', 2048)}

    def read(self, p):
        p = Path(p); s = str(p.resolve())
        if s not in self.used:
            actual = digest(p); expected = self.frozen.get(s)
            if expected is None: raise ValueError('Input absent from frozen manifest: '+str(p))
            if actual != expected: raise ValueError('Frozen input changed: '+str(p))
            self.used[s] = actual
        return p

    def js(self, p): return json.loads(self.read(p).read_text())

    def parquet(self, p, **kw): return pq.read_table(self.read(p), **kw).to_pylist()

    def make_registry(self):
        idx = self.js('data/M6_v1/evaluation_index.json'); joint = self.js('data/M9_v1/evaluation_index.json')
        old = self.js('data/M41_v1/evaluation_complete.json')['runs']
        a = self.js('data/M5b_v1/completion.json')['A']['runs']; result = []
        for source in ['gpt2', 'llama']:
            for seed in [11, 22, 33]:
                for kind, method in [('b3', 'mlp_probe'), ('lite', 'clr')]:
                    result.append(dict(source=source, seed=seed, method=method, run=idx[f'single_{source}_{kind}_{seed}'][source], layout='native'))
                result.append(dict(source=source, seed=seed, method='clr_joint', run=joint[f'multi_lite_{seed}'][source], layout='native'))
                for kind, method in [('B6','attention_probe'), ('B3-prefix','mlp_prefix'), ('CLR-lite-single','clr_single'), ('B0','question_only')]:
                    f = Path(f'data/M10_v1/evaluated/{source}_{kind}_{seed}_full.json')
                    if f.exists(): result.append(dict(source=source, seed=seed, method=method, run=self.js(f)['run'], layout='native'))
                    elif source == 'gpt2' and kind != 'B6':
                        result.append(dict(source=source, seed=seed, method=method, run=old[f'B0_{seed}'] if kind=='B0' else a[f'{kind}_{seed}'], layout='old'))
                if source == 'gpt2':
                    legacy = 'runs/20261103T045628621237Z_2964c31f2153_11a348'
                    for kind, method in [('B2','linear_probe'), ('B4','lto')]:
                        result.append(dict(source=source, seed=seed, method=method, run=legacy, layout='legacy', kind=kind))
                    ref = self.js(f'data/M10b_v1/evaluated/B5c_gpt2_{seed}_full.json')
                    result.append(dict(source=source, seed=seed, method='latentqa', run=ref['run'], layout='native'))
                else:
                    result.append(dict(source=source, seed=seed, method='linear_probe', run=self.js('data/M10e_v1/final_llama_B2.json')['run'], layout='external', kind='B2'))
                    result.append(dict(source=source, seed=seed, method='lto', run=self.js('data/M10d_v1/final.json')['run'], layout='external', kind='B4'))
                for kind, method in [('HSRM','hsrm'), ('CLUE','clue')]:
                    if kind=='CLUE' and seed!=11: continue
                    result.append(dict(source=source, seed=seed, method=method, run=self.js(f'data/M10e_v1/final_{source}_{kind}.json')['run'], layout='external', kind=kind))
        return result

    def path(self, reg, q):
        run = Path(reg['run']); seed=reg['seed']; split=self.split
        if reg['layout']=='native': return run/f'{reg["source"]}_{q}_{split}.parquet'
        if reg['layout']=='old': return run/f'{q}_{split}_False.parquet'
        if reg['layout']=='legacy': return run/f'{reg["kind"]}_{q}_{seed}_{split}_False.parquet'
        return run/f'{reg["kind"]}_{q}_{"None" if reg["method"]=="clue" else seed}_{split}.parquet'

    def predictions(self, source, q, ids, qa):
        output = {}
        lookup = defaultdict(list)
        for r in qa:
            if r['qtype']==q and r['state_ref']['traj_id']==-1:
                lookup[r['problem_id'], r['state_ref']['t'], r['template_id'], r['label']].append(r)
        for reg in self.registry:
            if reg['source'] != source: continue
            p = self.path(reg,q)
            if not p.exists(): continue
            rs = self.parquet(p, filters=[('problem_id','in',list(ids))]); selected={}
            for row_index,r in enumerate(rs):
                ref = r.get('state_ref',r)
                if ref.get('kind','clean')!='clean' or ref.get('traj_id',-1)!=-1: continue
                if 'options' not in r:
                    match=lookup[r['problem_id'],ref['t'],r['template_id'],r['label']]
                    if not match: continue
                    opts={tuple(x['options']) for x in match}
                    if len(opts)!=1: raise ValueError('Ambiguous old options')
                    r={**r,'options':list(opts.pop())}
                k=key(r)
                if 'calibrated_logits' in r: logits=np.array(r['calibrated_logits'],np.float64)
                elif 'temperature' in r: logits=np.array(r['logits'],np.float64)/r['temperature']
                else:
                    cal=self.js(p.parent/'calibration.json')[q]['temperature']
                    logits=apply(np.array([r['logits']]),cal)[0]
                val=dict(probabilities=softmax(logits).tolist(), calibrated_logits=logits.tolist(),
                         origin='frozen_prediction', file=str(p), row_index_in_problem_filter=row_index,
                         raw_logits=r.get('logits'), template_id=r['template_id'])
                if k in selected and selected[k]['probabilities']!=val['probabilities']: raise ValueError('Ambiguous predictions')
                selected[k]=val
                self.saved.append(dict(source=source, method=reg['method'], seed=reg['seed'], qtype=q,
                                       problem_id=r['problem_id'], t=ref['t'], options=r['options'], **val))
            output[reg['method'],reg['seed']]=selected
        return output

    def norm(self, source):
        f = 'runs/20261103T045628621237Z_2964c31f2153_11a348/normalization.npz' if source=='gpt2' else 'data/M6_v1/llama_normalization.npz'
        d=np.load(self.read(f)); return [torch.from_numpy(d[k]).to(self.c['device']) for k in ['mean','scale']]

    def model(self, source, method, q):
        dim=self.sources[source][2]
        reg=next(r for r in self.registry if r['source']==source and r['method']==method and r['seed']==11)
        ec=yaml.safe_load(self.read(Path(reg['run'])/'config.yaml').read_text())
        if method in ['clr','mlp_probe']:
            m=NativeLite(dim) if method=='clr' else NativeB3(dim); checkpoint=ec['checkpoint']
        elif method=='clr_joint':
            m=MultiLite({'gpt2':768,'llama':2048});checkpoint=ec['checkpoint']
        elif method=='attention_probe':
            ref=self.js(f'data/M10_v1/trained/{source}_B6_11_full.json')
            m=AttentionProbe(dim,hidden=ref['config']['hidden']);checkpoint=ref['checkpoints'][q]['path']
        elif method=='latentqa':
            from clr.models.m10_b5c import B5c
            ref=self.js('data/M10b_v1/trained/B5c_gpt2_11_full.json')
            # The base decoder version is pinned in the frozen training ref.
            for f in Path(ref['config']['qwen_path']).glob('*.safetensors'):
                self.extra.append(dict(path=str(f),sha256=digest(f),role='pinned_pretrained_decoder'))
            m=B5c(ref['config']['qwen_path'],dim);checkpoint=ref['checkpoint']
        else: raise ValueError('New inference not allowed for '+method)
        state=torch.load(self.read(checkpoint),map_location='cpu',weights_only=True)
        if method=='latentqa': m.restore_trainable(state)
        else: m.load_state_dict(state)
        return m.to(self.c['device']).eval().requires_grad_(False),reg

    @torch.no_grad()
    def infer(self, source, method, q, rows, bank=None, prefix=None, mask=None, embeddings=False):
        if not rows: return [],{}
        started=time.monotonic();device=self.c['device'];m,reg=self.model(source,method,q)
        cal=self.js(Path(reg['run'])/'calibration.json')[q]['temperature'];norm=self.norm(source)
        pool=[];hidden=[];logits=[];option=None
        if method=='attention_probe': option=OptionFeatures(self.c['tokenizer'])
        bs=self.c['latentqa_batch'] if method=='latentqa' else self.c['batch']
        body=m.body if method=='clr_joint' else m
        captured={};hook=None
        if embeddings:
            module=body.norm if method in ['clr','clr_joint'] else m.models[q].encoder
            hook=module.register_forward_hook(lambda mod,args,out: captured.update(value=out.detach()))
        try:
            for start in range(0,len(rows),bs):
                rr=rows[start:start+bs]
                if bank is not None: zz,mm=bank.batch(rr,device)
                else: zz,mm=prefix[start:start+bs].to(device),mask[start:start+bs].to(device)
                with (nullcontext() if method=='mlp_probe' else torch.autocast('cuda',dtype=torch.bfloat16)):
                    if method=='mlp_probe': x=m(*probe_inputs(rr,zz,mm,self.templates,*norm),q)
                    elif method=='attention_probe': x=m(option.batch(rr,device),zz,mm,q)
                    elif method=='latentqa': x=m(rr,zz,mm,self.templates)
                    else:
                        text=self.serializer.batch(rr,device)
                        x=m(text,zz,mm,q,source) if method=='clr_joint' else m(text,zz,mm,q)
                logits.extend(x.float().cpu().numpy())
                if embeddings:
                    h=captured['value'].float()
                    if method=='mlp_probe': hidden.extend(torch.relu(h[:,0]).cpu().numpy())
                    else:
                        offset=1 if method=='clr_joint' else 0
                        latent=h[:,offset:offset+zz.shape[1]]
                        pool.extend(((latent*mm[:,:,None]).sum(1)/mm.sum(1,keepdim=True)).cpu().numpy())
                self.budget()
        finally:
            if hook: hook.remove()
        out=apply(np.array(logits,dtype=np.float64),cal);probs=softmax(out,axis=1)
        result=[dict(probabilities=p.tolist(),calibrated_logits=l.tolist(),origin='seed11_frozen_weight_inference',
                     file=None,template_id=r['template_id'],calibration_file=str(Path(reg['run'])/'calibration.json')) for r,p,l in zip(rows,probs,out)]
        self.inference_seconds+=time.monotonic()-started
        del m,body,option
        gc.collect();torch.cuda.empty_cache()
        return result,{'clr_pool':np.array(pool,np.float32),'mlp_hidden':np.array(hidden,np.float32)}

    def budget(self):
        if (time.monotonic()-self.started)/3600 > self.c['gpu_budget_hours']: raise TimeoutError('M11 GPU/wall budget')

    def bank(self, source, ids):
        root=self.sources[source][0]
        entries=self.parquet(Path(root)/self.split/'manifest.parquet',columns=['shard'],filters=[('problem_id','in',list(ids))])
        for f in {r['shard'] for r in entries}: self.read(Path(root)/self.split/f)
        return Bank(root,self.split,'validate' if self.smoke else 'final',set(ids))

    def metadata(self, source):
        root=self.sources[source][0]
        return self.parquet(Path(root)/self.split/'manifest.parquet', filters=[('kind','=','clean'),('t','=',0)])

    def qa(self, source, ids=None):
        filters=[('is_heldout_template','=',False)]
        if ids is not None: filters.append(('problem_id','in',list(ids)))
        return [r for r in self.parquet(Path(self.sources[source][1])/f'{self.split}.parquet',filters=filters) if r['state_ref']['traj_id']==-1]

    def conformal(self, source, method, seed, t):
        reg=next(r for r in self.registry if r['source']==source and r['method']==method and r['seed']==seed)
        p=Path(reg['run'])/'conformal.json'
        if not p.exists(): return None
        rows=self.js(p)
        rows=[r for r in rows if r['qtype']=='Q3' and r['t']==t and r['alpha']==.1 and not r['heldout']
              and r.get('calibration','temperature')=='temperature' and r.get('subset','all')=='all'
              and r.get('split',self.split)==self.split and r.get('seed',seed)==seed
              and (reg['layout']!='legacy' or r.get('baseline')==reg['kind'])]
        if not rows: return None
        assert len({r['threshold'] for r in rows})==1
        return dict(alpha=.1,threshold=rows[0]['threshold'],calibration_split='cal_B',file=str(p),rule='1-p_j <= threshold; null threshold means +infinity')

    def reading(self, source):
        meta=self.metadata(source); byid={r['problem_id']:r for r in meta}
        eligible=[p for p,r in byid.items() if not r['cot_parse_failed'] and len(r['intermediate_values'])>=3]
        ids=rank_ids(eligible,1 if self.smoke else self.c['reading_n']);assert len(ids)==(1 if self.smoke else 12)
        self.selection[source+'_reading']={'population':len(byid),'eligible':len(eligible),'ids':ids,'split':self.split}
        qa_all=self.qa(source);qa=[r for r in qa_all if r['problem_id'] in ids]
        bank=self.bank(source,ids);root=self.sources[source][0]
        early={(r['problem_id'],r['t']):r for r in self.parquet(Path(root)/self.split/'early_exit.parquet',filters=[('problem_id','in',ids)])}
        mcpath=Path(root)/self.split/'mc.parquet'
        mc={(r['problem_id'],r['t']):r for r in self.parquet(mcpath,filters=[('problem_id','in',ids),('policy','=','pi')])} if mcpath.exists() else {}
        pred={q:self.predictions(source,q,ids,qa) for q in ['Q1','Q2','Q3','Q4']}
        # Frozen Q4 reciprocal donors are real intermediate values of other
        # problems. Extend deterministically from that same semantic pool.
        donor=[]
        for pid in rank_ids(byid,len(byid)):
            r=byid[pid]
            if not r['cot_parse_failed']:
                donor.extend((pid,v) for v in sorted({number(v) for v in r['intermediate_values']}))
        rows4=[]; candidates={}; supplement={}
        for pid in ids:
            positive=list(dict.fromkeys(number(v) for v in byid[pid]['intermediate_values']))
            negative=sorted({r['options'][0] for r in qa if r['problem_id']==pid and r['qtype']=='Q4' and r['label']==0})
            original=list(negative);donors=[]
            for dp,v in donor:
                if len(negative)>=len(positive):break
                if dp!=pid and v not in positive and v not in negative:negative.append(v);donors.append({'problem_id':dp,'value':v})
            assert len(negative)==len(positive) and not set(positive)&set(negative)
            candidates[pid]=positive+negative;supplement[pid]=dict(frozen_negatives=original,additional_donors=donors)
            for t in range(7):
                for v in candidates[pid]: rows4.append(dict(problem_id=pid,state_ref=dict(kind='clean',traj_id=-1,t=t),qtype='Q4',template_id=0,options=[v],label=int(v in positive),label_soft=float(v in positive),replicate=0))
        self.selection[source+'_Q4']={'template_id':0,'supplement':supplement}
        required=['mlp_probe','attention_probe','clr','clr_joint']+(['latentqa'] if source=='gpt2' else [])
        for method in required:
            saved=pred['Q4'].setdefault((method,11),{})
            todo=[r for r in rows4 if key(r) not in saved]
            inferred,_=self.infer(source,method,'Q4',todo,bank)
            saved.update({key(r):v for r,v in zip(todo,inferred)})
        output=[]
        for pid in ids:
            r=byid[pid]
            item={k:r[k] for k in ['problem_id','question','gold_cot','gold_answer','intermediate_values','model_answer','correct','cot_parse_failed']}
            item['Q4_candidates']=[dict(value=v,label=int(v in {number(x) for x in r['intermediate_values']})) for v in candidates[pid]]
            item['steps']=[]
            for t in range(7):
                a=early[pid,t];m=mc.get((pid,t))
                # At t6 deterministic continuation equals the saved sigma0
                # final answer; explicitly analytic instead of inventing MC.
                step=dict(t=t,early_exit_answer=a['answer'],early_exit_correct=a['correct'],
                          K16_Vhat=m['value'] if m else float(r['correct']) if t==6 else None,
                          mc_origin='frozen_K16_pi' if m else 'analytic_terminal' if t==6 else 'unavailable',Q4=[],Q3={},Q1={},Q2={})
                for rr in [x for x in rows4 if x['problem_id']==pid and x['state_ref']['t']==t]:
                    methods={}
                    for (method,seed),values in pred['Q4'].items():
                        if key(rr) in values:methods.setdefault(method,{})[str(seed)]={**values[key(rr)],'P_yes':values[key(rr)]['probabilities'][1]}
                    step['Q4'].append(dict(value=rr['options'][0],label=rr['label'],template_id=0,methods=methods))
                for q in ['Q1','Q2','Q3']:
                    rr=sorted([x for x in qa if x['problem_id']==pid and x['qtype']==q and x['state_ref']['t']==t],key=lambda x:(x['template_id'],x['replicate']))
                    if not rr and self.smoke and q=='Q1':
                        rr=[dict(problem_id=pid,state_ref=dict(kind='clean',traj_id=-1,t=t),qtype=q,template_id=0,options=[],label=0,replicate=0)]
                    assert rr, (source,pid,t,q)
                    qr=rr[0];block=dict(template_id=qr['template_id'],options=qr['options'],label=qr['label'],methods={})
                    for (method,seed),values in pred[q].items():
                        if key(qr) not in values:continue
                        value=dict(values[key(qr)])
                        if q=='Q3':
                            cp=self.conformal(source,method,seed,t)
                            if cp is not None:
                                thr=np.inf if cp['threshold'] is None else cp['threshold']
                                value['conformal']={**cp,'set_indices':np.flatnonzero(1-np.array(value['probabilities'])<=thr).tolist()}
                        else:value['P_yes']=value['probabilities'][1]
                        block['methods'].setdefault(method,{})[str(seed)]=value
                    step[q]=block
                item['steps'].append(step)
            output.append(item)
        p=self.dest/source/'qual_reading.json';json_write(p,dict(README='Gold text is display-only, never reader input. Q3 label predicts source final answer, not gold; option indices are zero-based. Q4 uses fixed seen template 0. Each method maps seed strings to exact frozen probabilities or explicitly marked seed11 inference. K16 uses frozen pi; no rollouts added.',source=source,split=self.split,problems=output))
        return len(ids)

    def reasoning(self):
        source='gpt2';meta={r['problem_id']:r for r in self.metadata(source)}
        ids=rank_ids(meta,4 if self.smoke else self.c['reasoning_n']);bank=self.bank(source,ids);qa=self.qa(source,ids)
        self.selection['reasoning']={'ids':ids,'split':self.split,'t':[0,2,4,6]}
        root=Path(self.sources[source][0])/self.split
        mcpath=root/'mc.parquet'
        mc={(r['problem_id'],r['t']):r for r in self.parquet(mcpath,filters=[('problem_id','in',ids),('policy','=','pi')])} if mcpath.exists() else {}
        early={(r['problem_id'],r['t']):r for r in self.parquet(root/'early_exit.parquet',filters=[('problem_id','in',ids)])}
        rows=[]
        for pid in ids:
            for t in [0,2,4,6]:
                qr=sorted([r for r in qa if r['problem_id']==pid and r['qtype']=='Q1' and r['state_ref']['t']==t],key=lambda r:(r['template_id'],r['replicate']))
                if not qr and self.smoke:qr=[dict(problem_id=pid,state_ref=dict(kind='clean',traj_id=-1,t=t),qtype='Q1',template_id=0,options=[],replicate=0)]
                assert qr;rows.append(qr[0])
        probs={};features={}
        for method in ['clr','mlp_probe']:
            inferred,emb=self.infer(source,method,'Q1',rows,bank,embeddings=True)
            saved=self.predictions(source,'Q1',ids,qa)[method,11]
            probs[method]=np.array([saved[key(r)]['probabilities'][1] if key(r) in saved else np.nan for r in rows])
            features.update({k:v for k,v in emb.items() if v.size})
            features[method+'_inference_probability']=np.array([r['probabilities'][1] for r in inferred])
        data=dict(problem_id=np.array([r['problem_id'] for r in rows]),t=np.array([r['state_ref']['t'] for r in rows]),
                  template_id=np.array([r['template_id'] for r in rows]),z=np.stack([bank.z[bank.index[r['problem_id'],-1,r['state_ref']['t']]].numpy() for r in rows]).astype(np.float16),
                  final_correct=np.array([meta[r['problem_id']]['correct'] for r in rows]),
                  Vhat=np.array([mc[r['problem_id'],r['state_ref']['t']]['value'] if (r['problem_id'],r['state_ref']['t']) in mc else float(meta[r['problem_id']]['correct']) for r in rows]),
                  early_exit_correct=np.array([early[r['problem_id'],r['state_ref']['t']]['correct'] for r in rows]),
                  clr_Q1_probability=probs['clr'],mlp_Q1_probability=probs['mlp_probe'],**features)
        for k,v in data.items():assert len(v)==len(rows),(k,v.shape)
        np.savez_compressed(self.dest/'embed_reasoning.npz',**data)
        return len(rows)

    def siblings(self):
        run=Path(self.js('data/M5b_v1/bc_full_B_11.json')['run'])
        files=sorted(run.glob('B_states_*_11.pt'));byid={}
        for f in files:
            d=torch.load(self.read(f),map_location='cpu',weights_only=True);byid[d['problem_id']]=(f,d)
        ids=rank_ids(byid,1 if self.smoke else self.c['sibling_n'])
        self.selection['siblings']={'ids':ids,'split':'val300','observed_t':[1,3],'frozen_parent_t':[0,2],'seeds':[11,22,33]}
        rows=[];prefix=[];masks=[];labels=[];all_z=[]
        for seed in [11,22,33]:
            for pid in ids:
                f,d=byid[pid];fs=f.with_name(f.name.replace('_11.pt',f'_{seed}.pt'))
                dd=d if seed==11 else torch.load(self.read(fs),map_location='cpu',weights_only=True)
                result=fs.with_name(fs.name.replace('B_states_','B_').replace('.pt','.parquet'))
                records=self.parquet(result)
                for parent_t in [0,2]:
                    h=dd['history'][parent_t]
                    for child in range(17):
                        rr=next(r for r in records if r['t']==parent_t and r['child']==child)
                        assert h.shape==(17,parent_t+2,768),h.shape
                        labels.append(dict(problem_id=pid,seed=seed,t=parent_t+1,parent_t=parent_t,child=child,is_sigma0=child==0,Vhat=rr['Vhat'],clr_frozen_value=rr['CLR-lite'],mlp_frozen_value=rr['B3']))
                        all_z.append(h[child,-1].numpy().astype(np.float16))
                        if seed==11:
                            row=dict(problem_id=pid,state_ref=dict(kind='traj',traj_id=child,t=parent_t+1),qtype='Q1',template_id=0,options=[])
                            rows.append(row);z=torch.zeros(7,768);z[:h.shape[1]]=h[child].float();prefix.append(z);masks.append(torch.arange(7)<h.shape[1])
        # M5b saved ranking LOGITS, not probabilities. Preserve them verbatim
        # and apply the existing (same-weight) Q1 temperatures separately.
        temperatures={}
        for method in ['clr','mlp_probe']:
            for seed in [11,22,33]:
                reg=next(r for r in self.registry if r['source']=='gpt2' and r['method']==method and r['seed']==seed)
                temperatures[method,seed]=self.js(Path(reg['run'])/'calibration.json')['Q1']['temperature']
        def probability(rr,method):
            raw=np.array([[0.,r['clr_frozen_value' if method=='clr' else 'mlp_frozen_value']] for r in rr],np.float64)
            out=[]
            for r,x in zip(rr,raw):out.append(softmax(apply(x[None],temperatures[method,r['seed']])[0])[1])
            return np.array(out,np.float64)
        self.selection['siblings']['value_semantics']='Original M5b saved raw ranking logits retained as *_value; *_probability applies frozen M6 same-weight Q1 temperatures without fitting. Template0.'
        # Main seed11 embeddings and frozen ranking values are supplemented
        # by exactly the original three-seed candidate values, not averaged.
        seed11=[r for r in labels if r['seed']==11]
        np.savez_compressed(self.dest/'embed_siblings.npz',
            problem_id=np.array([r['problem_id'] for r in seed11]),t=np.array([r['t'] for r in seed11]),
            child=np.array([r['child'] for r in seed11]),is_sigma0=np.array([r['is_sigma0'] for r in seed11]),
            z=np.stack([z[r['t']].numpy() for z,r in zip(prefix,seed11)]).astype(np.float16),
            clr_Q1_value=np.array([r['clr_frozen_value'] for r in seed11]),mlp_Q1_value=np.array([r['mlp_frozen_value'] for r in seed11]),
            clr_Q1_probability=probability(seed11,'clr'),mlp_Q1_probability=probability(seed11,'mlp_probe'),
            Vhat=np.array([r['Vhat'] for r in seed11]),
            all_seed=np.array([r['seed'] for r in labels]),all_problem_id=np.array([r['problem_id'] for r in labels]),
            all_t=np.array([r['t'] for r in labels]),all_child=np.array([r['child'] for r in labels]),
            all_z=np.stack(all_z),all_clr_probability=probability(labels,'clr'),all_mlp_probability=probability(labels,'mlp_probe'),
            all_clr_value=np.array([r['clr_frozen_value'] for r in labels]),all_mlp_value=np.array([r['mlp_frozen_value'] for r in labels]),all_Vhat=np.array([r['Vhat'] for r in labels]))
        # Both requested reasoning parts are also available in the named file.
        combined=dict(np.load(self.dest/'embed_reasoning.npz',allow_pickle=False))
        sibling=dict(np.load(self.dest/'embed_siblings.npz',allow_pickle=False))
        combined.update({'sibling_'+k:v for k,v in sibling.items()})
        np.savez_compressed(self.dest/'embed_reasoning.npz',**combined)
        return len(seed11)

    @torch.no_grad()
    def audit(self):
        from types import SimpleNamespace
        from clr.audit.m71_reader import Bank as AuditBank, Serializer as AuditSerializer
        from clr.audit.features import FeatureSequence, probe_summary
        from clr.models.m71 import Reader
        from clr.data.m7 import qa_rows, TEAMS
        folder=Path('data/M71_v1/pool/source')/self.split
        entries=[json.loads(x) for x in self.read(folder/'manifest.jsonl').read_text().splitlines()]
        byid={e['group_id']:e for e in entries};ids=rank_ids(byid,2 if self.smoke else self.c['audit_n'])
        assert len(ids)==(2 if self.smoke else 600)
        self.selection['audit']={'ids':ids,'split':self.split,'layers':[7,16,25,34],'categories':['clean','counterfactual'],'A4_template_id':0}
        bank=SimpleNamespace(z={},pool={});rows=[];means=[];norms=[];labels=[]
        for gid in ids:
            e=byid[gid];bundle=load_bundle(self.read(e['file']))
            for bi,category in enumerate(['clean','counterfactual']):
                mid=[e['id'],e['cf_id']][bi];saved=bundle['messages'][bi];seq=FeatureSequence(**saved)
                bank.z[mid]=saved;bank.pool[mid]=probe_summary(seq).flatten().half()
                ll=[]
                for layer in e['layers']:
                    use=seq.mask&(seq.layers==layer)&(seq.kinds==0)
                    assert use.any();ll.append(seq.values[use].float().mean(0).numpy())
                means.append(ll);norms.append(e['raw_norms'][bi])
                source=dict(id=mid,e=e['e'][bi],paired_e=e['e'][1-bi],route=e['route'][bi],paired_route=e['route'][1-bi])
                rr=qa_rows(dict(id=e['id'],group_id=gid),source,e['commitment'],category,[0])
                r=next(r for r in rr if r['task']=='A4');rows.append(r)
                route='ABCD'.index(e['commitment'].split('Route ')[1][0])
                expected=e['T'][e['z_pub']].index(route)
                labels.append(dict(group_id=gid,message_id=mid,category=category,e=source['e'],route=source['route'],expected_e=expected,commitment=e['commitment']))
        serializer=AuditSerializer(self.serializer.tokenizer)
        device=self.c['device'];probs={};scores={};alarms={};hidden=[];options=[]
        for kind,method in [('CLR','clr'),('B3','mlp_probe')]:
            ref=self.js(f'data/M71_v1/pool/training/{kind}_11.json')
            model=Reader(kind);model.load_state_dict(torch.load(self.read(ref['checkpoint']),map_location='cpu',weights_only=True));model=model.to(device).eval().requires_grad_(False)
            captured={}
            module=model.body.norm if kind=='CLR' else model.heads['A4'][0]
            hook=module.register_forward_hook(lambda m,a,o: captured.update(value=o.detach()))
            start=time.monotonic()
            try:
                for first in range(0,len(rows),self.c['batch']):
                    rr=rows[first:first+self.c['batch']];text=serializer.batch(rr,device)
                    src=AuditBank.batch(bank,rr,device) if kind=='CLR' else torch.stack([bank.pool[r['message_id']] for r in rr]).to(device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(text,src,'A4')
                    h=captured['value'].float()
                    if kind=='CLR':
                        selected=h[torch.arange(len(rr),device=device)[:,None],text['markers']+6]
                        selected=selected.gather(1,text['restore'][:,:,None].expand(-1,-1,selected.shape[-1]))
                        options.extend(selected.cpu().numpy())
                    else:hidden.extend(torch.nn.functional.gelu(h).cpu().numpy())
                    self.budget()
            finally:hook.remove()
            self.inference_seconds+=time.monotonic()-start
            del model,src,out;gc.collect();torch.cuda.empty_cache()
            for seed in [11,22,33]:
                ref=self.js(f'data/M71_v1/pool/evaluated/{kind}_{seed}.json');run=Path(ref['run'])
                saved=load_bundle(self.read(run/f'{self.split}_A4_seen.pt.gz'))
                lookup={(r['message_id'],r['template_id']):saved['probs'][i].numpy() for i,r in enumerate(saved['rows']) if r['template_id']==0}
                probs[method+'_'+str(seed)]=np.stack([lookup[r['message_id'],0] for r in rows])
                if kind=='CLR':
                    cal=self.js(f'data/M71_v1/pool/calibration/CLR_{seed}.json')
                    raw=[json.loads(x) for x in self.read(run/f'{self.split}_scores.jsonl').read_text().splitlines()]
                    lookup={(r['group_id'],r['category']):r['16'] for r in raw}
                    ss=np.array([lookup[r['group_id'],r['category']] for r in labels])
                    scores[str(seed)]=ss
                    for alpha in ['0.01','0.05']:
                        thr=cal['thresholds']['16'][alpha];alarms[str(seed)+'_'+alpha]=ss>(np.inf if thr is None else thr)
        self.selection['audit']['consistency_question_rule']='clr.audit.m71_evaluation.consistency_rows -> frozen consistency_questions; seed=reader seed, k=16; score copied from frozen *_scores.jsonl'
        np.savez_compressed(self.dest/'embed_audit.npz',
            group_id=np.array([r['group_id'] for r in labels]),message_id=np.array([r['message_id'] for r in labels]),category=np.array([r['category'] for r in labels]),
            f_layers=np.array(means,np.float32),f_mean=np.array(means,np.float32).mean(1),layer_ids=np.array([7,16,25,34]),
            kv_mean_square=np.array(norms,np.float64),kv_rms=np.sqrt(np.array(norms,np.float64)),
            clr_option_representation=np.array(options,np.float32),mlp_hidden=np.array(hidden,np.float32),
            option_names=np.array(TEAMS),private_e=np.array([r['e'] for r in labels]),private_route=np.array([r['route'] for r in labels]),
            expected_e=np.array([r['expected_e'] for r in labels]),commitment=np.array([r['commitment'] for r in labels]),
            **{k+'_A4_probability':v for k,v in probs.items()},**{'Gamma_'+k:v for k,v in scores.items()},**{'alarm_'+k:v for k,v in alarms.items()})
        return len(labels)

    def dependent(self):
        # The original evaluation writes scores only. Never create new PGD
        # tensors or infer missing attacks. Selection is over the fixed panel.
        if self.smoke:
            self.selection['dependent']={'status':'not read in smoke; test results only'}
            return 0
        records=[];sources=[]
        for ref in sorted(Path('data/M8c_v1').glob('final_test*_part*.json')):
            d=self.js(ref);p=Path(d['run'])/'results.jsonl'
            if not p.exists(): continue
            sources.append(str(p))
            for line in self.read(p).read_text().splitlines():
                r=json.loads(line)
                if r['kind']=='E3' and r['epsilon']==.1:records.append({**r,'source_file':str(p)})
        ids=rank_ids({r['group_id'] for r in records},self.c['dependent_n'])
        assert len(ids)==100
        # Keep every saved seed/design/budget trial; do not select attacks by
        # success, audit evasion, or score. Deduplicate byte-identical restarts.
        chosen=[];seen=set()
        for r in records:
            if r['group_id'] not in ids:continue
            identity=json.dumps({k:v for k,v in r.items() if k!='source_file'},sort_keys=True)
            if identity in seen:continue
            seen.add(identity);chosen.append(r)
        for r in chosen:
            r['D0_audit']=r['audit'].get('D0:'+str(r['attack_seed']))
            r['R4_audit']=r['audit'].get('R4:'+str(r['attack_seed']))
            r['f_before']=r['f_after']=r['receiver_features_before']=r['receiver_features_after']=None
            r['missing_fields_reason']='Perturbed tensors and cross-design audit of this identical attack were not saved; no new PGD/receiver/auditor calls.'
        self.selection['dependent']={'ids':ids,'split':'fixed_test200','epsilon':.1,'kind':'E3','sources':sources,'all_saved_design_seed_step_trials':True,
            'perturbed_tensor_available':False,'reason':'Original frozen scripts/m8c_evaluate.py persists results.jsonl scores and deletes attack/final/feat; no perturbed f or receiver tensor was persisted.'}
        json_write(self.dest/'dependent_scores.json',dict(README='Existing M8c E3 epsilon=.1 white-box results only; no new PGD or receiver forward. Each row retains target/clean_pred/pred/changed_to_target and clean/post audit scores and alarms. audit keys identify only the attack-design auditor; unmeasured other auditors stay unavailable. No before/after tensors were saved.',selection=self.selection['dependent'],rows=chosen))
        return len(chosen)

    def finish(self, counts):
        self.selection.update(hash_rule='sort ascending sha256(UTF8(id + "qual")), tie-break raw ID; no outcome/probability inputs',
                              Q4_rule='all unique frozen gold intermediate values + equal negatives; original own frozen negative first, then other problems frozen gold values by donor hash and lexical numeric-string order; exclude all positives and duplicates',
                              Q3_rule='original frozen seen template and original options for each state; label is sigma0 final-source answer; no reordering',
                              representation_seed=11,existing_prediction_seeds=[11,22,33],no_training=True,no_recalibration=True,no_source_rollouts=True)
        json_write(self.dest/'selection.json',self.selection)
        json_write(self.dest/'frozen_predictions.json',dict(README='Long form exact saved scientific predictions for all selected reading/embedding problems, including original varying templates. Reconstruct probabilities with float64 scipy softmax on calibrated_logits; raw_logits copied when available. This preserves three-seed predictions even where fixed Q4 template0 requires new seed11 inference. Duplicate rows can occur because a problem participates in multiple export views.',rows=self.saved))
        # Reload the delivered JSON before equality checks, catching serialization.
        saved=json.loads((self.dest/'frozen_predictions.json').read_text())['rows']
        checks=[];reloaded={}
        for source in ['gpt2','llama']:
            ids=self.selection[source+'_reading']['ids'][:3]
            for pid in ids:
                values=[r for r in saved if r['source']==source and r['problem_id']==pid]
                n=0
                for r in values:
                    if r['file'] not in reloaded:reloaded[r['file']]=self.parquet(r['file'],filters=[('problem_id','in',self.selection[source+'_reading']['ids'])])
                    rows=reloaded[r['file']]
                    raw=rows[min(r['row_index_in_problem_filter'],len(rows)-1)]
                    # reasoning exports may use a wider filter; identify by
                    # exact metadata/logits if its ordinal no longer applies.
                    def same(x):
                        ref=x.get('state_ref',x)
                        return x['problem_id']==pid and ref['t']==r['t'] and x['template_id']==r['template_id'] and (x.get('raw_logits',x.get('logits'))==r['raw_logits'] if 'calibrated_logits' not in x else x['calibrated_logits']==r['calibrated_logits'])
                    if not same(raw):raw=next(x for x in rows if same(x))
                    if 'calibrated_logits' in raw:l=np.array(raw['calibrated_logits'],np.float64)
                    elif 'temperature' in raw:l=np.array(raw['logits'],np.float64)/raw['temperature']
                    else:l=apply(np.array([raw['logits']]),self.js(Path(r['file']).parent/'calibration.json')[r['qtype']]['temperature'])[0]
                    expected=softmax(l)
                    assert np.array_equal(expected.view(np.uint64),np.array(r['probabilities'],np.float64).view(np.uint64))
                    n+=1
                checks.append(dict(source=source,problem_id=pid,probability_rows=n,float64_bitwise_equal=True))
        used_after={p:digest(p) for p in self.used}
        assert used_after==self.used,'Frozen source mutated during export'
        json_write(self.dest/'checks.json',dict(README='Fixed first 3 selected problems per source (6 total); exported JSON is reloaded and every frozen probability row is compared bitwise against softmax of the original frozen prediction/calibration. This does not claim new inference is bitwise identical across batches.',sample_checks=checks,source_files_checked=len(self.used),frozen_inputs_unchanged=True,counts=counts,smoke=self.smoke))
        json_write(self.dest/'provenance.json',dict(README='Absolute input paths with SHA256, checked against original milestone manifests before reading and again after export. No frozen result was overwritten. pretrained_dependencies separately records pinned base model files.',source_sha256=self.used,freeze_manifest_sha256=self.anchors,pretrained_dependencies=self.extra,run=str(self.run.path),config=self.c))
        self.readmes()
        size=sum(p.stat().st_size for p in self.dest.rglob('*') if p.is_file())
        assert size <= self.c['max_bytes'],size
        return size

    def readmes(self):
        descriptions={
            'embed_reasoning.npz':'Main rows: 300 problems × t={0,2,4,6}; sibling_* arrays contain part (b), also available separately in embed_siblings.npz. z: original fp16 [N,768]. clr_pool: last normalized reader layer latent token mean over valid observed prefix, [N,256], in Q1 forward. mlp_hidden: Q1 probe ReLU hidden [N,128]. *_Q1_probability: exact frozen calibrated seed11 probabilities. *_inference_probability: paired representation-extraction forward, batch rounding may differ. Vhat: frozen pi K16 (t6 analytic if absent); final/early_exit correctness booleans. template_id is the original frozen Q1 seen template. No 2D reduction has been performed.',
            'embed_siblings.npz':'Main arrays: seed11, 40 val300 problems × observed t={1,3} ×17 children. z: actual observed post-forward child, fp16 [N,768]; child0 is sigma0. clr_Q1_value/mlp_Q1_value are exact original M5b ValueRunner raw ranking logits (NOT probabilities). *_Q1_probability applies the existing same-weight frozen M6 Q1 temperature, without refitting. Vhat is original child K16 oracle. all_* arrays retain independent seed11/22/33 states, raw values, probabilities and MC labels. Parent t={0,2}; observed t=parent+1. Different seeds have different perturbed candidates; never merge by child alone.',
            'embed_audit.npz':'Rows: 600 groups × clean/counterfactual. f_layers [N,4,512]: equal-weight record-segment mean (kind0, including public segment), excludes terminal-token extras. f_mean [N,512]: mean of the four layer means. layer_ids={7,16,25,34}. kv_mean_square [N,4,2]: original per-layer K/V valid-element mean square; kv_rms=sqrt thereof (not Frobenius norm). clr_option_representation [N,4,256]: final normalized Lite option-marker representations, restored to option_names order, A4 template0 seed11. mlp_hidden [N,256]: A4 pooled B3 GELU hidden. *_A4_probability: exact stored float32 probabilities for all three seeds. Gamma_seed: original mean negative log-probability of k16 fixed consistency questions, higher is less consistent. alarm_seed_alpha: Gamma exceeds frozen clean cal_B threshold, alpha=.01/.05; false is no alarm. private_e/route are actual message private values; expected_e is inverse public table applied to unchanged commitment. Options use TEAMS order. No audit score or threshold was refit.',
        }
        for p in list(self.dest.rglob('*.json'))+list(self.dest.glob('*.npz')):
            text=descriptions.get(p.name)
            if text is None:
                d=json.loads(p.read_text());text=d.get('README', {'selection.json':'Fixed result-independent selection populations, IDs, hash rule, template IDs and Q4 supplement provenance.', 'provenance.json':'Frozen input hashes and run/config provenance.', 'checks.json':'Consistency validation and export row counts.'}.get(p.name,'See fields and provenance.json.'))
            p.with_name(p.name+'.README.md').write_text(text+'\n')
        (self.dest/'README.md').write_text('M11-qual v1: raw data for reviewer plotting; no training, recalibration, dimensionality reduction or PGD.\n\nEach data file has a .README.md sidecar. Load NPZ with numpy.load(..., allow_pickle=False). JSON uses full float precision; option/set indices are zero-based. See selection.json for fixed selection and provenance.json/checks.json for evidence. Frozen predictions and newly inferred seed11 supplemental Q4 values are explicitly distinguished. Frozen scientific results remain unchanged. M8c only saved scores, so dependent_scores.json has no perturbation tensors.\n')

    def execute(self):
        self.started=time.monotonic()
        if not self.smoke:
            gate=json.loads((Path(self.c['output'])/'smoke_passed.json').read_text());assert gate['passed']
        counts={}
        for source in ['gpt2','llama']:
            print('reading',source,flush=True);counts[source+'_reading']=self.reading(source)
        print('reasoning',flush=True);counts['reasoning_states']=self.reasoning()
        print('siblings',flush=True);counts['sibling_states']=self.siblings()
        print('audit',flush=True);counts['audit_messages']=self.audit()
        print('dependent',flush=True);counts['dependent_trials']=self.dependent()
        print('checks',flush=True);size=self.finish(counts)
        return counts,size

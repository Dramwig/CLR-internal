"""M7 calibration and evaluation. Natural commitments are a separate, final-only analysis."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from clr.data.m7 import TEMPLATES,TEAMS,commitment_route,threshold,corrupt
from clr.audit.m7_reader import Bank,Serializer


def expected_private(entry,commitment=None):
    route=commitment_route(commitment or entry['commitments'][0])
    table=entry['T'][entry['z_pub']]
    if sorted(table)!=list(range(4)):raise ValueError('M7 audit requires the alpha=1 permutation table')
    return table.index(route)


def scalar(value):return None if not np.isfinite(value) else float(value)

def fit_temperature(logits,labels):
    logits=np.asarray(logits);labels=np.asarray(labels,dtype=int)
    def loss(logt):
        p=softmax(logits/math.exp(logt),axis=-1)
        return -np.log(p[np.arange(len(p)),labels].clip(1e-12)).mean()
    out=minimize_scalar(loss,bounds=(-3,3),method='bounded')
    return float(math.exp(out.x))


def ece(probs,labels):
    confidence=probs.max(1);correct=probs.argmax(1)==labels
    bins=np.minimum((confidence*15).astype(int),14)
    return sum(np.mean(bins==i)*abs(np.mean(confidence[bins==i])-np.mean(correct[bins==i]))
               for i in range(15) if np.any(bins==i))


def qa_summary(rows,probs):
    y=np.array([r['label'] for r in rows]);pred=probs.argmax(1)
    target=np.eye(probs.shape[1])[y]
    return {'n':len(rows),'groups':len({r['group_id'] for r in rows}),
            'accuracy':float(np.mean(pred==y)),'nll':float(-np.log(probs[np.arange(len(y)),y].clip(1e-12)).mean()),
            'brier':float(np.square(probs-target).sum(1).mean()),'ece':float(ece(probs,y)),
            'auroc':float(roc_auc_score(y,probs[:,1])) if probs.shape[1]==2 and len(set(y))==2 else None}


@torch.no_grad()
def predict(model,bank,rows,serializer,device,batch_size,*,states=None):
    result={}
    for task in ['A1','A2','A3','A4']:
        indices=[i for i,r in enumerate(rows) if r['task']==task]
        for start in range(0,len(indices),batch_size):
            ii=indices[start:start+batch_size];rs=[rows[i] for i in ii]
            if model.kind=='B0':
                z=torch.zeros(len(rs),1,2048,device=device);mask=torch.ones(len(rs),1,device=device,dtype=torch.bool)
            elif states is None:z,mask=bank.batch(rs,device)
            else:
                zz=[states[r['message_id']] for r in rs];width=max(map(len,zz))
                z=torch.zeros(len(zz),width,2048,dtype=torch.float16);mask=torch.zeros(len(zz),width,dtype=torch.bool)
                for j,x in enumerate(zz):z[j,:len(x)]=x;mask[j,:len(x)]=True
                z=z.to(device);mask=mask.to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(serializer.batch(rs,device),z,mask,task)
            for i,v in zip(ii,out.float().cpu().numpy()):result[i]=v
    return [result[i] for i in range(len(rows))]


def consistency_questions(entry,view_id,seed,k=16):
    # Uses only the public T, public z and visible commitment to specify expected
    # answers. It never consults e_priv or the source message's oracle labels.
    rng=np.random.default_rng(int(hashlib.sha256((entry['id']+str(seed)).encode()).hexdigest()[:16],16))
    expected_e=expected_private(entry);comm=entry['commitments'][0];route=commitment_route(comm)
    rows=[]
    for j in range(k):
        task=['A1','A2','A3','A4'][int(rng.integers(4))];tid=int(rng.integers(4))
        fact='';target=1
        if task=='A1':
            index=int(rng.integers(4));fact=TEAMS[index];target=int(index==expected_e)
        elif task=='A2':fact=f'Route {chr(65+route)}'
        elif task=='A4':target=expected_e
        q=TEMPLATES[task][tid].format(fact=fact,commitment=comm)
        rows.append({'episode_id':entry['id'],'group_id':entry['group_id'],'message_id':view_id,
                     'task':task,'template_id':tid,'question_text':q,'options':list(TEAMS) if task=='A4' else ['No','Yes'],
                     'label':target,'consistency_index':j})
    return rows


def views(bank,with_mismatch=True):
    mismatches={r['id']:r for r in map(json.loads,(Path(bank.entries[0]['file']).parent/'mismatch_v2.jsonl').open())} if with_mismatch else {}
    lookup={e['id']:e for e in bank.entries};out=[]
    for e in bank.entries:
        mid=e['id']
        items=[('clean',bank.z[mid],e['receiver_probs'][0]),
               ('counterfactual',bank.z[e['cf_id']],e['receiver_probs'][1])]
        if with_mismatch:
            donor=mismatches[mid]['donor_id']
            items.append(('mismatch',bank.z[donor],mismatches[mid]['receiver_probs']))
        for kind in ['truncate','zero_layer','noise']:
            if kind=='truncate':zz=bank.z[mid][:e['truncate_valid_tokens'][0]]
            else:zz=corrupt(bank.z[mid],kind,e['noise_seeds'][0])
            if kind=='noise':zz=zz.bfloat16().half() # Actual receiver casts the manipulated cache to bf16.
            items.append(('corrupted_'+kind,zz,e['corrupted_receiver_probs'][kind][0]))
        for category,z,receiver in items:
            out.append({'entry':e,'category':category,'z':z,'receiver':receiver,'view_id':mid+'::'+category})
    return out


def norm_features(z):
    half=z.shape[1]//2
    return np.array([math.log(max(float(z[:,:half].float().square().mean()),1e-12)),
                     math.log(max(float(z[:,half:].float().square().mean()),1e-12))])


def text_features(entry,receiver,commitment=None):
    a=commitment_route(commitment or entry['commitments'][0]);b=int(np.argmax(receiver))
    return np.r_[np.eye(4)[a],np.eye(4)[b],np.eye(16)[a*4+b],float(a==b)]


def fit_detectors(train,with_mismatch=True):
    vv=views(train,with_mismatch)
    normals=np.stack([norm_features(v['z']) for v in vv if v['category']=='clean'])
    mean=normals.mean(0);scale=normals.std(0).clip(1e-4)
    x=np.stack([text_features(v['entry'],v['receiver']) for v in vv]);y=np.array([v['category']!='clean' for v in vv],int)
    text=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=11).fit(x,y)
    return mean,scale,text


def score_detectors(bank,detectors):
    mean,scale,text=detectors;result=[]
    for v in views(bank):
        result.append({'id':v['entry']['id'],'group_id':v['entry']['group_id'],'category':v['category'],
                       'norm':float(np.square((norm_features(v['z'])-mean)/scale).sum()),
                       'text':float(text.predict_proba(text_features(v['entry'],v['receiver'])[None])[0,1])})
    return result


def score_reader(model,bank,serializer,c,temperatures,seed):
    vv=views(bank);states={v['view_id']:v['z'] for v in vv};rows=[]
    for v in vv:rows+=consistency_questions(v['entry'],v['view_id'],seed)
    raw=predict(model,bank,rows,serializer,c['device'],c['batch_size'],states=states)
    allp=[softmax(x/temperatures[r['task']]) for x,r in zip(raw,rows)]
    output=[]
    for vi,v in enumerate(vv):
        rr=rows[vi*16:(vi+1)*16];pp=allp[vi*16:(vi+1)*16]
        logp=np.array([math.log(max(float(p[r['label']]),1e-12)) for r,p in zip(rr,pp)])
        output.append({'id':v['entry']['id'],'group_id':v['entry']['group_id'],'category':v['category'],
                       **{str(k):float(-logp[:k].mean()) for k in [1,4,16]}})
    return output


def qa_calibration(model,bank,serializer,c,mixed=False):
    # Exactly one seen-template clean instance per group and task.
    selected=[]
    for task in ['A1','A2','A3','A4']:
        for e in bank.entries:
            category='counterfactual' if mixed and int(hashlib.sha256((e['id']+task+'category').encode()).hexdigest()[:8],16)%2 else 'clean'
            rows=[r for r in bank.rows if r['episode_id']==e['id'] and r['task']==task and r['category']==category]
            selected.append(rows[int(hashlib.sha256((e['id']+task).encode()).hexdigest()[:8],16)%len(rows)])
    return selected,predict(model,bank,selected,serializer,c['device'],c['batch_size'])

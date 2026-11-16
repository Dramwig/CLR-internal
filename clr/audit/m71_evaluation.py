"""Feature-only inference, fixed random questions and clean calibration."""
import hashlib
import numpy as np
import torch
from scipy.special import softmax
from clr.audit.m7_evaluation import consistency_questions,fit_temperature,qa_summary
from clr.data.m7 import threshold


def views(bank):
    out=[]
    for index,entry in enumerate(bank.entries):
        donor=bank.entries[(index+1)%len(bank.entries)]
        if donor['group_id']==entry['group_id']:raise ValueError('Mismatch group overlap')
        out.extend({'entry':entry,'category':category,'message_id':mid} for category,mid in
                   [('clean',entry['id']),('counterfactual',entry['cf_id']),('mismatch',donor['id'])])
        for kind in ['truncate','zero_layer','noise']:
            mid=entry['id']+'::corrupted_'+kind
            if mid in bank.controls:out.append({'entry':entry,'category':'corrupted_'+kind,'message_id':mid})
    return out


def as_m7_entry(entry):
    return {**entry,'commitments':[entry['commitment']]}


def consistency_rows(view,seed,k=16):
    return consistency_questions(as_m7_entry(view['entry']),view['message_id'],seed,k)


@torch.no_grad()
def predict(model,bank,rows,serializer,config):
    from scripts.m71_train import source_for,text_for
    result={}
    model.eval()
    for task in ['A1','A2','A3','A4']:
        indices=[i for i,r in enumerate(rows) if r['task']==task]
        for start in range(0,len(indices),config['batch_size']):
            ii=indices[start:start+config['batch_size']];rr=[rows[i] for i in ii]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(text_for(model,serializer,rr,config['device']),source_for(model,bank,rr,config['device']),task)
            for i,value in zip(ii,out.float().cpu().numpy()):result[i]=value
    return [result[i] for i in range(len(rows))]


def calibration_rows(bank,mixed=True):
    rows=[]
    for entry in bank.entries:
        for task in ['A1','A2','A3','A4']:
            h=hashlib.sha256((entry['id']+task).encode()).hexdigest()
            category=['clean','counterfactual'][int(h[:8],16)%2] if mixed else 'clean'
            selected=[r for r in bank.rows if r['episode_id']==entry['id'] and r['category']==category and r['task']==task]
            rows.append(selected[int(h[8:16],16)%len(selected)])
    return rows


def temperatures(model,bank,serializer,config):
    rows=calibration_rows(bank);raw=predict(model,bank,rows,serializer,config)
    return {task:fit_temperature([x for r,x in zip(rows,raw) if r['task']==task],
                                  [r['label'] for r in rows if r['task']==task])
            for task in ['A1','A2','A3','A4']}


def score_reader(model,bank,serializer,config,temp,seed):
    vv=views(bank);result=[]
    # Bound CPU memory while keeping each message's random question set fixed.
    for start in range(0,len(vv),128):
        chunk=vv[start:start+128];rows=[r for v in chunk for r in consistency_rows(v,seed)]
        raw=predict(model,bank,rows,serializer,config)
        p=[softmax(x/temp[r['task']]) for x,r in zip(raw,rows)]
        for i,v in enumerate(chunk):
            rr=rows[i*16:(i+1)*16];pp=p[i*16:(i+1)*16]
            logp=np.array([-np.log(max(float(x[r['label']]),1e-12)) for r,x in zip(rr,pp)])
            result.append({'id':v['entry']['id'],'group_id':v['entry']['group_id'],'category':v['category'],
                           **{str(k):float(logp[:k].mean()) for k in [1,4,16]}})
    return result


def norm_vector(values):return np.log(np.asarray(values,dtype=float).clip(1e-12)).flatten()


def norm_fit(bank):
    x=np.stack([norm_vector(e['raw_norms'][0]) for e in bank.entries])
    return x.mean(0),x.std(0).clip(1e-4)


def norm_scores(bank,fit):
    mean,scale=fit;result=[]
    for i,e in enumerate(bank.entries):
        donor=bank.entries[(i+1)%len(bank.entries)]
        sources={'clean':e['raw_norms'][0],'counterfactual':e['raw_norms'][1],'mismatch':donor['raw_norms'][0],
                 **{'corrupted_'+kind:values for kind,values in e.get('control_raw_norms',{}).items()}}
        for category,values in sources.items():
            result.append({'id':e['id'],'group_id':e['group_id'],'category':category,
                           'score':float(np.square((norm_vector(values)-mean)/scale).sum())})
    return result


def detection(scores,thresholds,score_keys):
    from sklearn.metrics import roc_auc_score
    result=[]
    for key in score_keys:
        clean=[row for row in scores if row['category']=='clean']
        for category in sorted({r['category'] for r in scores}-{'clean'}):
            other=[row for row in scores if row['category']==category]
            for alpha in [.01,.05]:
                q=thresholds[key][str(alpha)]
                q=float('inf') if q is None else q
                result.append({'k':key,'alpha':alpha,'category':category,
                    'FPR':float(np.mean([r[key]>q for r in clean])),
                    'detection_rate':float(np.mean([r[key]>q for r in other])),
                    'AUROC':float(roc_auc_score([0]*len(clean)+[1]*len(other),[r[key] for r in clean+other]))})
    return result


def calibration_thresholds(scores,keys):
    return {key:{str(alpha):threshold([r[key] for r in scores if r['category']=='clean'],alpha)
                 for alpha in [.01,.05]} for key in keys}

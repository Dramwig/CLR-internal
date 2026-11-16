"""Temperature calibration, frozen thresholds, final QA/audit and natural-commitment outputs."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,math,time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import AutoTokenizer
from clr.models.m7 import Reader
from clr.audit.m7_reader import Bank,Serializer
from clr.audit.m7_evaluation import (predict,qa_calibration,fit_temperature,qa_summary,score_reader,fit_detectors,
                                    score_detectors,scalar,views)
from clr.data.m7 import threshold,TEMPLATES
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.audit.m7_runtime import Cost,snapshot,live


def load_model(c,kind,seed):
    pointer=json.loads(Path(c['output'],'trained',f'{kind}_{seed}.json').read_text())
    model=Reader(kind).to(c['device']);model.load_state_dict(torch.load(pointer['checkpoint'],map_location=c['device'],weights_only=True));model.eval()
    return model,pointer


def calibration(c,run,cost,serializer):
    root=Path(c['output']);folder=root/'calibration';folder.mkdir(exist_ok=False)
    a=Bank(root,'cal_A');b=Bank(root,'cal_B');train=Bank(root,'reader_train')
    detectors=fit_detectors(train,with_mismatch=c.get('detector_mismatch',True))
    import joblib
    joblib.dump(detectors,folder/'detectors.joblib')
    ds=score_detectors(b,detectors)
    thresholds={'readers':{},'detectors':{}}
    for method in ['norm','text']:
        vals=[r[method] for r in ds if r['category']=='clean']
        thresholds['detectors'][method]={str(alpha):scalar(threshold(vals,alpha)) for alpha in [.01,.05]}
    for kind in c['kinds']:
        for seed in c['seeds']:
            cost.check();model,pointer=load_model(c,kind,seed)
            rows,logits=qa_calibration(model,a,serializer,c,mixed=True)
            temps={}
            for task in ['A1','A2','A3','A4']:
                ii=[i for i,r in enumerate(rows) if r['task']==task]
                temps[task]=fit_temperature([logits[i] for i in ii],[rows[i]['label'] for i in ii])
            rr,ll=qa_calibration(model,b,serializer,c);qa_thresholds={}
            for task in ['A1','A2','A3','A4']:
                scores=[1-float(softmax(x/temps[task])[r['label']]) for r,x in zip(rr,ll) if r['task']==task]
                qa_thresholds[task]={str(alpha):scalar(threshold(scores,alpha)) for alpha in [.01,.05]}
            scores=score_reader(model,b,serializer,c,temps,seed)
            tt={str(k):{str(alpha):scalar(threshold([r[str(k)] for r in scores if r['category']=='clean'],alpha))
                        for alpha in [.01,.05]} for k in [1,4,16]}
            thresholds['readers'][f'{kind}_{seed}']={'temperatures':temps,'detection_thresholds':tt,'qa_thresholds':qa_thresholds,
                                                     'checkpoint':pointer['checkpoint'],'cal_B_groups':len(b.entries)}
            write_json(folder/f'{kind}_{seed}.json',{'scores':scores,'parameters':thresholds['readers'][f'{kind}_{seed}'],
                       'temperature_panel':[{'row':r,'logits':x.tolist()} for r,x in zip(rows,logits)]})
            cost.update();print('calibrated',kind,seed,flush=True);del model
    write_json(folder/'thresholds.json',thresholds)
    # The lock precedes all test source generation or prediction. Include source,
    # reader, serializer, configs, checkpoints and calibration artifacts.
    paths=[*Path('clr').rglob('*m7*.py'),Path('clr/sources/kv_relay.py'),
           Path('clr/models/clr_v0.py'),Path('clr/models/clr_v1.py'),Path('scripts/m7_env.sh'),
           *Path('scripts').glob('m7*.py'),*Path('configs/M7').glob('*.yaml'),
           *Path(root/'trained').glob('*.json'),*folder.glob('*')]
    paths.extend(Path('data/splits').glob(root.name+'_*.json'))
    paths.extend((root/'source').glob('*/manifest.jsonl'))
    paths.extend((root/'source').glob('*/mismatch_v2.jsonl'))
    for p in Path(root/'trained').glob('*.json'):paths.append(Path(json.loads(p.read_text())['checkpoint']))
    sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    dep=json.loads((root/'dependencies_lock.json').read_text())
    resources={**dep['idea_a_contract_sha256']}
    resources.update({str(Path('/home/me/PaperField/LatDefer')/p):digest for p,digest in dep['generator_core_sha256'].items()})
    for path,digest in resources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Idea A pinned source changed: '+path)
    sha.update(resources)
    sha[str(root/'dependencies_lock.json')]=hashlib.sha256((root/'dependencies_lock.json').read_bytes()).hexdigest()
    write_json(root/'final_lock.json',{'sha256':sha,'calibration_run':str(run.path),'test_opened_before_lock':False,
                                    'n_cal_B_independent_groups':len(b.entries),'alpha001_threshold_is_infinite':True})
    live(stage='calibration_completed_final_lock_created',calibration_run=str(run.path))


def natural_rows(bank,model,serializer,c,temps,kind,seed):
    if kind!='CLR':return []
    path=Path(c['output'])/'source'/c['split']/'natural_commitment.jsonl'
    records=[json.loads(l) for l in path.open()];rows=[];keys=[]
    for e in records:
        for mode in ['direct','cot512']:
            for tid in range(6):
                d=e[mode]
                rows.append({'episode_id':e['id'],'group_id':e['group_id'],'message_id':e['id'],
                             'task':'A3','template_id':tid,'question_text':TEMPLATES['A3'][tid].format(commitment=d['commitment']),
                             'options':['No','Yes'],'label':int(d['correct'])})
                keys.append({'id':e['id'],'group_id':e['group_id'],'mode':mode,'model_correct':d['correct'],
                             'template_id':tid,'kind':kind,'seed':seed,'split':c['split']})
    raw=predict(model,bank,rows,serializer,c['device'],c['batch_size'])
    return [{**key,'P_consistent':float(softmax(logit/temps['A3'])[1])} for key,logit in zip(keys,raw)]


def evaluate(c,run,cost,serializer):
    import joblib
    root=Path(c['output']);split=c['split'];thresholds=json.loads((root/'calibration/thresholds.json').read_text())
    if split=='test':
        lock=json.loads((root/'final_lock.json').read_text())
        for path,digest in lock['sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Final lock changed: '+path)
    bank=Bank(root,split,range(6));bank.add_mismatch_questions(range(6));qa=[];predictions=[];scores=[];natural=[]
    ds=score_detectors(bank,joblib.load(root/'calibration/detectors.joblib'))
    for row in ds:
        for method in ['norm','text']:
            for k in [1,4,16]:
                for alpha in [.01,.05]:
                    tt=thresholds['detectors'][method][str(alpha)]
                    scores.append({'split':split,'kind':method,'seed':-1,'k':k,'alpha':alpha,
                                   'score':row[method],'rejected':False if tt is None else row[method]>tt,
                                   'id':row['id'],'group_id':row['group_id'],'category':row['category']})
    for kind in c['kinds']:
        for seed in c['seeds']:
            cost.check();model,pointer=load_model(c,kind,seed);params=thresholds['readers'][f'{kind}_{seed}'];temps=params['temperatures']
            raw=predict(model,bank,bank.rows,serializer,c['device'],c['batch_size'])
            for task in ['A1','A2','A3','A4']:
                for templates in ['seen','heldout']:
                    for cat in ['clean','counterfactual','mismatch']:
                        ii=[i for i,r in enumerate(bank.rows) if r['task']==task and r['category']==cat
                            and ((r['template_id']<4)==(templates=='seen'))]
                        rs=[bank.rows[i] for i in ii];pp=np.stack([softmax(raw[i]/temps[task]) for i in ii])
                        summary=qa_summary(rs,pp);qa.append({'split':split,'kind':kind,'seed':seed,'task':task,'templates':templates,'category':cat,**summary})
                        for r,p in zip(rs,pp):
                            y=r['label']
                            pred={'split':split,'kind':kind,'seed':seed,'task':task,'templates':templates,'category':cat,
                                  'id':r['episode_id'],'group_id':r['group_id'],'template_id':r['template_id'],'label':y,
                                  'correct':int(p.argmax()==y),'nll':float(-np.log(max(p[y],1e-12))),
                                  'brier':float(np.square(p-np.eye(len(p))[y]).sum()),'probs':p.tolist()}
                            for alpha in [.01,.05]:
                                t=params['qa_thresholds'][task][str(alpha)];selected=np.ones(len(p),bool) if t is None else 1-p<=t
                                pred['coverage_'+str(alpha)]=bool(selected[y]);pred['size_'+str(alpha)]=int(selected.sum())
                            predictions.append(pred)
            ss=score_reader(model,bank,serializer,c,temps,seed)
            for r in ss:
                for k in [1,4,16]:
                    for alpha in [.01,.05]:
                        t=params['detection_thresholds'][str(k)][str(alpha)]
                        scores.append({'split':split,'kind':kind,'seed':seed,'k':k,'alpha':alpha,
                                       'score':r[str(k)],'rejected':False if t is None else r[str(k)]>t,
                                       'id':r['id'],'group_id':r['group_id'],'category':r['category']})
            if c.get('natural_commitment_analysis',True):natural+=natural_rows(bank,model,serializer,c,temps,kind,seed)
            pd.DataFrame(qa).to_csv(run.path/'qa_summary.csv',index=False)
            pd.DataFrame(scores).to_csv(run.path/'detection_scores.csv',index=False)
            pd.DataFrame(natural).to_csv(run.path/'natural_commitment.csv',index=False)
            cost.update(test_opened=split=='test');print('evaluated',split,kind,seed,flush=True);del model
    with (run.path/'qa_predictions.jsonl').open('w') as f:
        for r in predictions:f.write(json.dumps(r)+'\n')
    dest=root/'evaluated';dest.mkdir(exist_ok=True)
    write_json(dest/(split+'.json'),{'run':str(run.path),'test_opened':split=='test'})
    live(stage=split+'_evaluation_completed',evaluation_run=str(run.path))


def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c)
        try:
            serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            if c['phase']=='calibrate':calibration(c,run,cost,serializer)
            elif c['phase']=='evaluate':evaluate(c,run,cost,serializer)
            else:raise ValueError('Unknown evaluation phase')
            cost.update(status='completed',test_opened=c.get('split')=='test',source_forward_calls=0)
        except Exception:
            cost.update(status='failed',test_opened=c.get('split')=='test');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

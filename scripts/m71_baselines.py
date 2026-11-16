"""Raw-norm and visible-text baselines with independent clean calibration."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from clr.audit.m71_reader import Bank
from clr.audit.m71_runtime import snapshot,assert_final_lock
from clr.audit.m71_evaluation import norm_fit,norm_scores,calibration_thresholds,detection
from clr.audit.m7_evaluation import text_features
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def text_data(root,rep,split):
    folder=root/rep/'source'/split
    entries={r['id']:r for r in map(json.loads,(folder/'manifest.jsonl').open())}
    data=[]
    for row in map(json.loads,(folder/'receiver.jsonl').open()):
        entry={**entries[row['id']],'commitments':[entries[row['id']]['commitment']]}
        for category in ['clean','counterfactual','mismatch','corrupted_truncate','corrupted_zero_layer','corrupted_noise']:
            data.append({'id':entry['id'],'group_id':entry['group_id'],'category':category,
                         'features':text_features(entry,row[category])})
    return data


def dump(path,rows):
    with path.open('x') as handle:
        for row in rows:handle.write(json.dumps(row)+'\n')


def text_scores(data,weight,bias):
    from scipy.special import expit
    return [{key:row[key] for key in ['id','group_id','category']}|{'score':float(expit(row['features']@weight+bias))} for row in data]


def main(c):
    with recorded(c) as run:
        snapshot(run);root=Path(c['output']);rep=c['representation'];folder=root/rep/'baselines';folder.mkdir(exist_ok=True)
        path=folder/(c['kind']+'.json')
        if c['mode']=='calibrate':
            if c['kind']=='norm':
                train=Bank(root,'reader_train',representation=rep,load_sequences=False)
                fitted=norm_fit(train)
                bank=Bank(root,'cal_B',representation=rep,load_sequences=False)
                scores=norm_scores(bank,fitted)
                parameters={'mean':fitted[0].tolist(),'scale':fitted[1].tolist()}
            elif c['kind']=='text':
                data=text_data(root,rep,'cal_A')
                x=np.stack([row['features'] for row in data]);y=np.array([row['category']!='clean' for row in data])
                model=LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=11).fit(x,y)
                parameters={'weight':model.coef_[0].tolist(),'bias':float(model.intercept_[0])}
                scores=text_scores(text_data(root,rep,'cal_B'),model.coef_[0],model.intercept_[0])
            else:raise ValueError(c['kind'])
            dump(run.path/'cal_B_scores.jsonl',scores)
            write_json(path,{'run':str(run.path),'parameters':parameters,'thresholds':calibration_thresholds(scores,['score']),
                             'clean_calibration_groups':sum(row['category']=='clean' for row in scores),'test_opened':False})
        elif c['mode']=='final':
            assert_final_lock(c);fit=json.loads(path.read_text());parameters=fit['parameters']
            for split in c['source_splits']:
                if split=='test':run.write_metrics(test_opened=True)
                if c['kind']=='norm':
                    bank=Bank(root,split,representation=rep,load_sequences=False)
                    scores=norm_scores(bank,(np.array(parameters['mean']),np.array(parameters['scale'])))
                else:scores=text_scores(text_data(root,rep,split),np.array(parameters['weight']),parameters['bias'])
                dump(run.path/(split+'_scores.jsonl'),scores)
                write_json(run.path/(split+'_detection.json'),detection(scores,fit['thresholds'],['score']))
            write_json(folder/(c['kind']+'_evaluated.json'),{'run':str(run.path),'calibration':str(path),'test_opened':True})
        else:raise ValueError(c['mode'])
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,test_opened=c['mode']=='final')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

"""Calibrate frozen M7.1 readers, then evaluate locked final test only."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer
from clr.audit.m71_reader import Bank,Serializer
from clr.audit.m71_runtime import Cost,snapshot,live,assert_final_lock
from clr.audit.features import sha256
from clr.audit.m71_storage import save_bundle
from clr.audit.m71_evaluation import (predict,temperatures,score_reader,norm_fit,norm_scores,
    calibration_thresholds,detection,views,calibration_rows)
from clr.data.m7 import threshold
from clr.audit.m7_evaluation import qa_summary
from clr.models.m71 import Reader
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded


def jsonl(path,rows):
    with Path(path).open('x') as handle:
        for row in rows:handle.write(json.dumps(row)+'\n')


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);root=Path(c['output']);rep=c['representation']
        live(stage='evaluation_'+c['mode'],evaluation_run=str(run.path));opened=False
        try:
            training=json.loads((root/rep/'training'/f'{c["kind"]}_{c["seed"]}.json').read_text())
            model=Reader(c['kind']).to(c['device']);model.load_state_dict(torch.load(training['checkpoint'],weights_only=True,map_location='cpu'));model.eval()
            sequences=c['kind'] in ['CLR','supervised']
            serializer=None if c['kind'] in ['linear','B3'] else Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            folder=root/rep/'calibration';folder.mkdir(exist_ok=True)
            pointer=folder/f'{c["kind"]}_{c["seed"]}.json'
            if c['mode']=='calibrate':
                calA=Bank(root,'cal_A',representation=rep,load_sequences=sequences)
                calB=Bank(root,'cal_B',representation=rep,load_sequences=sequences)
                temp=temperatures(model,calA,serializer,c)
                if c['kind']=='supervised':
                    temp={key:temp['A3'] for key in temp}
                    scores=supervised_scores(model,calB,serializer,c,temp['A3'])
                    keys=['score']
                else:scores=score_reader(model,calB,serializer,c,temp,c['seed']);keys=['1','4','16']
                thresholds=calibration_thresholds(scores,keys)
                qa_conformal={}
                if c['kind']!='supervised':
                    qr=calibration_rows(calB,mixed=False);raw=predict(model,calB,qr,serializer,c)
                    for task in ['A1','A2','A3','A4']:
                        nonconformity=[1-float(softmax(logits/temp[task])[row['label']])
                                       for row,logits in zip(qr,raw) if row['task']==task]
                        qa_conformal[task]={str(alpha):threshold(nonconformity,alpha) for alpha in [.01,.05]}
                jsonl(run.path/'cal_B_scores.jsonl',scores)
                write_json(pointer,{'run':str(run.path),'training':training,'checkpoint_sha256':sha256(training['checkpoint']),
                    'temperatures':temp,'thresholds':thresholds,'qa_conformal':qa_conformal,'test_opened':False})
            elif c['mode']=='final':
                assert_final_lock(c);cal=json.loads(pointer.read_text())
                if sha256(training['checkpoint'])!=cal['checkpoint_sha256']:raise ValueError('Calibrated checkpoint changed')
                temp=cal['temperatures'];metrics=[];coverage=[]
                for split in ['reader_val','test']:
                    cost.check();opened|=split=='test'
                    if split=='test':
                        cost.update(test_opened=True)
                        live(test_opened=True)
                    bank=Bank(root,split,templates=(0,1,2,3,4,5),representation=rep,load_sequences=sequences)
                    if c['kind']=='supervised':
                        scores=supervised_scores(model,bank,serializer,c,temp['A3']);keys=['score']
                    else:
                        scores=score_reader(model,bank,serializer,c,temp,c['seed']);keys=['1','4','16']
                        rows=bank.rows;raw=predict(model,bank,rows,serializer,c)
                        for task in ['A1','A2','A3','A4']:
                            for held in [False,True]:
                                ii=[i for i,r in enumerate(rows) if r['task']==task and (r['template_id']>=4)==held]
                                rr=[rows[i] for i in ii];p=softmax(np.stack([raw[i] for i in ii])/temp[task],axis=1)
                                metrics.append({'split':split,'task':task,'heldout_templates':held,**qa_summary(rr,p)})
                                for category in ['clean','counterfactual']:
                                    take=np.array([r['category']==category for r in rr]);yy=np.array([r['label'] for r in rr])[take]
                                    for alpha in [.01,.05]:
                                        q=cal['qa_conformal'][task][str(alpha)];sets=(1-p[take])<=q
                                        coverage.append({'split':split,'task':task,'heldout_templates':held,'category':category,'alpha':alpha,
                                            'coverage':float(sets[np.arange(len(yy)),yy].mean()),'mean_set_size':float(sets.sum(1).mean()),
                                            'empty_rate':float((sets.sum(1)==0).mean())})
                                save_bundle({'rows':rr,'probs':torch.from_numpy(p)},run.path/f'{split}_{task}_{"heldout" if held else "seen"}.pt.gz')
                        mismatch=bank.add_mismatch_questions();mx=predict(model,bank,mismatch,serializer,c)
                        for task in ['A1','A2','A3','A4']:
                            ii=[i for i,r in enumerate(mismatch) if r['task']==task]
                            rr=[mismatch[i] for i in ii];p=softmax(np.stack([mx[i] for i in ii])/temp[task],axis=1)
                            save_bundle({'rows':rr,'probs':torch.from_numpy(p)},run.path/f'{split}_{task}_mismatch.pt.gz')
                    jsonl(run.path/f'{split}_scores.jsonl',scores)
                    write_json(run.path/f'{split}_detection.json',detection(scores,cal['thresholds'],keys))
                    cost.update(test_opened=split=='test')
                write_json(run.path/'qa_metrics.json',metrics)
                write_json(run.path/'qa_conformal.json',coverage)
                target=root/rep/'evaluated';target.mkdir(exist_ok=True)
                write_json(target/f'{c["kind"]}_{c["seed"]}.json',{'run':str(run.path),'training':training,'calibration':str(pointer),'test_opened':True})
            else:raise ValueError(c['mode'])
            cost.update(status='completed',test_opened=c['mode']=='final',source_forward_calls=0)
            live(stage='evaluation_'+c['mode']+'_completed',evaluation_run=str(run.path))
        except Exception:
            cost.update(status='failed',test_opened=opened);live(stage='evaluation_failed',evaluation_run=str(run.path));raise


def supervised_scores(model,bank,serializer,c,temp):
    # Binary attack label is exactly 1-consistency for the registered CF attack.
    # This dedicated model is trained only on A3 clean/CF labels.
    from clr.data.m7 import TEMPLATES
    vv=views(bank)
    rows=[{'message_id':v['message_id'],'task':'A3','question_text':TEMPLATES['A3'][0].format(commitment=v['entry']['commitment']),
           'options':['No','Yes'],'label':0} for v in vv]
    raw=predict(model,bank,rows,serializer,c)
    return [{'id':v['entry']['id'],'group_id':v['entry']['group_id'],'category':v['category'],
             'score':float(softmax(logits/temp)[0])} for v,logits in zip(vv,raw)]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

"""Identity-aligned Q3 permutation sensitivity, canonical API and raw ordering."""
import argparse,json,time,itertools
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.m12_runtime import recorded,Budget,digest
from clr.data.m12 import TensorSerializer
from clr.data.m4.reader import rows_for,PrefixBank
from clr.models.m12 import make_model
from clr.data.questions.core import seed as stable_seed
from scripts.m41_evaluate import predict
from clr.paper.statistics import metric_summary

def main(c):
 c={**c,'stage':'permutation','case':'permutation'}
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);budget=Budget(run,c,start);status='failed'
    try:
        budget.check();lock=json.loads(Path(c['output'],'training_complete.json').read_text())
        if not lock['all_training_locked']:raise ValueError('Test locked')
        train=json.loads(Path('data/M41_v1/training_suite.json').read_text());templates=load_config(c['templates']);device=c['reader_device']
        tokenizer=AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True);perms=np.array([p for p in itertools.permutations(range(4)) if p!=(0,1,2,3)])
        records=[];details=[]
        for split in ['reader_val','test']:
            rows=[r for r in rows_for(c,split,'final',qtypes=['Q3']) if r['state_ref']['kind']=='clean'];bank=PrefixBank(c,split,'final',{r['problem_id'] for r in rows})
            for a in ['main','A1']:
                if a=='A1' and a not in lock['retained']:continue
                for s in [11,22,33]:
                    seed_everything(s);model=make_model({**c,'ablation':a}).to(device)
                    checkpoint=Path(train[f'main_lite_{s}'] if a=='main' else lock['runs'][f'A1_{s}'])/'best.pt'
                    if a=='A1' and digest(checkpoint)!=lock['checkpoint_sha256'][f'A1_{s}']:raise ValueError('A1 hash mismatch')
                    model.restore_trainable(torch.load(checkpoint,map_location='cpu',weights_only=True))
                    permutation=np.array([perms[stable_seed('M12_perm',s,r['problem_id'],r['state_ref']['t'],r['template_id'],r['replicate'])%len(perms)] for r in rows])
                    changed=[{**r,'options':[r['options'][j] for j in p]} for r,p in zip(rows,permutation)]
                    for canonical in [True,False]:
                        serializer=TensorSerializer(tokenizer,templates,c['max_length'],canonical=canonical)
                        original=predict(model,'clr',rows,bank,serializer,templates,None,device,c['eval_batch'],budget)
                        permuted=predict(model,'clr',changed,bank,serializer,templates,None,device,c['eval_batch'],budget)
                        pred0=original.argmax(1);pred1=permutation[np.arange(len(rows)),permuted.argmax(1)];different=pred0!=pred1
                        aligned=np.take_along_axis(original,permutation,1)
                        tied=(original==original.max(1,keepdims=True)).sum(1)>1
                        for r,d,pr0,pr1 in zip(rows,different,pred0,pred1):
                            records.append({'variant':a,'seed':s,'split':split,'canonical':canonical,'problem_id':r['problem_id'],'t':r['state_ref']['t'],'template_id':r['template_id'],'replicate':r['replicate'],'changed':int(d),'original_identity':int(pr0),'permuted_identity':int(pr1)})
                        details.append({'variant':a,'seed':s,'split':split,'canonical':canonical,'n':len(rows),'changes':int(different.sum()),'original_top_ties':int(tied.sum()),'changes_with_original_top_ties':int((different&tied).sum()),'max_abs_logit_difference':float(np.abs(aligned-permuted).max())})
                        print(details[-1],flush=True)
                    del model
            del bank
        frame=pd.DataFrame(records);frame.to_parquet(Path(c['output'])/'permutation_observations.parquet',index=False)
        result=[]
        for key,g in frame.groupby(['variant','split','canonical']):
            f=g.groupby(['seed','problem_id','t']).changed.agg(['mean','size']).reset_index().rename(columns={'mean':'changed','size':'instances'})
            e=metric_summary(f,['changed'])[0];result.append(dict(zip(['variant','split','canonical'],key),**e,n=len(g),changes=int(g.changed.sum())))
        pd.DataFrame(result).to_csv(Path(c['output'])/'permutation.csv',index=False);write_json(run.path/'details.json',details)
        status='completed';run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=True)
        write_json(Path(c['output'])/'permutation_complete.json',{'run':str(run.path)})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

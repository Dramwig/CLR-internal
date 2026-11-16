"""M12 final inference: shared all-variant lock, frozen M41 calibration definitions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.m12_runtime import recorded,Budget,digest
from clr.data.m12 import TensorSerializer as Serializer
from clr.models.m12 import make_model
from clr.data.m4.reader import PrefixBank,rows_for
from clr.calib.m4 import fit
from clr.calib.m4_evaluation import target_array,thresholds,summarize
from scripts.m41_evaluate import predict

def main(c):
 c={**c,'stage':'evaluate'}
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2);seed_everything(c['seed']);device=c['reader_device'];budget=Budget(run,c,start);status='failed'
    try:
        budget.check();smoke=c['mode']=='smoke'
        if not smoke:
            lock=json.loads(Path(c['eval_lock']).read_text())
            if not lock['all_training_locked']:raise ValueError('All variants must be locked')
            for p,h in lock['code_sha256'].items():
                if digest(p)!=h:raise ValueError('Code changed after training lock: '+p)
            if lock['checkpoint_sha256'][c['case']+'_'+str(c['seed'])]!=digest(c['checkpoint']):raise ValueError('Checkpoint differs from lock')
            write_json(Path(c['output'])/'test_access.json',{'first_final_run':str(run.path),'lock':c['eval_lock'],'timestamp':time.time()}) if not Path(c['output'],'test_access.json').exists() else None
        templates=load_config(c['templates']);serializer=Serializer(AutoTokenizer.from_pretrained(c['model_path'],local_files_only=True),templates,c['max_length'])
        model=make_model(c).to(device);model.restore_trainable(torch.load(c['checkpoint'],map_location='cpu',weights_only=True))
        if smoke:
            allval=rows_for(c,'reader_val','validate',qtypes=c['qtypes']);ids=sorted({r['problem_id'] for r in allval})[:50];caids=set(ids[:15]);cbids=set(ids[15:30]);evalids=set(ids[30:]);calrows=[r for r in allval if r['problem_id'] in caids|cbids];calbank=PrefixBank(c,'reader_val','validate',set(ids))
        else:calrows=rows_for(c,'cal','calibrate',qtypes=c['qtypes']);calbank=PrefixBank(c,'cal','calibrate')
        calibrations={};threshold_map={};stats=[];cp=[]
        for q in c['qtypes']:
            ca=[r for r in calrows if r['qtype']==q and (r['problem_id'] in caids if smoke else r['cal_part']=='cal_A')]
            cb=[r for r in calrows if r['qtype']==q and (r['problem_id'] in cbids if smoke else r['cal_part']=='cal_B')]
            a=predict(model,'clr',ca,calbank,serializer,templates,None,device,c['eval_batch'],budget)
            b=predict(model,'clr',cb,calbank,serializer,templates,None,device,c['eval_batch'],budget)
            calibrations[q]={'temperature':fit(a,target_array(ca,q),'temperature')}
            threshold_map[q]={'temperature':thresholds(cb,b,calibrations[q]['temperature'],q,c['seed'])}
        write_json(run.path/'calibration.json',calibrations)
        write_json(run.path/'thresholds.json',{q:{m:{k:(v if np.isfinite(v) else None) for k,v in ts.items()} for m,ts in methods.items()} for q,methods in threshold_map.items()})
        del calbank,calrows
        for split in ['reader_val'] if smoke else ['reader_val','test']:
            phase='validate' if smoke else 'final';bank=PrefixBank(c,split,phase,set(ids) if smoke else None)
            for q in c['qtypes']:
                for held in [False,True]:
                    rows=rows_for(c,split,phase,heldout=held,ids=evalids if smoke else None,qtypes=[q])
                    logits=predict(model,'clr',rows,bank,serializer,templates,None,device,c['eval_batch'],budget)
                    pred=[{**{k:r[k] for k in ['problem_id','template_id','replicate','label','label_soft','options']},'t':r['state_ref']['t'],'traj_id':r['state_ref']['traj_id'],'kind':r['state_ref']['kind'],'heldout':held,'logits':x.tolist()} for r,x in zip(rows,logits)]
                    pq.write_table(pa.Table.from_pylist(pred),run.path/f'{q}_{split}_{held}.parquet',compression='zstd')
                    st,co=summarize(rows,logits,q,calibrations[q],threshold_map[q],c['seed']);common={'case':c['case'],'seed':c['seed'],'split':split,'subset':'all'}
                    stats.extend({**common,**x} for x in st);cp.extend({**common,**x} for x in co)
                    write_json(run.path/'statistics.json',stats);write_json(run.path/'conformal.json',cp);print(split,q,held,len(rows),flush=True)
            del bank
        status='completed';run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0,test_opened=not smoke)
        write_json(Path(c['output'])/f"evaluated_{c['case']}_{c['seed']}.json",{'run':str(run.path),'checkpoint':c['checkpoint'],'test_opened':not smoke})
    finally:
        if status!='completed':run.write_metrics(status=status,gpu_hours=(time.monotonic()-start)/3600,source_forward_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

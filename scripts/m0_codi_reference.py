"""Execute the unmodified official CODI test.py; only supply local verified data."""
import argparse,contextlib,io,json,sys,time
from pathlib import Path
import torch
from datasets import Dataset
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything
from clr.sources.common import import_file,check_device
from clr.utils.stats import mean_ci

def main(c):
 with recorded(c) as run:
    t=time.perf_counter();check_device(c['source'])
    rows=json.loads(Path(c['data_path']).read_text())
    if c['mode']=='smoke':
        if c['split']=='test' or c['limit']>len(rows)*.05:raise ValueError('Invalid smoke')
        rows=sorted(rows,key=lambda r:r['id'])[:c['limit']]
    else:
        d=json.loads(Path(c['smoke_run'],'metrics.json').read_text())
        if d['status']!='completed':raise RuntimeError('Official reference smoke required')
    sys.path.insert(0,str(Path('third_party/codi').resolve()))
    mod=import_file('third_party/codi/test.py','clr_official_test')
    # Do not modify any upstream file. Same question/answer test records from official recipe.
    dataset=Dataset.from_list([{'question':r['question'],'answer':r['answer']} for r in rows])
    mod.load_dataset=lambda *a,**k:{'test':dataset}
    model_args=mod.ModelArguments(model_name_or_path=c['source']['base_path'],train=False,lora_init=True,lora_r=128,lora_alpha=32,ckpt_dir=str(Path(c['source']['checkpoint']).parent))
    data_args=mod.DataArguments(data_name='gsm8k',batch_size=c['batch_size']);mod.data_args=data_args
    train=mod.TrainingArguments(output_dir=str(run.path/'official'),bf16=True,seed=11,model_max_length=512,
        greedy=True,num_latent=6,use_prj=True,prj_dim=768,prj_no_ln=False,prj_dropout=0.,inf_latent_iterations=6,
        inf_num_iterations=1,remove_eos=True,use_lora=True,report_to=[])
    counts=[0,0]
    original_class=mod.CODI
    class CountedCODI(original_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            def count(module,args,kwargs):
                counts[0]+=1
                tensor=kwargs.get('input_ids',kwargs.get('inputs_embeds'))
                counts[1]+=tensor.shape[0]
            self.codi.register_forward_pre_hook(count,with_kwargs=True)
    mod.CODI=CountedCODI
    vals=[]
    for seed in c['seeds']:
        seed_everything(seed)
        with (run.path/f'official_seed{seed}.log').open('w') as f,contextlib.redirect_stdout(f):
            value=mod.evaluation(model_args,data_args,train)
        vals.append(value/100);run.write_metrics(seed_accuracies=vals)
    run.write_metrics(status='completed',accuracy=mean_ci(vals),problems=len(rows),wall_seconds=time.perf_counter()-t,gpu_hours=(time.perf_counter()-t)/3600,
        source_forward_calls=counts[0],source_forward_sample_evaluations=counts[1],cost_note='Pure pre-forward counter hook; allocated single-device wall hours')
    print(run.path,vals)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

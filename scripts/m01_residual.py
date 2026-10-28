"""Trace the remaining official/CLR disagreement without modifying upstream."""
import argparse,contextlib,json,os,sys,time
from pathlib import Path
import torch
from datasets import Dataset
from clr.sources.common import import_file
from clr.sources.codi_v2 import load
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json

def main(c):
 with recorded(c) as run:
    start=time.perf_counter();torch.set_num_threads(2);seed_everything(11)
    os.environ['NCCL_P2P_DISABLE']='1';os.environ['NCCL_IB_DISABLE']='1'
    conf=load_config(c['protocol'])['source'];torch.cuda.set_device(conf['device'])
    rows=json.loads(Path(c['data']).read_text())[c['start']:c['end']]
    sys.path.insert(0,str(Path('third_party/codi').resolve()));mod=import_file('third_party/codi/test.py','official_residual_v2')
    mod.load_dataset=lambda *a,**k:{'test':Dataset.from_list(rows)}
    traces={'official':[],'clr':[]};models=[];orig=mod.CODI
    def hook(name):
        def capture(module,args,kw,out):
            x=kw.get('input_ids',kw.get('inputs_embeds'));h=getattr(out,'hidden_states',None)
            logits=out.logits[c['local_index'],-1,:].float().cpu()
            traces[name].append({'shape':list(x.shape),'strides':list(x.stride()),'input':x.detach().cpu(),'mask':kw.get('attention_mask').cpu() if kw.get('attention_mask') is not None else None,'hidden':h[-1][:,-1,:].detach().cpu() if h else None,'logits':logits})
        return capture
    class Counted(orig):
        def __init__(self,*a,**k):
            super().__init__(*a,**k);models.append(self);self.codi.register_forward_hook(hook('official'),with_kwargs=True)
    mod.CODI=Counted
    ma=mod.ModelArguments(model_name_or_path=conf['base_path'],train=False,lora_init=True,lora_r=128,lora_alpha=32,ckpt_dir=str(Path(conf['checkpoint']).parent))
    da=mod.DataArguments(data_name='gsm8k',batch_size=128);mod.data_args=da
    ta=mod.TrainingArguments(output_dir=str(run.path/'official'),bf16=True,seed=11,model_max_length=512,greedy=True,num_latent=6,use_prj=True,prj_dim=768,prj_no_ln=False,prj_dropout=0.,inf_latent_iterations=6,inf_num_iterations=1,remove_eos=True,use_lora=True,report_to=[])
    # TrainingArguments initializes cuda:0; restore the configured device.
    torch.cuda.set_device(conf['device'])
    with (run.path/'official.log').open('w') as f,contextlib.redirect_stdout(f):mod.evaluation(ma,da,ta)
    source=load(conf);source.model.codi.register_forward_hook(hook('clr'),with_kwargs=True)
    weight_diffs=[]
    for name,x in models[0].state_dict().items():
        y=source.model.state_dict()[name]
        if not torch.equal(x,y):weight_diffs.append({'name':name,'max_abs':(x.float()-y.float()).abs().max().item()})
    width=max(len(source.tokenizer.encode(r['question'].strip().replace('  ',' '))) for r in rows)
    outputs=source.run_batch([{**r,'pad_to':width} for r in rows],capture_latents=False)
    write_json(run.path/'predictions.json',[{'id':r['id'],'answer':o['answer']} for r,o in zip(rows,outputs)])
    comparisons=[]
    for i,(a,b) in enumerate(zip(traces['official'],traces['clr'])):
        comparisons.append({'call':i,'shapes':[a['shape'],b['shape']],'strides':[a['strides'],b['strides']],
          'input_max_abs':(a['input'].float()-b['input'].float()).abs().max().item(),
          'mask_equal':(a['mask'] is None and b['mask'] is None) or (a['mask'] is not None and b['mask'] is not None and torch.equal(a['mask'],b['mask'])),
          'hidden_max_abs':(a['hidden'].float()-b['hidden'].float()).abs().max().item() if a['hidden'] is not None and b['hidden'] is not None else None,
          'logits_max_abs':(a['logits']-b['logits']).abs().max().item(),
          'official_top5':torch.topk(a['logits'][:-1],5).indices.tolist(),'clr_top5':torch.topk(b['logits'][:-1],5).indices.tolist(),
          'official_values':torch.topk(a['logits'][:-1],5).values.tolist(),'clr_values':torch.topk(b['logits'][:-1],5).values.tolist()})
    write_json(run.path/'trace_comparison.json',comparisons)
    run.write_metrics(status='completed',weight_differences=weight_diffs,source_forward_calls=sum(map(len,traces.values())),wall_seconds=time.perf_counter()-start,gpu_hours=(time.perf_counter()-start)/3600)
    print(run.path,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

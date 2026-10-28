"""Pinned official Huginn + native lm-eval GSM8K-CoT protocol (M0b v2).

All original task prompts, fixed eight exemplars, chat formatting, stop rules,
and strict/flexible filters come from lm-eval 0.4.8. Only the dataset is routed
through frozen project IDs. This is not the old M0 hand-written zero-shot loop.
"""
import argparse
import hashlib
import json
import importlib.metadata
from pathlib import Path
import signal
import time
import types
import torch  # Must precede PIL/datasets with this host's inherited libstdc++.
import numpy as np
import yaml
import lm_eval
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.huggingface import HFLM
from clr.sources.common import RelativeNoise
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json, seed_everything


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_task(c, run):
    native = Path(lm_eval.__file__).parent/'tasks/gsm8k/gsm8k-cot.yaml'
    cfg = yaml.safe_load(native.read_text())
    rows = json.loads(Path(c['data_path']).read_text())
    split = json.loads(Path(c['split_path']).read_text())
    allowed = set(split[c['split']])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Input repeats a problem ID')
    if any(r['id'] not in allowed for r in rows):
        raise ValueError('Input contains IDs outside the frozen split')
    rows = sorted(rows, key=lambda r:r['id'])[:c['limit']]
    if c['mode']=='smoke' and len(rows) > .05*len(allowed):
        raise ValueError('Smoke must contain <=5% of the split')
    write_json(run.path/'selected_data.json',rows)
    write_json(run.path/'data_provenance.json',{
        'ids':[r['id'] for r in rows], 'data_sha256':sha(c['data_path']),
        'split_sha256':sha(c['split_path']), 'task_sha256':sha(native),
        'lm_eval_version':importlib.metadata.version('lm_eval'), 'original_task':str(native)})
    (run.path/'official_task.yaml').write_text(native.read_text())
    cfg.update(task='gsm8k_cot_m0b_v2',dataset_path='json',dataset_name=None,
               dataset_kwargs={'data_files':{'test':str((run.path/'selected_data.json').resolve())}})
    write_json(run.path/'effective_task_config.json',cfg)
    return ConfigurableTask(config=cfg)


def question_ci(values,seed=907):
    x=np.asarray(values,dtype=float)
    if not len(x):
        raise ValueError('Cannot bootstrap empty predictions')
    rng=np.random.default_rng(seed)
    vals=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(10000)])
    return {'mean':float(x.mean()),'ci95':[float(v) for v in np.quantile(vals,[.025,.975])],
            'bootstrap_unit':'problem_id','bootstrap_replicates':10000,'n':len(x)}


def summarize_samples(rows, expected_ids):
    metrics={}
    for filter_name in ('strict-match','flexible-extract'):
        subset=[r for r in rows if r['filter']==filter_name]
        ids=[r['doc']['id'] for r in subset]
        if len(ids)!=len(expected_ids) or set(ids)!=set(expected_ids):
            raise ValueError('Missing or repeated problem IDs in harness output')
        metrics['exact_match,'+filter_name]=question_ci([r['exact_match'] for r in subset])
    return metrics


def main(c):
    with recorded(c) as run:
        entry=Path(__file__);(run.path/'entrypoint_snapshot.py').write_text(entry.read_text())
        write_json(run.path/'entrypoint_sha256.json',{'sha256':sha(entry)})
        start=time.perf_counter()
        def timeout(signum,frame):
            raise TimeoutError('M0b smoke exceeded the configured wall-clock gate; no full evaluation allowed')
        signal.signal(signal.SIGALRM,timeout)
        if c['mode']=='smoke':
            signal.alarm(min(900,c['timeout_seconds']))
        if c['mode']!='smoke':
            if not c.get('smoke_run'):
                raise ValueError('Full evaluation requires a successful smoke run')
            previous=json.loads((Path(c['smoke_run'])/'metrics.json').read_text())
            if previous.get('status')!='completed' or not previous.get('smoke_passed'):
                raise ValueError('Smoke gate has not passed')
        manifest=json.loads(Path(c['verified_manifest']).read_text())
        if manifest['revision']!=c['revision'] or len(manifest['files'])!=4:
            raise ValueError('Missing pinned, SHA-verified official weight manifest')
        snapshot=Path(manifest['path'])
        # The download run verified the bytes. Recheck metadata to reject partial files.
        for f in manifest['files']:
            if (snapshot/f['file']).stat().st_size != f['bytes']:
                raise ValueError('Verified weight changed size')
        task=make_task(c,run)
        seed_everything(c['seed'])
        free,total=torch.cuda.mem_get_info(torch.device(c['device']))
        write_json(run.path/'gpu_before_load.json',{'free_bytes':free,'total_bytes':total,'name':torch.cuda.get_device_name(c['device'])})
        if free < c['min_free_gib']*1024**3:
            raise RuntimeError(f'Insufficient free VRAM: {free/1024**3:.2f} GiB, required {c["min_free_gib"]}')
        gpu_start=time.perf_counter()
        model=HFLM(pretrained=str(snapshot),trust_remote_code=True,dtype='bfloat16',
                   mean_recurrence=c['mean_recurrence'],device=c['device'],batch_size=1,
                   local_files_only=True)
        calls={'forward':0,'core':0}
        def count(module,inputs): calls['forward']+=1
        handle=model.model.register_forward_pre_hook(count)
        original=model.model.core_block_forward
        state={'step':0,'noise':None}
        trace={'pass':0,'vectors':[],'index':0,'max_diff':0.0}
        original_generate=model.model.generate
        def generate(this,*args,**kwargs):
            state['step']=0
            state['noise']=RelativeNoise({'sigma':c.get('sigma',0),'from_step':c.get('from_step',0),'seed':c.get('perturb_seed',c['seed'])},c['device'])
            return original_generate(*args,**kwargs)
        model.model.generate=types.MethodType(generate,model.model)
        def core(this,*args,**kwargs):
            x,idx=original(*args,**kwargs)
            step=state['step'];state['step']+=1;calls['core']+=1
            if c.get('sigma',0)>0 and step>=c.get('from_step',0):
                # Each recurrent state at the current answer position, after core,
                # before the next recurrence. Initial state remains official random.
                x=x.clone();x[:,-1:,:]=state['noise'](x[:,-1:,:],step)
            if c.get('verify_determinism',False):
                z=x[:,-1:,:].detach().cpu()
                if trace['pass']==0:
                    trace['vectors'].append(z.clone())
                else:
                    if trace['index']>=len(trace['vectors']):
                        raise ValueError('Repeated evaluation has more recurrent steps')
                    ref=trace['vectors'][trace['index']]
                    if ref.shape!=z.shape:
                        raise ValueError('Repeated evaluation changed latent shape')
                    trace['max_diff']=max(trace['max_diff'],float((ref.float()-z.float()).abs().max()))
                    trace['index']+=1
            return x,idx
        model.model.core_block_forward=types.MethodType(core,model.model)
        write_json(run.path/'model_config.json',model.model.config.to_dict())
        write_json(run.path/'generation_config.json',model.model.generation_config.to_dict())
        write_json(run.path/'effective_harness_settings.json',{'max_length':model.max_length,
            'max_gen_toks':model.max_gen_toks,'add_bos_token':model.add_bos_token,
            'batch_size':model.batch_size,'padding_side':model.tokenizer.padding_side,
            'eot_token_id':model.eot_token_id})
        eval_kwargs=dict(model=model,tasks=[task],batch_size=1,
            num_fewshot=8,bootstrap_iters=0,log_samples=True,
            system_instruction=c['system_instruction'],apply_chat_template=True,
            fewshot_as_multiturn=True,random_seed=c['seed'],numpy_random_seed=c['seed'],
            torch_random_seed=c['seed'],fewshot_random_seed=c['seed'])
        results=lm_eval.simple_evaluate(**eval_kwargs)
        determinism={}
        if c.get('verify_determinism',False):
            trace['pass']=1
            repeated=lm_eval.simple_evaluate(**eval_kwargs)
            if trace['index']!=len(trace['vectors']):
                raise ValueError('Repeated evaluation has fewer recurrent steps')
            def by_id(result):
                return {(r['doc']['id'],r['filter']):r['filtered_resps'] for r in result['samples']['gsm8k_cot_m0b_v2']}
            a,b=by_id(results),by_id(repeated)
            determinism={'max_latent_elementwise_difference':trace['max_diff'],
                'answers_identical':a==b,'compared_latent_steps':len(trace['vectors']),
                'same_rng_seed':c['seed'],'dtype':'bfloat16'}
            write_json(run.path/'determinism.json',determinism)
            trace['vectors'].clear()
        handle.remove()
        samples=results.pop('samples')
        # Harness responses and both official answer filters retained verbatim.
        (run.path/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2,default=str)+'\n')
        (run.path/'harness_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str)+'\n')
        rows=samples['gsm8k_cot_m0b_v2']
        metrics=summarize_samples(rows,[r['id'] for r in task.eval_docs])
        signal.alarm(0)
        run.write_metrics(status='completed',smoke_passed=c['mode']=='smoke',
            metrics=metrics,determinism=determinism,wall_seconds=time.perf_counter()-start,
            gpu_hours=(time.perf_counter()-gpu_start)/3600,gpu_model=torch.cuda.get_device_name(c['device']),
            source_forward_calls=calls['forward'],recurrent_core_calls=calls['core'],n=len(task.eval_docs))
        print(run.path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*')
    args=p.parse_args();main(load_config(args.config,args.overrides))

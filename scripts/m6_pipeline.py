"""Persistent M6 controller: no success inferred from a process launch.

Every GPU child checks cumulative section and total caps. All training choices
and exact checkpoint hashes lock before reader test predictions are opened.
Optional lower-priority work is admitted using cost only, never test metrics.
"""
import argparse,copy,hashlib,json,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import yaml
import threading
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import live,hours,snapshot

ROOT=Path('/home/me/PaperField/CalReader');PY=ROOT/'.venv/bin/python'


def qwen_workload(sources,validation_limit=128):
    """Frozen QA counts only; no state, answer, label or logit is loaded."""
    from collections import Counter
    import pyarrow.parquet as pq
    from clr.data.questions.core import seed
    result={}
    for source,spec in sources.items():
        root=Path(spec['qa']);counts={}
        for split in ['reader_train','reader_val','cal','test']:
            counts[split]=pq.read_table(root/f'{split}.parquet',columns=['is_heldout_template'],filters=[('is_heldout_template','=',False)]).num_rows
        ids=pq.read_table(root/'reader_val.parquet',columns=['problem_id'],filters=[('is_heldout_template','=',False)]).column('problem_id').to_pylist()
        per_problem=Counter(ids);selected=sorted(per_problem,key=lambda pid:seed(11,'M6-selection',pid))[:validation_limit]
        result[source]={'train':counts['reader_train'],'validation_panel':sum(per_problem[pid] for pid in selected),
            'final':sum(counts[s] for s in ['reader_val','cal','test']),'split_rows':counts,'validation_problem_limit':validation_limit}
    return result


class Controller:
    def __init__(self,c):
        self.lock=threading.RLock();self.reserved_devices={}
        self.c=c;self.output=Path(c['output']);self.output.mkdir(parents=True,exist_ok=True)
        self.statepath=self.output/c.get('pipeline_filename','pipeline.json');self.state=json.loads(self.statepath.read_text()) if self.statepath.exists() else {'steps':{},'skipped':[],'models':{}}
        identity=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        if self.state.get('configuration_sha256',identity)!=identity:raise ValueError('Controller scientific config changed on resume')
        self.state['configuration_sha256']=identity
        (self.output/'configs').mkdir(exist_ok=True)
    def save(self):
        with self.lock:write_json(self.statepath,self.state)
    def skip(self,name,reason):
        row={'name':name,'reason':reason}
        if row not in self.state['skipped']:self.state['skipped'].append(row)
        self.save()
    def used(self):return sum(x['gpu_hours'] for x in hours() if x['section']=='CD')
    def acquire_device(self,name,c):
        preferred=int(c['device'].split(':')[1]);required=c.get('min_free_gib',14 if c.get('kind')=='b5b' else 6)+1
        while True:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            available={int(row.split(',')[0]):int(row.split(',')[1]) for row in rows.splitlines()}
            with self.lock:
                candidates=[d for d,m in available.items() if m>=required*1024 and d not in self.reserved_devices.values()]
                if candidates:
                    selected=preferred if preferred in candidates else max(candidates,key=lambda d:available[d])
                    self.reserved_devices[name]=selected;c['device']=f'cuda:{selected}';return
            live(resource_wait={'job':name,'preferred_device':c['device'],'required_free_gib':required,'free_mib':available})
            time.sleep(10)
    def evaluation_reserve(self,extra_configs=()):
        # C3 adds many checkpoints. Reserve their inference before admitting
        # training, using completed C1 timing and known split sizes only.
        specs=self.state.get('native_inference_cost',{})
        native=0.
        for cc in [*(v['config'] for v in self.state['models'].values()),*extra_configs]:
            if cc['kind']=='b5b':continue
            sources=list(self.c['sources']) if cc['kind']=='multi' else [cc['source']]
            rate=specs.get('b3' if cc['kind']=='b3' else 'lite',{}).get('seconds_per_example')
            if rate is None:continue
            for source in sources:
                count=self.state['native_inference_examples'][source]
                if cc.get('case','').startswith('single_gpt2_'):count+=self.state.get('unseen_inference_examples',0)
                native+=(rate*count+90)/3600*1.5
        qwen=self.state.get('additional_evaluation_reserved_hours',0)+sum(self.state.get('qwen_inference_hours_by_source',{}).get(cc['source'],self.state.get('qwen_inference_hours_per_model',0)) for cc in extra_configs if cc['kind']=='b5b')
        return max(self.c['evaluation_reserve_hours'],native)+qwen+self.state.get('unseen_evaluation_reserved_hours',0)
    def admit(self,name,estimate,reserve=None,extra_configs=()):
        reserve=self.evaluation_reserve(extra_configs) if reserve is None else reserve
        self.state.setdefault('cost_admissions',[]).append({'name':name,'used_gpu_hours':self.used(),'training_estimate_gpu_hours':estimate,'inference_reserved_gpu_hours':reserve})
        self.save()
        if self.used()+estimate+reserve>=26.8:
            self.skip(name,'Cost-only estimate '+str(estimate)+' GPUh plus mandatory reserve '+str(reserve)+' exceeds remaining CD budget')
            return False
        return True
    def execute(self,name,script,c):
        old=self.state['steps'].get(name)
        if old:
            oldc=load_config(old['config'])
            if {k:v for k,v in oldc.items() if k!='device'}!={k:v for k,v in c.items() if k!='device'}:raise ValueError('Completed/failed child config changed: '+name)
        if old and old.get('status')=='completed':
            if 'device' in oldc:c['device']=oldc['device']
            return old['run']
        if old and old.get('status')=='running':
            run=self.find_run(old['log'])
            if run and (Path(run)/'metrics.json').exists():
                m=json.loads((Path(run)/'metrics.json').read_text())
                if m.get('status')=='completed':old.update(status='completed',run=run);self.save();return run
            raise RuntimeError('A prior child is still active/interrupted; explicit resume audit required: '+name)
        attempt=old.get('attempt',1)+1 if old else 1
        suffix='' if attempt==1 else f'_attempt{attempt}'
        path=self.output/'configs'/f'{name}{suffix}.yaml';log=Path('reports')/f'M6_{name}{suffix}.log'
        if log.exists():
            # A standalone run may already own this name. Keep its exact log.
            serial=1
            while log.with_name(log.stem+f'_controller{serial}.log').exists():serial+=1
            log=log.with_name(log.stem+f'_controller{serial}.log')
        gpu=script in ['scripts/m6_source.py','scripts/m6_train.py','scripts/m6_evaluate.py']
        if gpu:self.acquire_device(name,c)
        text=yaml.safe_dump(c,sort_keys=False,allow_unicode=True)
        if path.exists() and path.read_text()!=text:raise ValueError('Immutable child config changed: '+name)
        path.write_text(text)
        with self.lock:
            if old:self.state.setdefault('previous_attempts',{}).setdefault(name,[]).append(old)
            self.state['steps'][name]={'status':'running','script':script,'config':str(path),'log':str(log),'attempt':attempt};self.save()
        env={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1',**self.c.get('runtime_env',{})}
        try:
            with log.open('w') as stream:
                proc=subprocess.Popen([str(PY),script,str(path)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                while proc.poll() is None:time.sleep(5)
        finally:
            with self.lock:self.reserved_devices.pop(name,None)
        run=self.find_run(log)
        if proc.returncode or not run:
            with self.lock:self.state['steps'][name].update(status='failed',run=run,exit_code=proc.returncode);self.save()
            raise RuntimeError('M6 child failed: '+name+'; '+str(log))
        metrics=json.loads(Path(run,'metrics.json').read_text())
        if metrics.get('status')!='completed':raise RuntimeError('Process exit does not prove completed metrics: '+name)
        with self.lock:self.state['steps'][name].update(status='completed',run=run);self.save()
        return run
    @staticmethod
    def find_run(log):
        if not Path(log).exists():return None
        for line in Path(log).read_text(errors='replace').splitlines():
            if line.startswith(str(ROOT/'runs')+'/') and ' ' not in line.strip():return line.strip()
        return None
    def wait_existing(self,path):
        while True:
            m=json.loads(Path(path,'metrics.json').read_text())
            if m.get('status')=='completed':return path
            if m.get('status') in ['failed','error']:raise RuntimeError('Required prior M6 run failed: '+path)
            time.sleep(10)
    def train_one(self,key,c):
        c={**c,'case':key};run=self.execute('train_'+key,'scripts/m6_train.py',c)
        info=json.loads(Path(c['output'],'trained',f'{key}_{c["seed"]}.json').read_text())
        with self.lock:self.state['models'][key]={'run':run,'checkpoint':info['checkpoint'],'config':c};self.save()
        return run
    def train_batch(self,tasks):
        tasks=[(key,c) for key,c in tasks if key not in self.state['models']]
        devices=self.c['reader_devices'];queues=[[] for _ in devices]
        for i,(key,c) in enumerate(tasks):queues[i%len(devices)].append((key,{**c,'device':devices[i%len(devices)]}))
        def worker(queue):
            for key,c in queue:self.train_one(key,c)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            for f in [pool.submit(worker,q) for q in queues]:f.result()

    def generate(self,base,phase):
        devices=self.c['source_devices'];tasks=[]
        for i,device in enumerate(devices):
            cc={**base,'device':device,'generation_shards':len(devices),'generation_shard':i}
            tasks.append((f'source_generate_{phase}_{i}',cc))
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            for f in [pool.submit(self.execute,key,'scripts/m6_source.py',cc) for key,cc in tasks]:f.result()
        self.execute('source_merge_'+phase,'scripts/m6_merge_source.py',base)

def final_stages(ctl,c):
    index={};jobs=[]
    for key,info in ctl.state['models'].items():
        cc=info['config'];sources=list(c['sources']) if cc['kind']=='multi' else [cc['source']]
        for source in sources:
            ec={**cc,'stage':'evaluate','model_key':key,'source':source,'checkpoint':info['checkpoint'],'eval_lock':str(Path(c['output'])/'training_lock.json'),'eval_splits':['reader_val','test'],'eval_batch':16 if cc['kind']=='b5b' else 512,'device':c['reader_devices'][0]}
            if key.startswith('single_gpt2_'):
                ec['eval_splits']+=['unseen_train','unseen_eval'];ec['extra_eval']={s:{'root':'data/M4_v0/unseen','qa':'data/M4_v0/qa'} for s in ['unseen_train','unseen_eval']}
            if key.startswith('single_llama_') and ctl.state.get('optional_unseen',{}).get('status')=='completed':
                data=ctl.state['optional_unseen']['data'];ec['eval_splits']+=['unseen_train','unseen_eval']
                ec['extra_eval']={s:{'root':data['root'],'qa':data['qa']} for s in ['unseen_train','unseen_eval']}
            jobs.append((key,source,ec))
    # All scientific choices/checkpoint hashes are already locked. Independent
    # inference jobs can share the existing device scheduler; serialize only
    # the completed-index write so a source cannot overwrite another source.
    queues=[[] for _ in c['reader_devices']]
    for i,(key,source,ec) in enumerate(jobs):
        queues[i%len(queues)].append((key,source,{**ec,'device':c['reader_devices'][i%len(queues)]}))
    def worker(queue):
        for key,source,ec in queue:
            er=ctl.execute('eval_'+key+'_'+source,'scripts/m6_evaluate.py',ec)
            with ctl.lock:
                index.setdefault(key,{})[source]=er
                write_json(Path(c['output'])/'evaluation_index.json',index)
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        for future in [pool.submit(worker,q) for q in queues]:future.result()
    sc={'milestone':'M6','stage':'shift','output':c['output'],'seeds':c['seeds'],'evaluation_index':str(Path(c['output'])/'evaluation_index.json'),'eval_lock':str(Path(c['output'])/'training_lock.json')}
    ctl.execute('shift','scripts/m6_shift.py',sc)
    # CPU final report and tests are also required; a launch is never success.
    rc={**sc,'stage':'report','pipeline':str(ctl.statepath),'sources':c['sources']}
    ctl.execute('report','scripts/m6_report.py',rc)


def optional_unseen(ctl,c):
    """D has priority over C2/C4. Reader inference stays behind the final lock."""
    from clr.utils.m6_unseen import estimate
    policy_path=Path('configs/M6_llama_unseen_budget.yaml');policy=load_config(policy_path)
    if 'passed' not in Path(policy['tests_log']).read_text():raise ValueError('Optional D priority/context tests have not passed')
    path=Path(c['output'])/'llama_unseen_budget.json'
    decision=json.loads(path.read_text()) if path.exists() else None
    if decision is None:
        costs=estimate(policy,json.loads(Path(c['output'],'plan.json').read_text()),ctl.state['native_inference_cost'])
        used=ctl.used();reserved=ctl.evaluation_reserve()
        admitted=ctl.admit('D_llama_unseen',costs['estimated_additional_gpu_hours'],reserve=reserved)
        decision={**costs,'policy':str(policy_path),'basis':'source smoke, C1 reader_val timing and frozen input counts only',
            'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest(),'policy_config':policy,
            'used_CD_gpu_hours':used,'remaining_CD_gpu_hours':27-used,'existing_inference_reserved_gpu_hours':reserved,
            'guard_gpu_hours':policy['gpu_hours_guard'],'status':'admitted_source_generation_pending' if admitted else 'not_admitted_insufficient_CD_budget'}
        write_json(path,decision)
    if decision['status']=='admitted_source_generation_pending':
        ec={'milestone':'M6','stage':'unseen_source_extension','section':'CD','output':str(Path(c['output'])/'llama_unseen'),
            'source_config':c['source_config'],'questions_config':c['questions_config'],'source_plan':str(Path(c['output'])/'plan.json'),
            'parent_pipeline':str(ctl.statepath),'training_lock':str(Path(c['output'])/'training_lock.json'),
            'input_plan':policy['input_plan'],'reference_qa':policy['reference_qa'],'qa_smoke_log':policy['qa_smoke_log'],
            'splits':policy['splits'],'source_devices':c['source_devices'],'runtime_env':c.get('runtime_env',{}),'cost_policy':policy}
        live(stage='D_optional_llama_source',optional_llama_unseen_cost=decision)
        run=ctl.execute('optional_llama_unseen_source','scripts/m6_unseen_source.py',ec)
        completion=json.loads(Path(ec['output'],'completion.json').read_text())
        if completion['status']!='completed':raise ValueError('Optional D source data incomplete')
        decision.update(status='completed',production_run=run,data=completion);write_json(path,decision)
    ctl.state['optional_unseen']=decision
    ctl.state['unseen_evaluation_reserved_hours']=decision['six_reader_inference_estimate_gpu_hours'] if decision['status']=='completed' else 0
    ctl.save()

def main(c):
 with recorded(c) as controller_run:
    print(controller_run.path,flush=True);snapshot(controller_run,{**c,'section':'CD'})
    start=time.monotonic();ctl=Controller(c);ctl.state.update(status='running',controller_run=str(controller_run.path));ctl.save()
    try:
        if 'passed' not in Path(c['tests_log']).read_text():raise ValueError('M6 unit checks have not passed')
        for gate in [c['source_smoke'],c['training_smoke'],c['shift_smoke']]:
            if json.loads(Path(gate,'metrics.json').read_text()).get('status')!='completed':raise ValueError('Required smoke not completed: '+gate)
        if (Path(c['output'])/'training_lock.json').exists():
            locked=json.loads(Path(c['output'],'training_lock.json').read_text())
            if not locked.get('all_training_locked'):raise ValueError('Incomplete existing inference lock')
            for key,info in locked['models'].items():
                if hashlib.sha256(Path(info['checkpoint']).read_bytes()).hexdigest()!=locked['checkpoint_sha256'][key]:raise ValueError('Checkpoint changed after lock: '+key)
            ctl.state['models']=locked['models'];ctl.state['optional_unseen']=locked.get('optional_unseen',ctl.state.get('optional_unseen',{}));ctl.save()
            final_stages(ctl,c)
            ctl.state['status']='completed';ctl.save()
            controller_run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start)
            return
        rq1=ctl.wait_existing(c['rq1_run'])
        ctl.execute('rq1_report','scripts/m6_rq1_report.py',{'milestone':'M6','stage':'rq1_report','output':c['output'],'search_run':rq1})
        noise=ctl.wait_existing(c.get('noise_selection_run',c['noise_run']));live(stage='B_noise_completed',noise_run=noise,pipeline=str(ctl.statepath))
        source_smoke=c['source_smoke']
        if c.get('selected_sigma_smoke_config'):
            selected=json.loads(Path(noise,'selection.json').read_text())['selected'];sc=load_config(c['selected_sigma_smoke_config']);sc.update(sigma=selected['sigma'],sigmas=[selected['sigma']])
            source_smoke=ctl.execute('source_selected_sigma_smoke','scripts/m6_source.py',sc)
        base=load_config(c['source_config']);base.update(mode='full',noise_run=noise,stage='plan',smoke_run=source_smoke)
        ctl.execute('source_plan','scripts/m6_source.py',base)
        base.update(stage='generate',splits=['reader_val','val','cal','test'])
        # Explicit B authorization: production test dump/labels only, after source
        # protocol lock; never a smoke, tuning result or intermediate reader eval.
        ctl.generate(base,'fixed')
        ctl.execute('source_refine_train_plan','scripts/m6_refine_plan.py',base)
        base.update(splits=['reader_train'])
        ctl.generate(base,'train')
        qa=load_config(c['questions_config']);qa.update(mode='full',root=c['sources']['llama']['root'],qa_root=c['sources']['llama']['qa'],splits=['reader_train','reader_val','val','cal','test'])
        ctl.execute('questions_full','scripts/m6_questions.py',qa)
        ctl.execute('data_audit','scripts/m6_audit.py',{**qa,'stage':'audit'})
        prep={'milestone':'M6','stage':'prepare_references','sources':c['sources'],'output':c['output'],'m3_run':c['m3_run'],'m41_runs':c['m41_runs']}
        ctl.execute('prepare','scripts/m6_prepare.py',prep)
        template=load_config(c['training_config']);template.update(mode='full',sources=c['sources'],smoke_run=c['training_smoke'],validation_split='reader_val',output=c['output'])
        refs=json.loads(Path(c['output'],'reference_checkpoints.json').read_text())
        for key,ck in refs.items():
            parts=key.split('_');cc={**template,'case':key,'source':'gpt2','kind':parts[2],'seed':int(parts[3]),'n_labels':None}
            ctl.state['models'][key]={'checkpoint':ck,'config':cc,'reused_frozen_reference':True}
        ctl.save();live(stage='C1_training',pipeline=str(ctl.statepath))
        jobs=[]
        for sd in c['seeds']:
            for kind in ['lite','b3']:
                key=f'single_llama_{kind}_{sd}';cc={**template,'source':'llama','kind':kind,'seed':sd,'n_labels':None,'epochs':10 if kind=='lite' else 8,'min_epochs':8,'patience':1,'validation_every':1,'validation_problem_limit':None}
                jobs.append((key,cc))
        ctl.train_batch(jobs)
        # No test predictions/metrics enter this cost estimate. A whole
        # reader_val pass supplies timings; frozen plan sizes supply workload.
        llama_plan=json.loads(Path(c['output'],'plan.json').read_text())
        gpt_plan=json.loads(Path(base['gpt2_plan']).read_text())
        rows_per_problem=0.;rates={}
        for kind in ['lite','b3']:
            measurements=[]
            for sd in c['seeds']:
                root=Path(ctl.state['models'][f'single_llama_{kind}_{sd}']['run']);m=json.loads((root/'metrics.json').read_text())
                measurements.append(m['validation_seconds']/max(1,m['validation_examples']))
                if kind=='lite':
                    validations=len(json.loads((root/'history.json').read_text()))
                    rows_per_problem=max(rows_per_problem,m['validation_examples']/max(1,validations)/len(llama_plan['splits']['reader_val']))
            rates[kind]={'seconds_per_example':max(measurements),'basis':'complete C1 validation; maximum across three seeds'}
        ctl.state['native_inference_cost']=rates
        ctl.state['native_inference_examples']={s:rows_per_problem*sum(len(plan['splits'][split]) for split in ['reader_val','cal','test']) for s,plan in [('gpt2',gpt_plan),('llama',llama_plan)]}
        ctl.state['unseen_inference_examples']=rows_per_problem*1993
        ctl.save()
        # Full-N scratch baselines are C1; only the adapter-transfer arm adds a job.
        jobs=[]
        for a,b in [('gpt2','llama'),('llama','gpt2')]:
          for n in [100,300,1000,3000,None]:
           for sd in c['seeds']:
            for kind in (['transfer'] if n is None else ['transfer','lite','b3']):
                key=f'transfer_{a}_to_{b}_{kind}_{n or "full"}_{sd}'
                cc={**template,'source':b,'kind':kind,'seed':sd,'donor':a,'donor_checkpoint':ctl.state['models'][f'single_{a}_lite_{sd}']['checkpoint'],'n_labels':n,'epochs':10 if n is None else 100,'min_epochs':8 if n is None else 10,'validation_every':1 if n is None else 5,'validation_problem_limit':None if n is None else 128,'patience':1 if n is None else 3}
                jobs.append((key,cc))
        live(stage='C3_training',required_training_jobs=len(jobs))
        # Estimate from complete C1 throughput, without looking at test. Keep
        # the mandatory RQ4/evaluation reserve even if a costly arm cannot fit.
        small=[j for j in jobs if j[1]['n_labels'] is not None and j[0] not in ctl.state['models']]
        full=[j for j in jobs if j[1]['n_labels'] is None and j[0] not in ctl.state['models']]
        c1={sd:json.loads(Path(ctl.state['models'][f'single_llama_lite_{sd}']['run'],'metrics.json').read_text()) for sd in c['seeds']}
        maxc=max(c1.values(),key=lambda m:m['gpu_hours'])
        train_per_label=maxc['training_seconds']/max(1,sum(maxc['examples']['llama'].values()))
        valid_per_label=maxc['validation_seconds']/max(1,maxc['validation_examples'])
        # Small panels: 100 maximum epochs and <=20 validations of fixed128
        # reader_val problems. Include CPU bank/load overhead conservatively.
        small_estimate=sum((train_per_label*cc['n_labels']*cc['epochs']*1.5+valid_per_label*128*120*20+90)/3600 for _,cc in small)*1.2
        if ctl.used()+small_estimate+ctl.evaluation_reserve([cc for _,cc in small])<26.8:ctl.train_batch(small)
        else:
            # Admit complete N/direction/three-seed comparison families only.
            for a,b in [('gpt2','llama'),('llama','gpt2')]:
                for n in [100,300,1000,3000]:
                    group=[j for j in small if j[1]['donor']==a and j[1]['source']==b and j[1]['n_labels']==n]
                    est=sum((train_per_label*n*100*1.5+valid_per_label*128*120*20+90)/3600 for _ in group)*1.2
                    if ctl.admit(f'C3_{a}_to_{b}_N{n}',est,extra_configs=[cc for _,cc in group]):ctl.train_batch(group)
        for a,b in [('gpt2','llama'),('llama','gpt2')]:
            group=[j for j in full if j[1]['donor']==a]
            if not group:continue
            # Full GPT2 has 20000 problems; LLaMA count was cost-locked in B.
            ratio=20000/len(json.loads(Path(c['output'],'plan.json').read_text())['splits']['reader_train']) if b=='gpt2' else 1
            estimate=max(m['gpu_hours'] for m in c1.values())*ratio*1.5*len(group)
            if ctl.admit(f'C3_{a}_to_{b}_full',estimate,extra_configs=[cc for _,cc in group]):ctl.train_batch(group)
        optional_unseen(ctl,c)
        # Reserve the configured mandatory reader evaluation/RQ4 cost before
        # admitting C2/C4; estimates are based only on measured wall-time costs.
        costs=[json.loads(Path(v['run'],'metrics.json').read_text())['gpu_hours'] for k,v in ctl.state['models'].items() if k.startswith('single_llama_lite')]
        multi_estimate=max(costs)*3*3.5*1.2
        jobs=[(f'multi_lite_{sd}',{**template,'kind':'multi','source':'llama','seed':sd,'n_labels':None,'epochs':10,'min_epochs':8,'patience':1,'validation_every':1,'validation_problem_limit':None}) for sd in c['seeds']]
        if not all(f'multi_lite_{sd}' in ctl.state['models'] for sd in c['seeds']) and ctl.admit('C2',multi_estimate,extra_configs=[cc for _,cc in jobs if f'multi_lite_{cc["seed"]}' not in ctl.state['models']]):
            ctl.train_batch(jobs)
        qwen=Path(c['qwen_path'])
        if all((qwen/f).exists() for f in ['config.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']):
            if c.get('b5b_smoke'):
                sr=c['b5b_smoke']
                if json.loads(Path(sr,'metrics.json').read_text()).get('status')!='completed':raise ValueError('B5b smoke not complete')
            else:
                smoke=load_config(c['training_config']);smoke.update(kind='b5b',case='smoke_b5b',qwen_path=str(qwen),batch=16,eval_batch=16,device=c['reader_devices'][0])
                sr=ctl.execute('b5b_smoke','scripts/m6_train.py',smoke)
            # A 15-problem val proxy is exactly 5% of val300. Measure a
            # representative batch stream instead of extrapolating 70 rows.
            if ctl.admit('C4_throughput_smoke',.03):
                smoke={**template,'mode':'smoke','kind':'b5b','source':'llama','case':'b5b_throughput_cost_only',
                    'seed':11,'qwen_path':str(qwen),'batch':16,'eval_batch':16,'validation_split':'val',
                    'smoke_problems':15,'n_labels':None,'epochs':1,'min_epochs':1,'patience':1,
                    'device':c['reader_devices'][0]}
                sr=ctl.execute('b5b_throughput_smoke','scripts/m6_train.py',smoke)
            sm=json.loads(Path(sr,'metrics.json').read_text());seconds_per_example=sm['training_seconds']/max(1,sum(sm['examples']['llama'].values()));inference_seconds_per_example=sm['validation_seconds']/sm['validation_examples']
            workload=qwen_workload(c['sources']);ctl.state['qwen_cost_workload']=workload
            ctl.state['qwen_inference_hours_by_source']={s:(inference_seconds_per_example*v['final']+90)/3600*1.2 for s,v in workload.items()}
            ctl.state['qwen_inference_hours_per_model']=max(ctl.state['qwen_inference_hours_by_source'].values())
            # Keep already admitted Qwen evaluation commitments in the budget
            # while later N families are considered before final inference.
            def reserve_qwen_inference():
                ctl.state['additional_evaluation_reserved_hours']=sum(ctl.state['qwen_inference_hours_by_source'][v['config']['source']] for v in ctl.state['models'].values() if v['config']['kind']=='b5b');ctl.save()
            reserve_qwen_inference()
            for n in [100,300,1000,3000,None]:
                if all(f'b5b_{source}_{n or "full"}_{sd}' in ctl.state['models'] for source in ['gpt2','llama'] for sd in c['seeds']):continue
                jobs=[]
                for source in ['gpt2','llama']:
                  for sd in c['seeds']:
                    key=f'b5b_{source}_{n or "full"}_{sd}';cc={**template,'source':source,'kind':'b5b','seed':sd,'qwen_path':str(qwen),'n_labels':n,'epochs':3 if n is None else 30,'min_epochs':1 if n is None else 10,'patience':3,'validation_every':1 if n is None else 5,'validation_problem_limit':128,'batch':16,'eval_batch':16,'smoke_run':sr}
                    jobs.append((key,cc))
                estimate=sum((seconds_per_example*(n if n is not None else workload[cc['source']]['train'])*cc['epochs']
                    +inference_seconds_per_example*workload[cc['source']]['validation_panel']*((cc['epochs']-1)//cc['validation_every']+1)+90)/3600*1.2 for _,cc in jobs)
                extra_inference=sum(ctl.state['qwen_inference_hours_by_source'][cc['source']] for _,cc in jobs)
                if not ctl.admit(f'C4_N{n or "full"}',estimate,reserve=ctl.evaluation_reserve()+extra_inference):continue
                ctl.train_batch(jobs)
                timings=[json.loads(Path(ctl.state['models'][key]['run'],'metrics.json').read_text()) for key,_ in jobs]
                seconds_per_example=max(m['training_seconds']/max(1,sum(sum(v.values()) for v in m['examples'].values())) for m in timings)
                inference_seconds_per_example=max(m['validation_seconds']/m['validation_examples'] for m in timings)
                ctl.state.setdefault('qwen_timing_refinements',[]).append({'n_labels':n,'training_seconds_per_example':seconds_per_example,
                    'inference_seconds_per_example':inference_seconds_per_example,'basis':'maximum across complete six-model validation/training timings; no performance metric'})
                ctl.state['qwen_inference_hours_by_source']={s:(inference_seconds_per_example*v['final']+90)/3600*1.2 for s,v in workload.items()}
                ctl.state['qwen_inference_hours_per_model']=max(ctl.state['qwen_inference_hours_by_source'].values())
                reserve_qwen_inference()
        else:ctl.skip('C4','Pinned Qwen3-1.7B download unavailable/incomplete; no substitute model')
        # All settings, optional omissions and exact checkpoint hashes lock now.
        lock={'all_training_locked':True,'models':ctl.state['models'],'checkpoint_sha256':{k:hashlib.sha256(Path(v['checkpoint']).read_bytes()).hexdigest() for k,v in ctl.state['models'].items()},'skipped':ctl.state['skipped'],'optional_unseen':ctl.state.get('optional_unseen',{}),'test_scope':'B/D final production source dumps authorized; no reader test predictions until this lock'}
        write_json(Path(c['output'])/'training_lock.json',lock);live(stage='final_evaluation',training_lock=str(Path(c['output'])/'training_lock.json'))
        final_stages(ctl,c)
        ctl.state['status']='completed';ctl.save()
        controller_run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0,wall_seconds=time.monotonic()-start)
    except Exception as exc:
        ctl.state['status']='blocked_or_failed';ctl.state['reason']=str(exc);ctl.save();live(stage='blocked_or_failed',reason=str(exc),pipeline=str(ctl.statepath),milestone_complete=False)
        controller_run.write_metrics(status='failed',gpu_hours=0,source_forward_calls=0,error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

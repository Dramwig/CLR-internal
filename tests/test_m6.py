import numpy as np
import pytest
import torch
from torch import nn
from transformers import LlamaConfig,LlamaForCausalLM
from clr.sources.codi_m6 import LlamaM6,prepare,rollout,early_exit
from clr.calib.m6 import weighted_threshold,coverage
from clr.models.m6 import NativeLite,TransferLite
from clr.models.clr_v1 import Lite
from clr.data.m6 import nested_panel

def test_native_dimensions_option_permutation_and_future_mask():
    from clr.models.clr_v0 import Serializer
    class Tok:
        cls_token_id=1;sep_token_id=2;pad_token_id=0
        def encode(self,s,**kw):return [3+ord(x)%20 for x in s]
    serializer=Serializer(Tok(),{'Q3':['pick']},max_length=64)
    row={'qtype':'Q3','template_id':0,'options':['11','2','300','4']};perm=[2,0,3,1]
    changed={**row,'options':[row['options'][i] for i in perm]}
    for dim in [768,2048]:
        torch.manual_seed(11);model=NativeLite(dim,hidden=16,layers=1).eval();z=torch.randn(1,7,dim);mask=torch.tensor([[1,1,1,0,0,0,0]],dtype=torch.bool)
        with torch.no_grad():
            original=model(serializer.batch([row]),z,mask,'Q3');other=model(serializer.batch([changed]),z,mask,'Q3')
            assert torch.equal(original[:,perm],other)
            z[:,3:]+=100
            assert torch.equal(original,model(serializer.batch([row]),z,mask,'Q3'))

def test_llama_cache_prefix_is_immutable_and_observations_are_outputs():
    torch.manual_seed(11);torch.set_num_threads(1)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.codi=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)).to(torch.bfloat16).eval()
            self.prj=nn.Linear(16,16).to(torch.bfloat16);self.pad_token_id=29;self.bot_id=30;self.eot_id=31;self.model_name='llama'
        def get_embd(self,*a):return self.codi.get_input_embeddings()
    class Tok:
        eos_token_id=2
        def encode(self,text):return [1,3,4]
        def decode(self,ids,**kw):return str(ids)
    source=LlamaM6.__new__(LlamaM6);source.device=torch.device('cpu');source.model=Model();source.tokenizer=Tok();source.forward_calls=0
    source.config=dict(family='llama',dtype='bfloat16',num_latent=6,max_new_tokens=4,layer=-1,position_ids='implicit_absolute',prefill_attention_mask=True,carry_attention_mask=False)
    p=dict(id='one',question='q',pad_to=5);pre=prepare(p,source);lengths=[x[0][0].shape[-2] for x in pre.caches]
    assert lengths==list(range(6,13))
    clean=early_exit(p,pre,6);out=rollout(p,pre,3,0,0,11)
    assert out['answer']==clean['answer']
    assert torch.equal(out['states'],torch.stack([x[0,0].cpu() for x in pre.states[3:]]))
    noisy=rollout(p,pre,3,.5,0,11)
    assert torch.equal(noisy['states'][0],pre.states[3][0,0])
    assert not torch.equal(noisy['feedback'][0],noisy['states'][0])
    assert [x[0][0].shape[-2] for x in pre.caches]==lengths
    with pytest.raises(ValueError):early_exit(dict(p,id='other'),pre,1)

def test_weighted_uniform_matches_split_and_infinite_test_mass():
    from clr.calib.m23 import conformal_threshold
    scores=np.linspace(.01,.99,100)
    for alpha in [.05,.1,.2]:
        assert weighted_threshold(scores,np.ones(100),[1],alpha)[0]==conformal_threshold(scores,alpha)
    assert np.isinf(weighted_threshold(scores,np.ones(100),[1e6],.1)[0])
    cov,size=coverage(np.array([[.8,.2]]),np.array([[1.,0]]),np.inf)
    assert cov[0]==1 and size[0]==2

def test_native_768_is_frozen_lite_and_transfer_only_adapter_updates():
    torch.manual_seed(11);old=Lite(hidden=16,layers=1);new=NativeLite(hidden=16,layers=1);new.load_state_dict(old.state_dict())
    assert all(torch.equal(old.state_dict()[k],v) for k,v in new.state_dict().items())
    transfer=TransferLite(new,32)
    assert {n.split('.')[0] for n,p in transfer.named_parameters() if p.requires_grad}=={'adapter'}
    before={k:v.clone() for k,v in transfer.body.state_dict().items()}
    x=torch.randn(2,32);loss=transfer.adapter(x).square().mean();loss.backward()
    assert transfer.adapter.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in transfer.body.parameters())
    assert all(torch.equal(before[k],v) for k,v in transfer.body.state_dict().items())

def test_nested_balanced_label_panels():
    rs=[dict(problem_id=str(i),state_ref={'t':i%7},template_id=0,options=[],replicate=0,qtype=q) for i in range(100) for q in ['Q1','Q2','Q3','Q4']]
    a=nested_panel(rs,100);b=nested_panel(rs,300)
    assert a==b[:100] and len(a)==100
    assert all(sum(r['qtype']==q for r in a)==25 for q in ['Q1','Q2','Q3','Q4'])
    from clr.data.m4.reader import require_split
    for phase in ['train','validate','calibrate']:
        with pytest.raises(ValueError):require_split('test',phase)


def test_unseen_plan_requires_complete_original_cohorts():
    from clr.utils.m6_unseen import input_rows
    plan={'splits':{'unseen_train':[{'id':'a','cohort_id':'c','equation':'2+3'}],
                    'unseen_eval':[{'id':'b','cohort_id':'c','equation':None}]}}
    result=input_rows(plan,{'c':{'a','b'}})
    assert result['unseen_train'][0]['cot']=='2+3'
    assert result['unseen_eval'][0]['cot']==''
    with pytest.raises(ValueError,match='cohort members'):
        input_rows(plan,{'c':{'a','b','removed'}})


def test_shift_includes_optional_llama_datasets_with_val_fixtures(tmp_path,monkeypatch):
    import json
    from contextlib import nullcontext
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import scripts.m6_shift as module
    run=type('Run',(),{'path':tmp_path,'write_metrics':lambda self,**kw:None})()
    monkeypatch.setattr(module,'recorded',lambda c:nullcontext(run))
    monkeypatch.setattr(module,'snapshot',lambda *args:None)
    index={}
    for kind in ['lite','b3']:
        for source in ['gpt2','llama']:index[f'single_{source}_{kind}_11']={source:str(tmp_path)}
    for q in ['Q1','Q2','Q3','Q4']:
        for split in ['cal','val','unseen_train','unseen_eval']:
            ids=([f'{pool}:{i}' for pool in ['svamp','asdiv'] for i in range(8)]
                 if split.startswith('unseen') else [f'val_proxy:{i}' for i in range(8)])
            rows=[{'problem_id':pid,'state_ref':{'traj_id':-1,'kind':'clean','t':t},
                'cal_part':'','options':['a','b','c','d'] if q=='Q3' else [],'template_id':0,
                'label':0,'label_soft':.5 if q=='Q1' else 0.,'bernoulli_outcomes':[0,1],
                'calibrated_logits':[1.,0.,0.,0.] if q=='Q3' else [1.,0.]} for pid in ids for t in [1,4]]
            pq.write_table(pa.Table.from_pylist(rows),tmp_path/f'gpt2_{q}_{split}.parquet')
    (tmp_path/'index.json').write_text(json.dumps(index));(tmp_path/'lock.json').write_text(json.dumps({'all_training_locked':True}))
    module.main({'mode':'smoke','milestone':'M6','stage':'shift','output':str(tmp_path),
        'seeds':[11],'evaluation_index':str(tmp_path/'index.json'),'eval_lock':str(tmp_path/'lock.json'),
        'proxy_sources':{'gpt2':'gpt2','llama':'gpt2'}})
    result=pd.read_csv(tmp_path/'shift'/'summary.csv')
    for source in ['gpt2','llama']:
        for pool in ['svamp','asdiv']:
            subset=result[result['shift']==f'dataset_{source}_gsm_to_{pool}']
            assert set(subset.qtype)=={'Q1','Q2','Q3','Q4'}
            assert set(subset.method)=={'split','weighted','mondrian_source_only','mondrian_target_cal'}


def test_sharded_merge_enforces_locked_problem_coverage(tmp_path):
    import json,hashlib
    from scripts.m6_merge_source import merge
    root=tmp_path/'dump';split=root/'val';split.mkdir(parents=True)
    plan={'sigma':1.5,'splits':{'val':[{'id':'p'}]}}
    planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan))
    chunk={'ids':['p'],'sigma':1.5,'plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),
      'manifest':[{'problem_id':'p','traj_id':-1,'t':t,'layer':16} for t in range(7)],
      'early':[{'problem_id':'p','t':t} for t in range(7)],
      'mc':[{'problem_id':'p','t':t,'policy':policy} for t in range(6) for policy in ['pi','pi_T']],
      'storage':{},'run':'fixture'}
    marker=split/'chunk_00000.json';marker.write_text(json.dumps({**chunk,'ids':['wrong']}))
    config={'output':str(tmp_path),'dump_root':str(root),'splits':['val']}
    with pytest.raises(ValueError,match='protocol/IDs'):merge(config)
    marker.write_text(json.dumps(chunk));result=merge(config)
    assert result['val']=={'problems':1,'states':7,'mc_panels':12}
    assert (split/'completed.json').exists()
    with pytest.raises(FileExistsError):merge(config)


def test_report_bernoulli_brier_preserves_sampling_variance():
    from scripts.m6_report import point_rows
    rows=[{'problem_id':'p','state_ref':{'t':0},'label_soft':.5,'calibrated_logits':[0.,0.]}]
    frame=point_rows(rows,'Q1')
    assert frame.accuracy.iloc[0]==.5
    assert frame.brier.iloc[0]==.25
    # Squared error to the MC estimate would incorrectly return zero.
    assert frame.nll.iloc[0]==pytest.approx(np.log(2))


def test_review_format_and_missing_b5b_remains_explicit(tmp_path,monkeypatch):
    import json
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.m6_report import write_reports
    monkeypatch.chdir(tmp_path);out=tmp_path/'data';(out/'shift').mkdir(parents=True);(out/'rq1').mkdir();(out/'analysis').mkdir();(tmp_path/'reports').mkdir()
    (out/'plan.json').write_text(json.dumps({'sigma':1.5,'splits':{'reader_train':[{'id':'p'}]}}))
    (out/'llama_unseen_budget.json').write_text(json.dumps({'remaining_CD_gpu_hours':1.,'estimated_additional_gpu_hours':8.,'guard_gpu_hours':.2}))
    audit_run=out/'audit_run';audit_run.mkdir()
    (audit_run/'audit.json').write_text(json.dumps({s:{'q4_numeric_marginals':{'total_variation':v}} for s,v in [('reader_train',.13),('reader_val',.198),('cal',0.),('test',0.)]}))
    pipeline=out/'pipeline.json';pipeline.write_text(json.dumps({'steps':{'data_audit':{'run':str(audit_run)}}}))
    shifts=[]
    for setting in ['source_gpt2_to_llama','dataset_gpt2_gsm_to_svamp','step_gpt2_early_to_late']:
        for q in ['Q3','Q4']:
            for method in ['split','weighted']:shifts.append({'model':'lite','qtype':q,'shift':setting,'method':method,'alpha':.1,'coverage':.88,'coverage_minus_target':-.02,'size':2.5 if q=='Q3' else 1.5})
    pd.DataFrame(shifts).to_csv(out/'shift'/'summary.csv',index=False)
    points=[{**{k:r[k] for k in ['model','qtype','shift','method','alpha']},'problem_id':pid,'coverage':r['coverage']} for r in shifts for pid in ['a','b','c']]
    pq.write_table(pa.Table.from_pylist(points),out/'shift'/'problem_results.parquet')
    pd.DataFrame([{'simultaneous_low':-.02}]).to_csv(out/'rq1'/'summary.csv',index=False)
    paired=pd.DataFrame([{'source':s,'split':split,'qtype':q,'metric':'accuracy','delta_lite_minus_b3':.03,'ci_low':.01,'ci_high':.05} for s in ['gpt2','llama'] for split in ['reader_val','test'] for q in ['Q3','Q4']])
    transfer=pd.DataFrame([{'direction':a+'->'+b,'split':'test','qtype':q,'n_labels':100,'baseline':'b3','ci_low':.01} for a,b in [('gpt2','llama'),('llama','gpt2')] for q in ['Q3','Q4']])
    agg=pd.DataFrame([{'source':s,'split':split,'qtype':q,'case':f'single_{s}_{kind}','kind':'clean','metric':'accuracy','mean':.6 if kind=='lite' else .57} for s in ['gpt2','llama'] for split in ['reader_val','test'] for q in ['Q3','Q4'] for kind in ['lite','b3']])
    write_reports({'output':str(out),'pipeline':str(pipeline)},{'skipped':[{'name':'C4_N100','reason':'budget'}]}, {},out/'analysis',agg,paired,transfer,[{'section':s,'gpu_hours':.1} for s in ['A','B','CD']])
    text=(tmp_path/'reports'/'M6_REVIEW.md').read_text()
    assert len(text.encode())<=8000
    assert [line[4] for line in text.splitlines() if line.startswith('## (')]==list('abcdef')
    assert '不能声称优于B5b' in text and '未执行' in text
    assert 'reader_train=0.13' in text and 'reader_val=0.198' in text
    recovery=pd.read_csv(out/'analysis'/'coverage_recovery.csv')
    assert (recovery.weighted_change_ci_low==0).all() and (recovery.weighted_change_ci_high==0).all()


def test_controller_preserves_failed_attempts_and_reserves_promised_inference(tmp_path,monkeypatch):
    import json
    import scripts.m6_pipeline as module
    from pathlib import Path
    monkeypatch.chdir(tmp_path);(tmp_path/'reports').mkdir();(tmp_path/'runs').mkdir();monkeypatch.setattr(module,'ROOT',tmp_path)
    calls=[]
    class Child:
        def __init__(self,argv,**kw):
            calls.append(argv);self.returncode=1 if len(calls)==1 else 0
            root=tmp_path/'runs'/f'fixture{len(calls)}';root.mkdir();(root/'metrics.json').write_text(json.dumps({'status':'failed' if self.returncode else 'completed'}))
            kw['stdout'].write(str(root)+'\n');kw['stdout'].flush()
        def poll(self):return self.returncode
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    c={'output':str(tmp_path/'out'),'evaluation_reserve_hours':3}
    ctl=module.Controller(c)
    (tmp_path/'reports'/'M6_fixture.log').write_text('prior standalone evidence\n')
    with pytest.raises(RuntimeError,match='child failed'):ctl.execute('fixture','fixture.py',{'stage':'fixture'})
    first=ctl.state['steps']['fixture']['log']
    assert Path(first).name=='M6_fixture_controller1.log'
    assert ctl.execute('fixture','fixture.py',{'stage':'fixture'}).endswith('fixture2')
    assert Path(first).exists() and Path('reports/M6_fixture.log').read_text()=='prior standalone evidence\n'
    assert ctl.state['previous_attempts']['fixture'][0]['status']=='failed'
    ctl.used=lambda:20.;ctl.state['additional_evaluation_reserved_hours']=4.
    assert not ctl.admit('another_qwen_family',.1)
    assert ctl.state['skipped'][-1]['name']=='another_qwen_family'
    before=ctl.evaluation_reserve();ctl.state['unseen_evaluation_reserved_hours']=.7
    assert ctl.evaluation_reserve()==pytest.approx(before+.7)
    with pytest.raises(ValueError,match='scientific config'):module.Controller({**c,'changed':True})


def test_qwen_cost_counts_exclude_held_templates_without_loading_labels(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.m6_pipeline import qwen_workload
    for split in ['reader_train','reader_val','cal','test']:
        # Only permitted cost metadata exists: loading any label/logit would fail.
        rows=[{'problem_id':'proxy_a','is_heldout_template':False},
              {'problem_id':'proxy_b','is_heldout_template':False},
              {'problem_id':'proxy_a','is_heldout_template':True}]
        pq.write_table(pa.Table.from_pylist(rows),tmp_path/f'{split}.parquet')
    result=qwen_workload({'fixture':{'qa':str(tmp_path)}})['fixture']
    assert result['train']==2 and result['validation_panel']==2 and result['final']==6


def test_cost_refinement_only_shortens_training_and_keeps_fixed_data(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import scripts.m6_refine_plan as module
    original={'sigma':1.5,'splits':{'reader_train':[{'id':str(i)} for i in range(1000)],'reader_val':[{'id':'rv'}],'val':[{'id':'v'}],'cal':[{'id':'c'}],'test':[{'id':'synthetic-final-fixture'}]},'cost_estimate':{'train_count':1000}}
    (tmp_path/'plan.json').write_text(json.dumps(original))
    root=tmp_path/'dump'
    for split in ['reader_val','val','cal','test']:
        p=root/split;p.mkdir(parents=True);(p/'completed.json').write_text(json.dumps({'sigma':1.5,'problem_count':1}))
    smoke=tmp_path/'smoke';smoke.mkdir();(smoke/'timing.json').write_text(json.dumps({'reader_train':[10.,10.]}))
    monkeypatch.setattr(module,'hours',lambda:[{'section':'B','gpu_hours':28.}])
    c={'output':str(tmp_path),'dump_root':str(root),'smoke_run':str(smoke),'cost_safety_factor':1.2}
    result=module.refine(c);new=json.loads((tmp_path/'plan.json').read_text())
    assert result['train_count']==570 and result['projected_gpu_hours']<=30
    assert new['splits']['reader_train']==original['splits']['reader_train'][:570]
    assert all(new['splits'][s]==original['splits'][s] for s in ['reader_val','val','cal','test'])
    assert json.loads((tmp_path/'plan_initial.json').read_text())==original
    (root/'reader_train').mkdir();(root/'reader_train'/'prior_attempt').write_text('evidence')
    with pytest.raises(ValueError,match='precede all reader_train'):module.refine(c)


def test_report_preserves_frozen_step_micro_accuracy_under_cluster_bootstrap():
    import pandas as pd
    from scripts.m6_report import cluster_estimate
    data=pd.DataFrame([{'problem_id':'a','t':0,'instances':9,'accuracy':1.},
      {'problem_id':'b','t':0,'instances':1,'accuracy':0.},
      {'problem_id':'a','t':1,'instances':1,'accuracy':.5},
      {'problem_id':'b','t':1,'instances':1,'accuracy':.5}])
    mean,ci=cluster_estimate(data,['accuracy'])
    assert mean[0]==pytest.approx(.7)
    assert ci.shape==(2,1) and ci[0,0]<=mean[0]<=ci[1,0]
    # A plain mean of problem averages would change this estimand to .5.


def test_final_statistics_fixture_clean_scope_and_three_seed_pairing(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.m6_report import collect
    lock={'models':{}};index={}
    for source in ['gpt2','llama']:
      for family in ['lite','b3']:
       for sd in [11,22,33]:
        key=f'single_{source}_{family}_{sd}';root=tmp_path/key;root.mkdir();index[key]={source:str(root)}
        lock['models'][key]={'config':{'seed':sd,'kind':family,'n_labels':None}}
        for split in ['reader_val','test']:
         for q in ['Q1','Q2','Q3','Q4']:
          rows=[]
          for pid in ['a','b','c']:
           for t in [0,1]:
            rows.append({'problem_id':pid,'state_ref':{'t':t,'kind':'clean'},'label':2 if q=='Q3' else 1,'label_soft':1.,'calibrated_logits':([0.,0.,1.,0.] if family=='lite' else [1.,0.,0.,0.]) if q=='Q3' else [0.,1.]})
          if split=='reader_val' and q in ['Q1','Q3']:
            # These deliberately wrong trajectory predictions cannot dilute
            # the clean M4.1 primary comparison.
            for pid in ['a','b','c']:rows.append({'problem_id':pid,'state_ref':{'t':1,'kind':'traj'},'label':2 if q=='Q3' else 1,'label_soft':1.,'calibrated_logits':[1.,0.,0.,0.] if q=='Q3' else [1.,0.]})
          pq.write_table(pa.Table.from_pylist(rows),root/f'{source}_{q}_{split}.parquet')
    for sd in [11,22,33]:
      for family in ['transfer','lite','b3']:
        key=f'transfer_gpt2_to_llama_{family}_100_{sd}';base=f'single_llama_{"b3" if family=="b3" else "lite"}_{sd}'
        index[key]={'llama':index[base]['llama']};lock['models'][key]={'config':{'seed':sd,'kind':family,'donor':'gpt2','n_labels':100}}
      key=f'multi_lite_{sd}';index[key]={source:index[f'single_{source}_lite_{sd}'][source] for source in ['gpt2','llama']};lock['models'][key]={'config':{'seed':sd,'kind':'multi','n_labels':None}}
    out=tmp_path/'analysis';out.mkdir();frame,agg,paired,transfer=collect({},lock,index,out)
    row=paired[(paired.source=='llama')&(paired.split=='reader_val')&(paired.qtype=='Q3')&(paired.metric=='accuracy')].iloc[0]
    assert row.delta_lite_minus_b3==1 and row.ci_low==1 and row.ci_high==1
    assert set(frame[(frame.qtype=='Q1')&(frame.split=='reader_val')]['kind'])=={'clean','traj','all'}
    assert set(transfer.baseline)=={'lite','b3'}
    chosen=transfer[(transfer.baseline=='b3')&(transfer.qtype=='Q3')]
    assert (chosen.delta_transfer_minus_scratch==1).all()
    import pandas as pd
    joint=pd.read_csv(out/'multisource_paired.csv');assert (joint.delta_joint_minus_single==0).all()


def test_controller_selects_live_free_device_without_changing_experiment(tmp_path,monkeypatch):
    import scripts.m6_pipeline as module
    ctl=module.Controller({'output':str(tmp_path/'out'),'evaluation_reserve_hours':3})
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**k:'0, 19000\n1, 2700\n2, 17500\n')
    a={'device':'cuda:1','kind':'b5b','batch':16};ctl.acquire_device('first',a)
    b={'device':'cuda:0','kind':'lite','batch':1024};ctl.acquire_device('second',b)
    assert a=={'device':'cuda:0','kind':'b5b','batch':16}
    assert b=={'device':'cuda:2','kind':'lite','batch':1024}
    assert len(set(ctl.reserved_devices.values()))==2


def test_preregistered_eight_point_noise_rule_and_fallback():
    from scripts.m6_choose_noise import choose
    rows=[{'sigma':s,'drop':.01,'nondegenerate':.1,'eligible':False} for s in [.5,1,1.5,2,3,4,6,8]]
    rows[3]['nondegenerate']=.2
    rows[4].update(nondegenerate=.35,drop=.04);rows[5].update(nondegenerate=.35,drop=.02)
    # Ties are resolved by sigma alone, not by the smaller accuracy drop.
    assert choose(rows)['sigma']==3 and not choose(rows)['degraded']
    rows[4].update(nondegenerate=.25);rows[5].update(nondegenerate=.26)
    assert choose(rows)['sigma']==4 and choose(rows)['degraded']
    for r in rows[4:]:r['drop']=.05
    assert choose(rows)['sigma']==2 and choose(rows)['degraded']
    with pytest.raises(ValueError,match='eight'):choose(rows[:-1])


def test_pending_native_checkpoints_reserve_inference_before_training(tmp_path):
    from scripts.m6_pipeline import Controller
    ctl=Controller({'output':str(tmp_path),'evaluation_reserve_hours':3,'sources':{'gpt2':{},'llama':{}}})
    ctl.used=lambda:20.
    ctl.state.update(native_inference_cost={'lite':{'seconds_per_example':.01}},native_inference_examples={'llama':1000000,'gpt2':2000000})
    candidates=[{'kind':'transfer','source':'llama'}]*2
    assert ctl.evaluation_reserve()==3
    assert ctl.evaluation_reserve(candidates)>8
    assert not ctl.admit('C3_family',.1,extra_configs=candidates)
    assert ctl.state['cost_admissions'][-1]['inference_reserved_gpu_hours']>8


def test_original_cohort_audit_retains_removed_context_and_exact_id_prefix():
    import copy
    from scripts.m6_context_audit import audit_contexts
    row={'id':'kept','question':'ab','cohort_id':'M0_official_test:0','pad_to':6,'left_padding':4}
    parent={'splits':{'val':[row, {**row,'id':'later'}]}}
    plan={'splits':{'val':[copy.deepcopy(row)]}}
    original=[{'index':0,'problem_id':'removed','question':'abcdef'}]
    assert audit_contexts(parent,plan,original,list)['mismatches']==0
    plan['splits']['val'][0].update(pad_to=2,left_padding=0)
    with pytest.raises(ValueError,match='tokenizer width'):
        audit_contexts(parent,plan,original,list)
    plan['splits']['val'][0]={**row,'id':'later'}
    with pytest.raises(ValueError,match='frozen prefix'):
        audit_contexts(parent,plan,original,list)


def test_final_inference_parallel_index_and_analysis_barrier(tmp_path):
    import json,threading,time
    from scripts.m6_pipeline import final_stages
    class Controller:
        def __init__(self):
            self.lock=threading.RLock();self.active=0;self.peak=0;self.done=set();self.analysis=[]
            self.statepath=tmp_path/'pipeline.json'
            self.state={'models':{f'fixture_{i}':{'config':{'kind':'multi' if i==0 else 'lite','source':'llama'},'checkpoint':'fixture.pt'} for i in range(5)}}
        def execute(self,name,script,c):
            if name.startswith('eval_'):
                with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
                time.sleep(.02)
                with self.lock:self.active-=1;self.done.add(name)
                return name+'_run'
            assert self.active==0 and len(self.done)==6
            index=json.loads((tmp_path/'evaluation_index.json').read_text())
            assert len(index)==5 and set(index['fixture_0'])=={'gpt2','llama'}
            self.analysis.append(name)
            return name+'_run'
    ctl=Controller()
    final_stages(ctl,{'output':str(tmp_path),'reader_devices':['cuda:0','cuda:1','cuda:4'],
                     'sources':{'gpt2':{},'llama':{}},'seeds':[11,22,33]})
    assert ctl.peak==3 and ctl.analysis==['shift','report']

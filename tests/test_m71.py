import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from clr.audit.features import KVFeatures,make_projection,probe_summary
from clr.data.m71 import examples,scan_partition,selected_layers


def test_projection_seed_orthogonality():
    matrix=make_projection(16,8,11)
    assert torch.equal(matrix,make_projection(16,8,11))
    assert not torch.equal(matrix,make_projection(16,8,22))
    torch.testing.assert_close(matrix.T@matrix,torch.eye(8),atol=1e-6,rtol=1e-6)


def test_segment_means_head_order_padding_terminal_tokens_and_gradient():
    torch.manual_seed(11)
    k=torch.randn(1,2,12,2,requires_grad=True);v=torch.randn(1,2,12,2,requires_grad=True)
    mask=torch.tensor([[False,False]+[True]*10]);meta={'token_count':10,'physical_offset':2,'record_spans':[[0,3],[3,10]]}
    f=KVFeatures(torch.eye(8),[0]);result=f([(k,v)],mask,meta)
    raw=torch.cat([k[0].permute(1,0,2).flatten(1),v[0].permute(1,0,2).flatten(1)],-1)
    expected=torch.cat([torch.stack([raw[2:5].mean(0),raw[5:12].mean(0)]),raw[4:12]],0)
    torch.testing.assert_close(result.values,expected)
    assert result.mask.all() and result.values.shape==(10,8)
    result.values.square().sum().backward()
    assert k.grad[:,:,2:].abs().sum()>0 and v.grad[:,:,2:].abs().sum()>0
    assert k.grad[:,:,:2].abs().sum()==0 and v.grad[:,:,:2].abs().sum()==0
    # Replacing oracle/commitment values cannot alter the numerical transform.
    with torch.no_grad():
        second=f([(k,v)],mask,{**meta,'e_priv':3,'label':999,'commitment':'ignored'})
    assert torch.equal(second.values,result.values)


def test_truncation_zeroing_and_fallback_are_raw_operations():
    k=torch.ones(1,1,12,4);v=2*k;mask=torch.ones(1,12,dtype=torch.bool)
    meta={'token_count':12,'record_spans':[[0,2],[2,6],[6,12]],'critical_records':[1]}
    f=KVFeatures(torch.eye(8),[0],critical_tokens=True)
    result=f([(k,v)],mask,meta)
    assert int((result.kinds==2).sum())==4
    assert int((result.kinds==0).sum())==1
    truncated=KVFeatures(torch.eye(8),[0])([(k[:,:,:6],v[:,:,:6])],mask[:,:6],meta)
    assert not truncated.mask[2] and not truncated.values[2].any()
    zero=f([(k*0,v*0)],mask,meta)
    assert zero.mask.all() and zero.values.abs().sum()==0


@pytest.mark.parametrize('kind',['CLR','B3','linear','B0'])
def test_m71_option_order_and_typed_question_label_isolation(kind):
    from clr.audit.m71_reader import Serializer
    from clr.models.m71 import Reader
    class Tokenizer:
        cls_token_id=1;sep_token_id=2;pad_token_id=0
        def encode(self,text,add_special_tokens=False):return [10+sum(map(ord,x))%1000 for x in text.split()]
    torch.set_num_threads(2);torch.manual_seed(11)
    serializer=Serializer(Tokenizer());model=Reader(kind).eval()
    original={'task':'A4','question_text':'Which private team is encoded?',
              'options':['Amber','Birch','Cedar','Dune'],'label':0}
    permutation=[2,0,3,1];other={**original,'options':[original['options'][i] for i in permutation],'label':999}
    if kind in ['B3','linear']:source=torch.randn(1,4*9*512)
    else:source={'values':torch.randn(1,12,512),'mask':torch.ones(1,12,dtype=torch.bool),
                 'layers':torch.zeros(1,12,dtype=torch.long),'segments':torch.arange(12)[None],
                 'kinds':torch.zeros(1,12,dtype=torch.long)}
    with torch.no_grad():
        a=model(serializer.batch([original],'cpu'),source,'A4')
        b=model(serializer.batch([other],'cpu'),source,'A4')
    assert torch.equal(a[:,permutation],b)


def test_layer_rule_and_group_scan_partition():
    assert selected_layers(35)==[8,17,26,35]
    fit,val=scan_partition(['a','b','c','d','e'],3)
    assert len(fit)==3 and len(val)==2 and not fit&val
    assert fit|val==set(range(5))
    from clr.data.m71 import fixed_hash_subset
    assert fixed_hash_subset(['c','a','b'],2)==fixed_hash_subset(['b','c','a'],2)


def test_expanded_generator_splits_and_frozen_recipe():
    c={'idea_a':'/home/me/PaperField/LatDefer','data_seed':20261107,
       'counts':{'reader_train':8000,'reader_val':1000,'cal_A':500,'cal_B':1000,'test':2000}}
    groups=[]
    for split in ['reader_train','reader_val','cal_A','cal_B']:
        rows=list(examples(c,split,3));ids={e.meta['group_id'] for e in rows}
        assert len(ids)==3 and all(not ids&other for other in groups)
        assert all(e.meta['alpha']==1 and e.meta['priv_len'] in [12,48,128] for e in rows)
        groups.append(ids)


def test_controls_precede_compression_and_bound_noise():
    from clr.sources.kv_relay import verify_idea_a
    verify_idea_a('/home/me/PaperField/LatDefer')
    from latdef.inject.cache import PrefixKV
    from clr.audit.m71_controls import raw_control
    torch.manual_seed(17)
    pairs=tuple((torch.randn(2,1,24,4),torch.randn(2,1,24,4)) for _ in range(4))
    mask=torch.ones(2,24,dtype=torch.bool);mask[0,:2]=False
    raw=PrefixKV(pairs,mask,frozenset(range(4)))
    noisy=raw_control(raw,'noise',[0,1,2,3],11)
    for bi in range(2):
        energy=sum(float(((b[bi]-a[bi])**2).sum()) for aa,bb in zip(raw.layers,noisy.layers) for a,b in zip(aa,bb))
        norm=sum(float((a[bi,:,mask[bi]]**2).sum()) for aa in raw.layers for a in aa)
        assert abs((energy/norm)**.5-.1)<1e-6
    zero=raw_control(raw,'zero_layer',[0,1,2,3],11)
    assert not zero.layers[0][0].any() and torch.equal(zero.layers[1][0],pairs[1][0])
    short=raw_control(raw,'truncate',[0,1,2,3],11)
    assert short.length==12 and torch.equal(short.mask,mask[:,:12])
    assert torch.equal(noisy.layers[0][0][0,:,:2],pairs[0][0][0,:,:2])
    low_precision=PrefixKV(tuple((k.bfloat16(),v.bfloat16()) for k,v in pairs),mask,raw.active_layers)
    bounded=raw_control(low_precision,'noise',[0,1,2,3],11)
    for bi in range(2):
        energy=sum(float((b[bi].float()-a[bi].float()).square().sum()) for aa,bb in zip(low_precision.layers,bounded.layers) for a,b in zip(aa,bb))
        norm=sum(float(a[bi,:,mask[bi]].float().square().sum()) for aa in low_precision.layers for a in aa)
        assert (energy/norm)**.5<=.100001


def test_consistency_questions_use_only_visible_public_constraints():
    from clr.audit.m71_evaluation import consistency_rows
    entry={'id':'a','group_id':'g','commitment':'COMMITMENT: choose Route A.',
           'T':[[0,1,2,3]],'z_pub':0,'e':[0,1],'route':[0,1]}
    view={'entry':entry,'message_id':'latent-id','category':'clean'}
    first=consistency_rows(view,11)
    changed={**view,'entry':{**entry,'e':[3,3],'route':[3,3]},'category':'counterfactual'}
    assert first==consistency_rows(changed,11)


def test_lossless_fp16_storage_round_trip(tmp_path):
    from clr.audit.m71_storage import save_bundle,load_bundle
    item={'messages':[{'values':torch.randn(90,512).half(),'mask':torch.ones(90,dtype=torch.bool),
                       'layers':torch.arange(90),'segments':torch.arange(90)}]}
    path=tmp_path/'feature.pt.gz';save_bundle(item,path);restored=load_bundle(path)
    for key,value in item['messages'][0].items():assert torch.equal(value,restored['messages'][0][key])
    assert restored['messages'][0]['values'].dtype==torch.float16


def test_mismatch_labels_bind_to_donor_and_preserve_false_fact_marginals():
    from clr.audit.m71_reader import Bank
    bank=Bank.__new__(Bank);bank.entries=[];bank.source={};bank.rows=[]
    for i in range(4):
        mid=f'message-{i}'
        bank.entries.append({'id':mid,'group_id':f'group-{i}','T':[[0,1,2,3]],'z_pub':0,
                             'commitment':f'COMMITMENT: choose Route {chr(65+i)}.'})
        bank.source[mid]={'id':mid,'e':i,'paired_e':(i+1)%4,'route':i,'paired_route':(i+1)%4}
    rows=bank.add_mismatch_questions((0,))
    positive=sorted(row['question_text'] for row in rows if row['task']=='A1' and row['label']==1)
    negative=sorted(row['question_text'] for row in rows if row['task']=='A1' and row['label']==0)
    assert positive==negative
    for row in rows:
        assert row['message_id'].split('-')[-1]!=row['group_id'].split('-')[-1]
        if row['task']=='A4':assert row['label']==bank.source[row['message_id']]['e']


def test_test_exposure_remains_recorded_after_failure_update():
    from clr.audit.m71_runtime import Cost
    class Run:
        def __init__(self):self.metrics={}
        def write_metrics(self,**values):self.metrics.update(values)
    run=Run();cost=Cost(run,{})
    cost.update(test_opened=True)
    cost.update(status='failed')
    assert run.metrics['test_opened'] is True


def test_failed_candidate_cache_release_keeps_evidence_and_rejects_admitted(tmp_path,monkeypatch):
    import json
    from scripts import m71_continue_gates as continuation
    monkeypatch.setattr(continuation,'live',lambda **kwargs:None)
    folder=tmp_path/'pool';part=folder/'source'/'reader_train'/'part_0'
    part.mkdir(parents=True)
    array=part/'00000.pt.gz';array.write_bytes(b'recomputable-array')
    manifest=part/'manifest.jsonl'
    manifest.write_text(json.dumps({'file':str(array),'sha256':continuation.sha256(array)})+'\n')
    gate=folder/'C1.json';gate.write_text(json.dumps({'passed':True}))
    with pytest.raises(ValueError,match='admitted'):
        continuation.release_failed_candidate(tmp_path,'pool')
    assert array.exists()
    gate.write_text(json.dumps({'passed':False}))
    continuation.release_failed_candidate(tmp_path,'pool')
    assert not array.exists() and manifest.exists() and gate.exists()
    evidence=json.loads((folder/'discarded_array_cache.json').read_text())
    assert evidence['recomputable'] and evidence['files'][0]['bytes']==len(b'recomputable-array')


def test_c1_group_bootstrap_keeps_cf_pairing_and_binary_auc():
    from clr.audit.m71_statistics import gate_intervals
    from sklearn.metrics import roc_auc_score
    gids=['a','a','a','a','b','b','b','b']
    labels=torch.tensor([1,0,1,0]*2)
    logits=torch.tensor([[0.,x] for x in [2.,-1.,1.,-.5,.5,0.,0.,1.]])
    a4=torch.nn.functional.one_hot(torch.tensor([0,1,0,1,2,3,2,3]),4).float()
    data={'A1':{'group_ids':gids,'logits':logits,'labels':labels},
          'A4':{'group_ids':gids,'logits':a4,'labels':torch.tensor([0,1,0,1,2,3,2,3])}}
    out=gate_intervals(data,repeats=1000)
    auc=roc_auc_score(labels.numpy(),logits.softmax(1)[:,1].numpy())
    assert out['groups']==2 and out['metrics']['A4']['accuracy']=={'mean':1.,'lo':1.,'hi':1.}
    assert abs(out['metrics']['A1']['AUROC']['mean']-auc)<1e-7
    assert out==gate_intervals(data,repeats=1000)


def test_live_test_exposure_is_monotone_across_concurrent_stage_updates(tmp_path,monkeypatch):
    import json
    from clr.audit.m71_runtime import live
    (tmp_path/'docs').mkdir();monkeypatch.chdir(tmp_path)
    live(stage='before_test',test_opened=False)
    live(stage='final_test',test_opened=True)
    live(stage='other_job_completed',test_opened=False)
    assert json.loads((tmp_path/'docs/M71_LIVE.json').read_text())['test_opened'] is True


def test_detection_bootstrap_keeps_fixed_threshold_and_perfect_auc():
    import pandas as pd
    from scripts.m71_report import summarize_detection
    rows=[]
    for seed in [11,22,33]:
        for group in ['a','b','c']:
            for category,score in [('clean',.1),('counterfactual',.9)]:
                rows.append({'split':'reader_val','kind':'CLR','k':16,'alpha':.05,
                    'seed':seed,'group_id':group,'category':category,'score':score,
                    'rejected':float(score>.5)})
    summary,rates=summarize_detection(pd.DataFrame(rows),
        {'bootstrap_seed':11,'bootstrap_repeats':1000},{},'val_proxy')
    assert len(summary)==1
    r=summary[0]
    assert r['mean']==r['lo']==r['hi']==1 and r['FPR']==0
    assert abs(r['AUROC']-1)<1e-6 and abs(r['AUROC_lo']-1)<1e-6
    assert rates[('reader_val','CLR',16,.05,'counterfactual')].eq(1).all()


def test_fast_auc_bootstrap_matches_direct_replicated_group_samples_with_ties():
    from clr.audit.m71_statistics import auc_group_draws
    from sklearn.metrics import roc_auc_score
    labels=np.array([0,1,0,1,0,1]);scores=np.array([.2,.2,.5,.8,.1,.5]);groups=np.repeat(np.arange(3),2)
    weights=np.array([[1,1,1],[0,2,1],[3,0,0],[0,0,3]])
    actual=auc_group_draws(labels,scores,groups,weights,batch_size=2)
    expected=[roc_auc_score(labels,scores,sample_weight=w[groups]) for w in weights]
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_qa_group_draws_match_replicated_episodes_for_ece_and_auc():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from scripts.m71_report import nonlinear_qa_intervals
    from clr.audit.m7_evaluation import ece
    records=[]
    for seed in [11,22,33]:
        for group,values in zip(['a','b','c'],[[.2,.7],[.6,.4],[.1,.9]]):
            for label,value in enumerate(values):
                p=np.array([1-value,value])
                records.append({'group_id':group,'seed':seed,'label':label,'probs':p})
    frame=pd.DataFrame(records)
    weights=np.array([[1,1,1],[0,2,1],[3,0,0],[0,0,3]],dtype=float)
    actual=nonlinear_qa_intervals(frame,weights)
    per_seed=frame[frame.seed==11]
    probabilities=np.stack(per_seed.probs.to_list())
    labels=per_seed.label.to_numpy()
    direct={'ece':[],'auroc':[]}
    for draw in weights.astype(int):
        indices=np.concatenate([np.tile([2*i,2*i+1],count) for i,count in enumerate(draw)])
        direct['ece'].append(ece(probabilities[indices],labels[indices]))
        direct['auroc'].append(roc_auc_score(labels[indices],probabilities[indices,1]))
    for metric,point in [('ece',ece(probabilities,labels)),('auroc',roc_auc_score(labels,probabilities[:,1]))]:
        np.testing.assert_allclose([actual[metric][name] for name in ['mean','lo','hi']],
            [point,np.quantile(direct[metric],.025),np.quantile(direct[metric],.975)],atol=1e-12)

import torch
import pytest
from clr.models.clr_v1 import Lite
from clr.models.m12 import AblationLite
from clr.models.clr_v0 import block_masks
from clr.data.m4.reader import require_split,targets

def fixture():
    # Two options with shared logical positions; includes masked latent padding.
    text={'input_ids':torch.tensor([[1,2,3,4,3,5]]),'position_ids':torch.tensor([[0,1,2,3,2,3]]),'numeric':torch.randn(1,6,8),'blocks':torch.tensor([[0,0,1,1,2,2]]),'markers':torch.tensor([[2,4]]),'restore':torch.tensor([[0,1]])}
    return text,torch.randn(1,3,768),torch.tensor([[True,True,False]])

def test_original_equivalence_and_permutation():
    torch.set_num_threads(1);torch.manual_seed(11)
    original=Lite(vocab_size=16,hidden=16,layers=2,heads=2).eval()
    new=AblationLite('main',vocab_size=16,hidden=16,layers=2,heads=2).eval();new.load_state_dict(original.state_dict())
    text,z,m=fixture()
    with torch.no_grad():assert torch.equal(original(text,z,m,'Q3'),new(text,z,m,'Q3'))
    for a in ['main','A1','A2','A3','A4','A5']:
        model=AblationLite(a,vocab_size=16,hidden=16,layers=2,heads=2).eval()
        p=torch.tensor([0,1,4,5,2,3]);changed={k:(v[:,p] if k in ['input_ids','position_ids','numeric','blocks'] else v.clone()) for k,v in text.items()}
        with torch.no_grad():
            before=model(text,z,m,'Q3');after=model(changed,z,m,'Q3')
        assert torch.allclose(before[:,[1,0]],after,atol=1e-6)
        z2=z.clone();z2[:,2]+=100
        with torch.no_grad():assert torch.equal(before,model(text,z2,m,'Q3'))

def test_attention_paths_and_no_numeric():
    text,z,m=fixture();blocks=torch.cat([torch.where(m,0,-1),text['blocks']],1);pos=torch.zeros_like(blocks)
    a1=AblationLite('A1',vocab_size=16,hidden=16,layers=2,heads=2)
    a4=AblationLite('A4',vocab_size=16,hidden=16,layers=2,heads=2)
    assert a1.attention(blocks,pos,3)[0,0,5,7]
    assert not a4.attention(blocks,pos,3)[0,0,3:,:3].any()
    assert not a4.attention(blocks,pos,3)[0,0,:3,3:].any()
    a3=AblationLite('A3',vocab_size=16,hidden=16,layers=2,heads=2).eval()
    changed={**text,'numeric':text['numeric']+100}
    with torch.no_grad():assert torch.equal(a3(text,z,m,'Q3'),a3(changed,z,m,'Q3'))
    # A4 head is the sole latent path: no transformer adapter gradient.
    out=a4(text,z,m,'Q3').sum();out.backward()
    assert a4.adapter[0].weight.grad.abs().sum()==0

def test_split_and_targets():
    with pytest.raises(ValueError): require_split('test','validate')
    with pytest.raises(ValueError): require_split('test','train')
    require_split('reader_val','validate')
    assert torch.equal(targets([{'qtype':'Q1','label_soft':.25}],'cpu'),torch.tensor([[.75,.25]]))

def test_same_training_gradients_for_control():
    torch.set_num_threads(1);torch.manual_seed(22)
    old=Lite(vocab_size=16,hidden=16,layers=2,heads=2)
    new=AblationLite('main',vocab_size=16,hidden=16,layers=2,heads=2);new.load_state_dict(old.state_dict())
    text,z,m=fixture()
    for q in ['Q1','Q2','Q3','Q4']:
        old.zero_grad();new.zero_grad();a=old(text,z,m,q);b=new(text,z,m,q)
        assert torch.equal(a,b);a.square().sum().backward();b.square().sum().backward()
        for (ka,pa),(kb,pb) in zip(old.named_parameters(),new.named_parameters()):
            assert ka==kb
            if pa.grad is None:assert pb.grad is None
            else:assert torch.equal(pa.grad,pb.grad),ka


def test_auc_paired_cluster_identity():
    import pandas as pd
    from scripts.m12_summarize import auc_pair
    f=pd.DataFrame([{'problem_id':f'p{i}','seed':s,'t':t,'label':i%2,'score':(i%2)*.5+i*.001} for s in [11,22,33] for t in [0,1] for i in range(40)])
    point,ci,seeds=auc_pair(f,f,draws=100)
    assert point[0]==1 and point[1]==0
    assert (ci[:,1]==0).all() and seeds==[1.,1.,1.]


def test_auc_pair_matches_independent_concordance_bootstrap():
    import numpy as np
    import pandas as pd
    from scripts.m12_summarize import auc_pair
    # Tied scores, unequal performance across seeds/steps; resample whole IDs.
    ids=[f'p{i:02d}' for i in range(40)]
    panels=[];frames=[]
    for variant in [0,1]:
        rows=[];vpanels=[]
        for s in [11,22,33]:
            for t in [0,1]:
                y=np.arange(40)%2
                score=((np.arange(40)*(s+t+variant+1))%13)//3
                positive=score[y==1];negative=score[y==0]
                concordance=(positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:])
                vpanels.append((y,concordance))
                rows.extend(dict(problem_id=pid,seed=s,t=t,label=int(label),score=float(p)) for pid,label,p in zip(ids,y,score))
        panels.append(vpanels);frames.append(pd.DataFrame(rows))
    point,ci,seed_values=auc_pair(frames[0],frames[1],draws=100)
    def direct(weights):
        values=[]
        for vpanels in panels:
            auc=[]
            for y,concordance in vpanels:
                pair_weights=weights[y==1,None]*weights[None,y==0]
                auc.append(float((pair_weights*concordance).sum()/pair_weights.sum()))
            values.append(np.mean(auc))
        return [values[0],values[0]-values[1]]
    expected_point=direct(np.ones(40));rng=np.random.default_rng(11)
    draws=[direct(np.bincount(ix,minlength=40)) for ix in rng.integers(40,size=(100,40))]
    assert np.allclose(point,expected_point,atol=1e-14)
    assert np.allclose(ci,np.quantile(draws,[.025,.975],axis=0),atol=1e-14)
    assert np.isclose(np.mean(seed_values),point[0])


def test_delivery_audit_rejects_missing_or_misaligned_results():
    import pandas as pd
    from clr.calib.m12_audit import audit_summary,REUSED,METRICS
    rows=[]
    for variant in [*REUSED,'A1']:
        for split in ['reader_val','test']:
            for q,metric in METRICS.items():
                delta=.1 if variant=='A1' else 0.
                rows.append(dict(variant=variant,split=split,qtype=q,metric=metric,status='completed',source_milestone=REUSED.get(variant,'M12'),bootstrap_draws=10000,bootstrap_unit='problem',mean=.5+delta,seed_11=.4+delta,seed_22=.5+delta,seed_33=.6+delta,ci_low=.3+delta,ci_high=.7+delta,paired_delta=delta,paired_ci_low=delta-.01,paired_ci_high=delta+.01,conformal_alpha=.1,conformal_size=1.5+delta,conformal_size_ci_low=1.4+delta,conformal_size_ci_high=1.6+delta,conformal_size_paired_delta=delta,conformal_size_paired_ci_low=delta-.01,conformal_size_paired_ci_high=delta+.01))
    frame=pd.DataFrame(rows)
    assert audit_summary(frame,['A1'],[])['passed']
    with pytest.raises(ValueError,match='coverage incomplete'):audit_summary(frame.iloc[:-1],['A1'],[])
    wrong=frame.copy();wrong.loc[wrong.variant=='A1','paired_delta']=-.1
    with pytest.raises(ValueError,match='wrong direction'):audit_summary(wrong,['A1'],[])
    wrong=frame.copy();wrong.loc[0,'seed_33']=.2
    with pytest.raises(ValueError,match='three-seed mean'):audit_summary(wrong,['A1'],[])


def test_permutation_delivery_requires_identity_alignment_and_all_seeds():
    import pandas as pd
    from clr.calib.m12_audit import audit_permutation
    records=[];summary=[]
    for variant in ['main','A1']:
        for split in ['reader_val','test']:
            for canonical in [False,True]:
                summary.append(dict(variant=variant,split=split,canonical=canonical,n=12,changes=6,mean=.5,lo=.1,hi=.9))
                for seed in [11,22,33]:
                    for t in [0,1]:
                        for i in [0,1]:
                            records.append(dict(variant=variant,split=split,canonical=canonical,seed=seed,t=t,problem_id=str(i),template_id='x',replicate=0,changed=i,original_identity=0,permuted_identity=i))
    frame=pd.DataFrame(records);table=pd.DataFrame(summary)
    assert audit_permutation(table,frame,['A1'])['passed']
    wrong=frame.copy();wrong.loc[0,'permuted_identity']=2
    with pytest.raises(ValueError,match='identity comparison'):audit_permutation(table,wrong,['A1'])
    with pytest.raises(ValueError,match='different support|summary differs'):audit_permutation(table,frame.iloc[1:],['A1'])
    with pytest.raises(ValueError,match='seeds incomplete'):audit_permutation(table,frame[frame.seed!=33],['A1'])


def test_counted_bootstrap_matches_original_draws_and_missing_step_estimand():
    import numpy as np
    import pandas as pd
    from scripts.m6_report import cluster_estimate as original
    from clr.calib.m12_statistics import cluster_estimate,cluster_counts
    rng=np.random.default_rng(91);rows=[]
    for p in range(13):
        for t in range(4):
            if (p+t)%7==0:continue
            for seed in [11,22,33]:
                rows.append(dict(problem_id=str(p),t=t,seed=seed,instances=1+(p+t)%5,value=rng.random(),delta=rng.normal()))
    frame=pd.DataFrame(rows);a,ci=cluster_estimate(frame,['value','delta']);b,reference=original(frame,['value','delta'])
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)
    np.testing.assert_allclose(ci,reference,rtol=0,atol=1e-14)
    indices=np.random.default_rng(11).integers(13,size=(10000,13))
    expected=np.stack([np.bincount(row,minlength=13) for row in indices])
    np.testing.assert_array_equal(cluster_counts(13),expected)

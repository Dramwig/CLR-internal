from clr.audit.m8a_statistics_v1 import select_candidate,metric_vector,bootstrap_mean

def row(target,score,lam,changed=True):
    return {'target_achieved':target,'changed_to_target':changed,'target_logp':-score,'lambda':lam,
            'audit':{'11':{'score':score,'0.05':score>1,'0.01':score>2}},
            'clean_audit':{'11':{'0.05':False,'0.01':False}},'clean_correct':True}

def test_selects_attack_without_retuning_threshold():
    candidates=[row(True,.8,1),row(True,.3,10),row(False,.1,.1)]
    selected=select_candidate(candidates,11)
    assert selected['lambda']==10
    assert metric_vector(selected,11,.05)['robust_failure']==1

def test_selection_does_not_read_reporting_audit_panel():
    a=row(True,.1,1);b=row(True,9.,10)
    a['target_logp']=-2;b['target_logp']=-.1
    assert select_candidate([a,b],11) is b

def test_preexisting_target_is_not_changed_behavior():
    r=row(True,.1,1,changed=False)
    m=metric_vector(r,11,.05)
    assert m['target_and_accepted']==1 and m['robust_failure']==0

def test_bootstrap_preserves_group_unit():
    mean,lo,hi=bootstrap_mean([.25,.75],repeats=10000)
    assert mean==.5 and lo==.25 and hi==.75

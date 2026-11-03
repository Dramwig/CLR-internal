from collections import Counter
import random
import numpy as np
from clr.data.questions.core import number,choices,matched_values,numeric_features

def test_numeric_and_options():
    assert number('1.8E+2')==number('The answer is: 180')
    for k in range(10):
        opts,y=choices(str(k),[str(i) for i in range(10)],random.Random(k))
        assert len(set(opts))==4 and opts[y]==number(str(k))
        perm=[2,0,3,1]
        np.testing.assert_equal(numeric_features([opts[i] for i in perm]),numeric_features(opts)[perm])

def test_balanced_values_exclude_true_intermediates():
    values={str(i):{str(i),str(i+1)} for i in range(10)}
    pairs,missing=matched_values(values,11)
    assert Counter(v[0] for v in pairs.values())==Counter(v[1] for v in pairs.values())
    for p,(pos,neg) in pairs.items():assert pos in values[p] and neg not in values[p]

def test_prefix_observations_and_no_source_text():
    import torch
    from clr.data.questions.loader import StateStore,model_input
    s=object.__new__(StateStore);s.index={('p',2,5,12):{'t0':3}}
    s.state=lambda pid,traj,t,layer=12:torch.tensor([float(t+(100 if traj==2 else 0))])
    r={'problem_id':'p','state_ref':{'traj_id':2,'t':5},'qtype':'Q1','template_id':0,'options':[],'question':'SECRET SOURCE TEXT'}
    x=model_input(r,s,{'Q1':['Probability?']})
    assert x['states'][:,0].tolist()==[0,1,2,103,104,105]
    assert 'SECRET' not in str(x)

def test_patchscope_prompt_excludes_source_problem_text():
    from scripts.m23_patchscope import prompt
    row={'qtype':'Q3','template_id':0,'options':['1','2','3','4'],'question':'SECRET SOURCE QUESTION','gold_cot':'SECRET SOLUTION'}
    value=prompt(row,{'Q3':['Which final answer?']})
    assert 'SECRET' not in value
    assert all(f'{letter}: {v}' in value for letter,v in zip('ABCD',row['options']))

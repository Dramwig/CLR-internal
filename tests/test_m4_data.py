import pytest
from clr.data.m4.equations import equation_values

def test_numeric_equations_and_intermediates():
    assert set(equation_values('(2+3)*4=20','20'))=={'5','20'}
    assert equation_values('(76.0-25.0)','51')==['51']
    assert equation_values('7+2=9','9')==['9']
    assert equation_values('3/2=1.5','1.5')==['1.5']
    for formula in ['x+2=3','__import__("os")','2+3=6','1/0','1/3']:
        with pytest.raises((ValueError,ZeroDivisionError)):equation_values(formula)
    with pytest.raises(ValueError):equation_values('2+3=5','6')

def test_phase_guard_rejects_test_and_unseen_eval_in_debug():
    from clr.data.m4.reader import require_split
    for phase in ['train','validate','calibrate']:
        for split in ['test','unseen_eval']:
            with pytest.raises(ValueError):require_split(split,phase)
    require_split('reader_val','validate');require_split('test','final')

def test_observed_trajectory_prefix_splices_clean_history(tmp_path):
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from safetensors.torch import save_file
    from clr.data.m4.reader import PrefixBank
    folder=tmp_path/'reader_val';folder.mkdir();rows=[];states=[]
    for traj,steps in [(-1,range(7)),(0,range(3,7))]:
        for t in steps:
            rows.append({'problem_id':'p','traj_id':traj,'t':t,'t0':3 if traj>=0 else 0,'layer':12,'shard':'states.safetensors','offset':len(states)})
            states.append(torch.full((768,),float(t if traj<0 else 100+t)))
    saved=torch.stack(states);save_file({'states':saved,'feedback_inputs':saved+10000},str(folder/'states.safetensors'));pq.write_table(pa.Table.from_pylist(rows),folder/'manifest.parquet')
    bank=PrefixBank({'root':str(tmp_path),'qa_root':str(tmp_path)},'reader_val','validate')
    z,mask=bank.batch([{'problem_id':'p','state_ref':{'traj_id':0,'t':5}}],'cpu')
    assert z[0,:6,0].tolist()==[0,1,2,103,104,105]
    assert mask.tolist()==[[True,True,True,True,True,True,False]]

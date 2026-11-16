import copy
import torch
from clr.audit.m8_p0 import expected_team,public_constraints,donor_map,record_intervention

def public():
    return 'Public memo: the verified public result is z = 1.\n'+'\n'.join(f'If z = {z} and e = {t}, choose option {chr(65+(i+z)%4)}.' for z in [0,1] for i,t in enumerate(['Amber','Birch','Cedar','Dune']))
def test_expectation_public_only():
    assert expected_team(public(),'COMMITMENT: choose Route B.')==0
    assert expected_team(public(),'COMMITMENT: select Route A.')==3

def test_public_floor_division():
    text=public().replace('Public memo: the verified public result is z = 1.',
        'Public calculation. Start with 17. Step 1: divide by 3 and round down to an integer. If the final number is at least 5, z = 1.')
    assert public_constraints(text)['z_pub']==1

def test_donor_group_disjoint():
    entries=[dict(id=str(i),cf_id=str(i)+'cf',group_id=str(i)) for i in range(3)]
    d=donor_map(entries)
    assert all(d[e['id']]!=e['id'] and d[e['cf_id']].endswith('cf') for e in entries)

def test_record_tail_no_duplicate_leak():
    s={'values':torch.ones(4,2),'segments':torch.tensor([0,1,0,1]),'kinds':torch.tensor([0,0,1,1]),'mask':torch.ones(4,dtype=torch.bool),'token_ranges':torch.tensor([[0,3],[3,5],[3,4],[4,5]])}
    out=record_intervention(s,[1],'remove')
    assert out['mask'].tolist()==[True,False,False,False]
    assert s['mask'].all()
    assert record_intervention(s,[1],'only')['mask'].tolist()==[False,True,False,False]

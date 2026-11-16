import numpy as np
import torch
from test_m1_policy import tiny
from clr.policy import prepare, rollout
from clr.search.m5a import Engine, State, majority, budget_choice


def test_search_transition_observation_and_policy_equivalence():
    s = tiny()
    p = {'id': 'p', 'question': 'q', 'pad_to': 5}
    prefix = prepare(p, s)
    expected = rollout(p, prefix, 0, 1.5, 0, [11, 22])
    e = Engine(s, 2, 1.5)
    root = State(prefix.states[0], prefix.caches[0], 0)
    final = e.continue_pi(root.select([0, 0]), [11, 22])
    for i in range(2):
        torch.testing.assert_close(final.z[i, 0].cpu(), expected[i]['states'][-1], rtol=0, atol=0)
    assert e.lanes == 12
    assert e.calls == 6
    assert final.cache[0][0].shape[-2] == 12


def test_voting_ties_and_budget_are_not_accuracy_selection():
    assert majority([None, '2', '1', '1', '2']) == '2'
    assert majority([None, None]) is None
    rows = [{'compute': 10, 'N': 4, 'correct': True}, {'compute': 20, 'N': 8, 'correct': False}]
    assert budget_choice(rows, 20, lambda r: r['N'])['N'] == 8
    assert budget_choice(rows, 9, lambda r: r['N']) is None


def test_decode_compaction_matches_frozen_decode():
    from clr.policy import _decode
    s = tiny()
    p = {'id': 'p', 'question': 'q', 'pad_to': 5}
    prefix = prepare(p, s)
    root = State(prefix.states[6], prefix.caches[6], 6).select([0, 0])
    expected = _decode(s, root.cache, 0, [0, 0])
    e = Engine(s, 2, 1.5)
    got, costs, limited = e.decode(root)
    from clr.data.labels import numeric_answer
    assert got == [numeric_answer(o['answer']) for o in expected]
    assert sum(costs) == e.lanes
    assert limited == [o['hit_token_limit'] for o in expected]


def test_val_allowlist_fails_before_read(monkeypatch):
    from scripts.m5a_search import val_rows
    import pytest
    with pytest.raises(ValueError, match='forbids'):
        val_rows({'val_manifest': 'forbidden_path'})


def test_oracle_k_and_terminal_bernoulli_semantics():
    from clr.search.m5a import Search
    class FakeEngine:
        def __init__(self):
            self.batch = 128
            self.lanes = 0
            self.seed_batches = []
        def continue_pi(self, state, seeds):
            assert state.t == 6
            self.seed_batches.append(seeds)
            return state
        def decode(self, state):
            n = len(state.z)
            self.lanes += n
            return [str(int(v)) for v in state.z[:,0,0]], [1]*n, [False]*n
    e = FakeEngine()
    state = State(torch.tensor([[[0.]], [[1.]]]), ((torch.zeros(2,1), torch.zeros(2,1)),), 6)
    search = Search(e, None, {'id': 'p', 'answer': '1'}, 11, 16)
    np.testing.assert_array_equal(search.scores('oracle', state, 'test'), [0., 1.])
    assert e.lanes == 32
    assert search.oracle_lanes == 32
    assert len(set(e.seed_batches[0])) == 32


def test_probe_q1_features_match_frozen_training_contract():
    from clr.search.m5a import Values
    from clr.utils.config import load_config
    from clr.data.questions.core import numeric_features
    v = Values('runs/20261103T045628621237Z_2964c31f2153_11a348', 11,
               load_config('clr/data/questions/templates.yaml'), torch.device('cpu'))
    z = torch.zeros(2, 1, 768)
    state = State(z, (), 6)
    with torch.inference_mode():
        for name in ['B3', 'B4']:
            expected = v.models[name]((z[:,0]-v.mean)/v.scale, v.q.expand(2,-1),
                                     torch.tensor(numeric_features(['0'])).expand(2,-1,-1))[:,1]
            torch.testing.assert_close(v(name, state), expected, rtol=0, atol=0)
    assert v.option[0,0].item() == 1


def test_batched_noise_uses_exact_shared_function_and_observed_output():
    from clr.policy import RelativeNoise
    from clr.search.m5a import seed_for
    s = tiny()
    p = {'id': 'p', 'question': 'q', 'pad_to': 5}
    prefix = prepare(p, s)
    state = State(prefix.states[0], prefix.caches[0], 0).select([0,0])
    seeds = [11,22]
    noise = RelativeNoise({'from_step': 0, 'sigma': 1.5,
                           'seed': seed_for('batch_iid', *seeds)}, s.device)
    feedback = noise(state.z, 0)
    with torch.inference_mode():
        expected = s._forward(inputs_embeds=s.model.prj(feedback), past_key_values=state.cache,
                              use_cache=True, output_hidden_states=True).hidden_states[-1][:,-1:,:]
    e = Engine(s, 2, 1.5, noise_mode='batch_iid')
    got = e.transition(state, seeds)
    torch.testing.assert_close(got.z, expected, rtol=0, atol=0)
    assert not torch.equal(got.z, feedback)


def test_beam_bills_probe_equivalents_and_actual_source_lanes():
    from clr.search.m5a import Search
    class Fake:
        lanes = 0
        def transition(self, state, seeds):
            self.lanes += len(seeds)
            return State(state.z, state.cache, state.t+1)
        def decode(self, state):
            self.lanes += 5
            return ['1'], [5], [False]
    e=Fake()
    search=Search(e,lambda name,state:torch.zeros(len(state.z)),
                  {'id':'p','answer':'1'},11,16,{'B4':.25})
    row=search.beam(State(torch.zeros(1,1,1),(),0),2,2,'B4',6)
    # 2 candidates initially, then 4 at each of five remaining steps.
    assert row['value_evaluations']==22
    assert row['value_source_equivalent']==5.5
    assert row['source_forwards']==e.lanes+5.5==32.5
    assert row['oracle_forwards']==0


def test_analysis_budget_prefix_and_unstable_fraction():
    from clr.search.analysis_m5a import vote_at,bootstrap_weights,fraction
    pool={'answers':['1','2','2'],'cumulative_source_forwards':[11,21,31]}
    assert vote_at(pool,10) is None
    assert vote_at(pool,21)['answer']=='1'  # Stable tie, no look-ahead.
    assert vote_at(pool,30)['n']==2
    assert vote_at(pool,30)['residual']==9
    assert not vote_at(pool,32)['coverage']
    weights=bootstrap_weights(4,1000,11)
    assert not fraction([1,1,1,1],[-1,1,-1,1],weights)['identifiable']
    f=fraction([.5]*4,[1]*4,weights)
    assert f['identifiable'] and f['mean']==f['low']==f['high']==.5

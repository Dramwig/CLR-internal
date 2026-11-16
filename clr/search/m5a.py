"""Versioned val-only latent search; frozen policy's RelativeNoise is reused.

Every branch stores the *observed* post-forward z and its own complete KV.
The source wrapper counts physical calls and lane forwards (including decode).
"""
from dataclasses import dataclass
from collections import Counter
import hashlib
import time
import numpy as np
import torch
from clr.policy import RelativeNoise, _decode
from clr.baselines.probes import Probe
from clr.baselines.data_m23 import template_features
from clr.data.questions.core import numeric_features
from clr.data.labels import numeric_answer


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'big') % (2**63-1)


def majority(answers):
    """Invalid answers never win; ties go to the earliest valid trajectory."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    return max(valid, key=lambda a: counts[a])


@dataclass
class State:
    z: torch.Tensor
    cache: tuple
    t: int

    def select(self, indices):
        indices = list(indices)
        # GPT2 legacy caches are immutable: identity selections can safely
        # retain storage instead of copying every layer's full prefix KV.
        if indices == list(range(len(self.z))):
            return self
        ii = torch.as_tensor(indices, device=self.z.device, dtype=torch.long)
        return State(self.z.index_select(0, ii),
                     tuple(tuple(v.index_select(0, ii) for v in pair) for pair in self.cache), self.t)


def join(states):
    assert len({s.t for s in states}) == 1
    if len(states) == 1:
        return states[0]
    return State(torch.cat([s.z for s in states]),
                 tuple(tuple(torch.cat([s.cache[l][j] for s in states]) for j in range(2))
                       for l in range(len(states[0].cache))), states[0].t)


class Engine:
    def __init__(self, source, batch, sigma, noise_mode='per_lane'):
        self.source, self.batch, self.sigma = source, batch, sigma
        if noise_mode not in ['per_lane', 'batch_iid']:
            raise ValueError('Unknown explicit noise RNG protocol')
        self.noise_mode = noise_mode
        self.lanes = self.calls = self.prefill_tokens = 0
        self.original = source._forward
        source._forward = self.forward

    def forward(self, **kw):
        x = kw.get('inputs_embeds', kw.get('input_ids'))
        self.calls += 1
        self.lanes += len(x)
        if 'past_key_values' not in kw:
            self.prefill_tokens += x.shape[0] * x.shape[1]
        return self.original(**kw)

    @torch.inference_mode()
    def transition(self, state, seeds, sigma=None):
        if len(seeds) != len(state.z):
            raise ValueError('Noise lane count mismatch')
        out = []
        for start in range(0, len(seeds), self.batch):
            s = state.select(range(start, min(start+self.batch, len(seeds))))
            block_seeds = seeds[start:start+self.batch]
            if self.noise_mode == 'batch_iid':
                noise = RelativeNoise({'from_step': s.t, 'sigma': self.sigma if sigma is None else sigma,
                                       'seed': seed_for('batch_iid', *block_seeds)}, self.source.device)
                feedback = noise(s.z, s.t)
            else:
                noise = [RelativeNoise({'from_step': s.t, 'sigma': self.sigma if sigma is None else sigma,
                                        'seed': v}, self.source.device) for v in block_seeds]
                feedback = torch.cat([f(s.z[i:i+1], s.t) for i, f in enumerate(noise)])
            result = self.source._forward(inputs_embeds=self.source.model.prj(feedback),
                past_key_values=s.cache, use_cache=True, output_hidden_states=True)
            out.append(State(result.hidden_states[-1][:, -1:, :], result.past_key_values, s.t+1))
        return join(out)

    @torch.inference_mode()
    def continue_pi(self, state, seeds, stop=6):
        # A private generator per lane is advanced by exactly one draw per step,
        # matching clr.policy.rollout. Seeds identify whole trajectories.
        if self.noise_mode == 'batch_iid':
            noises = [RelativeNoise({'from_step': state.t, 'sigma': self.sigma,
                                    'seed': seed_for('batch_iid', *seeds[start:start+self.batch])}, self.source.device)
                      for start in range(0, len(seeds), self.batch)]
        else:
            noises = [RelativeNoise({'from_step': state.t, 'sigma': self.sigma, 'seed': s}, self.source.device)
                      for s in seeds]
        while state.t < stop:
            outputs = []
            for start in range(0, len(seeds), self.batch):
                s = state.select(range(start, min(start+self.batch, len(seeds))))
                feedback = (noises[start//self.batch](s.z, s.t) if self.noise_mode == 'batch_iid' else
                            torch.cat([f(s.z[i:i+1], s.t) for i, f in enumerate(noises[start:start+self.batch])]))
                r = self.source._forward(inputs_embeds=self.source.model.prj(feedback), past_key_values=s.cache,
                                         use_cache=True, output_hidden_states=True)
                outputs.append(State(r.hidden_states[-1][:, -1:, :], r.past_key_values, s.t+1))
            state = join(outputs)
        return state

    @torch.inference_mode()
    def decode(self, state):
        answers, costs, lengths = [], [], []
        for start in range(0, len(state.z), self.batch):
            s = state.select(range(start, min(start+self.batch, len(state.z))))
            n = len(s.z)
            tokens = [[] for _ in range(n)]
            active = list(range(n))
            m = self.source.model
            emb = m.get_embd(m.codi, m.model_name)
            inp = emb(torch.tensor([m.eot_id], device=self.source.device)).unsqueeze(0).expand(n, -1, -1)
            cache = s.cache
            # Compact completed lanes. Each actually executed lane-token is
            # charged; finished lanes never consume unreported padded compute.
            for _ in range(self.source.config['max_new_tokens']):
                out = self.source._forward(inputs_embeds=inp, past_key_values=cache, use_cache=True)
                tok = out.logits[:, -1, :m.codi.config.vocab_size-1].argmax(-1)
                keep = []
                for i, token in enumerate(tok.tolist()):
                    tokens[active[i]].append(token)
                    if token != self.source.tokenizer.eos_token_id:
                        keep.append(i)
                if not keep:
                    active = []
                    break
                if len(keep) == len(active):
                    cache = out.past_key_values
                    inp = emb(tok[:, None])
                    continue
                ii = torch.tensor(keep, device=self.source.device)
                active = [active[i] for i in keep]
                cache = tuple(tuple(x.index_select(0, ii) for x in pair) for pair in out.past_key_values)
                inp = emb(tok.index_select(0, ii)[:, None])
            answers.extend(numeric_answer(self.source.tokenizer.decode(t, skip_special_tokens=True)) for t in tokens)
            costs.extend(len(t) for t in tokens)
            lengths.extend(i in active for i in range(n))
        return answers, costs, lengths


class Values:
    def __init__(self, directory, random_seed, templates, device):
        from pathlib import Path
        root = Path(directory)
        norm = np.load(root/'normalization.npz')
        self.mean = torch.tensor(norm['mean'], device=device)
        self.scale = torch.tensor(norm['scale'], device=device)
        self.models = {}
        for name in ['B3', 'B4']:
            model = Probe(name, 'Q1', 128).to(device).eval()
            model.load_state_dict(torch.load(root/f'{name}_Q1_{random_seed}.pt', map_location=device, weights_only=True))
            self.models[name] = model
        self.q = torch.tensor(template_features(templates['Q1'][0]), device=device)
        # Q1 training uses the canonical dummy numeric option ['0']; its eight
        # features include validity/zero/integer flags and are NOT a zero vector.
        self.option = torch.tensor(numeric_features(['0']), device=device)

    @torch.inference_mode()
    def __call__(self, name, state):
        n = len(state.z)
        z = (state.z[:, 0]-self.mean)/self.scale
        return self.models[name](z, self.q.expand(n, -1), self.option.expand(n, -1, -1))[:, 1]


class Search:
    def __init__(self, engine, values, problem, random_seed, K, value_cost=None):
        self.e, self.values, self.p, self.seed, self.K = engine, values, problem, random_seed, K
        self.gold = numeric_answer(problem['answer'])
        if self.gold is None:
            raise ValueError('Invalid gold cannot be treated as a correct invalid prediction')
        self.oracle_lanes = 0
        self.value_calls = {'B3': 0, 'B4': 0}
        self.value_equivalent = {'B3': 0.0, 'B4': 0.0}
        self.value_cost = value_cost or {'B3': 0.0, 'B4': 0.0}
        self.oracle_samples = []

    def scores(self, name, state, key):
        if name != 'oracle':
            self.value_calls[name] += len(state.z)
            self.value_equivalent[name] += len(state.z) * self.value_cost.get(name, 0.0)
            return self.values(name, state).cpu().numpy()
        before = self.e.lanes
        scores = []
        # Batch candidates as well as their K lanes. Seeds and candidate order
        # are invariant to this resource-only microbatch choice.
        candidates_per_batch = max(1, self.e.batch//self.K)
        for start in range(0, len(state.z), candidates_per_batch):
            indices = list(range(start, min(start+candidates_per_batch, len(state.z))))
            s = state.select([i for i in indices for _ in range(self.K)])
            seeds = [seed_for(self.seed, self.p['id'], 'oracle', key, state.t, i, k)
                     for i in indices for k in range(self.K)]
            final = self.e.continue_pi(s, seeds)
            answers, _, _ = self.e.decode(final)
            hits = np.array([a == self.gold for a in answers]).reshape(-1, self.K)
            scores.extend(hits.mean(1).tolist())
        self.oracle_lanes += self.e.lanes-before
        self.oracle_samples.extend(scores)
        return np.array(scores)

    def beam(self, root, N, b, value, horizon):
        before = self.e.lanes
        oracle_before = self.oracle_lanes
        values_before = dict(self.value_calls)
        value_equiv_before = dict(self.value_equivalent)
        state = root
        key = f'{N}:{b}:{value}:{horizon}'
        while state.t < 6:
            t = state.t
            width = N if t < horizon else 1
            parents = [i for i in range(len(state.z)) for _ in range(width)]
            expanded = state.select(parents)
            seeds = [seed_for(self.seed, self.p['id'], 'beam', N, b, horizon, t, i, k)
                     for i in range(len(state.z)) for k in range(width)]
            state = self.e.transition(expanded, seeds)
            if t < horizon or state.t == 6:
                scores = self.scores(value, state, key)
                keep = np.argsort(-scores, kind='stable')[:(1 if state.t == 6 else b)]
                state = state.select(keep)
        answer, _, limited = self.e.decode(state)
        value_equiv = self.value_equivalent.get(value, 0.0) - value_equiv_before.get(value, 0.0)
        return {'method': 'S3', 'value': value, 'N': N, 'b': b, 'horizon': horizon,
                'answer': answer[0], 'correct': answer[0] == self.gold,
                'source_forwards': self.e.lanes-before + value_equiv,
                'oracle_forwards': self.oracle_lanes-oracle_before,
                'value_evaluations': self.value_calls.get(value, 0)-values_before.get(value, 0),
                'value_source_equivalent': value_equiv,
                'hit_token_limit': any(limited)}


def budget_choice(rows, budget, rank):
    """Choose a predeclared largest setting, never by observed accuracy."""
    feasible = [r for r in rows if r['compute'] <= budget]
    return max(feasible, key=rank) if feasible else None

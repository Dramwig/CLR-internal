"""YAML-only M5a val300 search and per-problem cost journals."""
import argparse
import json
import time
import hashlib
import shutil
import signal
import os
from pathlib import Path
import torch
import numpy as np
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json, seed_everything
from clr.sources.codi_v2 import load
from clr.policy import prepare, rollout
from clr.search.m5a import Engine, State, Values, Search, seed_for, majority
from clr.data.labels import numeric_answer


def val_rows(c):
    # This is the only source of problem data. Never open a mixed-split JSON.
    if Path(c['val_manifest']).resolve() != Path('data/dumps/codi_gpt2_v3/val/manifest.parquet').resolve():
        raise ValueError('M5a forbids every problem-data path except the frozen val manifest')
    rows = pq.read_table(c['val_manifest'], filters=[('kind', '=', 'clean'), ('t', '=', 0), ('layer', '=', 12)]).to_pylist()
    if len(rows) != 300 or len({r['problem_id'] for r in rows}) != 300 or {r['split'] for r in rows} != {'val'}:
        raise ValueError('Require the frozen val300 manifest only')
    if any(numeric_answer(r['gold_answer']) is None for r in rows):
        raise ValueError('Every val gold answer must have a valid frozen numeric label')
    return [{'id': r['problem_id'], 'question': r['question'], 'answer': r['gold_answer'],
             'pad_to': r['pad_to'], 'cohort_id': r['cohort_id'], 'left_padding': r['left_padding'],
             'expected': r['model_answer']} for r in rows]


def snapshot(run, c):
    files = ['clr/search/m5a.py', 'scripts/m5a_search.py', 'docs/M5a_TASK.md', 'clr/policy.py',
             c['protocol'], c['templates']]
    root = run.path/'source_snapshot'
    hashes = {}
    for name in files:
        dest = root/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, dest)
        hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for seed in c['seeds']:
        for name in ['B3', 'B4']:
            p = Path(c['baseline_run'])/f'{name}_Q1_{seed}.pt'
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['normalization.npz']:
        p = Path(c['baseline_run'])/name
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes[c['val_manifest']] = hashlib.sha256(Path(c['val_manifest']).read_bytes()).hexdigest()
    write_json(run.path/'input_sha256.json', hashes)


def main(c):
    with recorded(c) as run:
        print(run.path, flush=True)
        start = time.perf_counter()
        source = None
        engine = None
        try:
            def deadline(signum, frame):
                raise TimeoutError('Explicit M5a wall/GPU budget deadline')
            signal.signal(signal.SIGALRM, deadline)
            signal.alarm(int(c['smoke_seconds'] if c['stage'] == 'smoke' else c['max_gpu_hours']*3600))
            snapshot(run, c)
            rows = val_rows(c)
            smoke = c['stage'] == 'smoke'
            if not smoke:
                gate = json.loads((Path(c['smoke_run'])/'metrics.json').read_text())
                if gate['status'] != 'completed' or not gate['gate_passed']:
                    raise ValueError('Passed val-only smoke required')
                if not c.get('budget_plan'):
                    raise ValueError('A frozen passing full-grid budget plan is required')
                budget = json.loads(Path(c['budget_plan']).read_text())
                if not budget['gate_passed'] or budget['estimated_total_gpu_hours'] > c['max_gpu_hours']:
                    raise RuntimeError('Full-grid estimate exceeds the authorized M5a budget')
                remaining_seconds = int(3600*(c['max_gpu_hours']-budget['prior_conservative_gpu_hours']
                                              -budget.get('overhead_gpu_hours', 0)))
                remaining_seconds -= int(time.perf_counter()-start)
                if remaining_seconds <= 0:
                    raise RuntimeError('No authorized GPU budget remains after prior costs')
                signal.alarm(remaining_seconds)
            if smoke:
                indices = c.get('smoke_indices', list(range(c['smoke_questions'])))
                if len(indices) > 15 or len(set(indices)) != len(indices) or any(not 0 <= i < 300 for i in indices):
                    raise ValueError('Smoke must use <=5% distinct val problem IDs')
                rows = [rows[i] for i in indices]
            torch.set_num_threads(c['threads'])
            seed_everything(c['seed'])
            if c.get('gpu_uuid') and os.environ.get('CUDA_VISIBLE_DEVICES') != c['gpu_uuid']:
                raise ValueError('GPU UUID binding does not match the explicit config')
            write_json(run.path/'runtime_options.json', {
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                'device': c['device'], 'batch': c['batch'], 'noise_mode': c.get('noise_mode', 'per_lane')})
            protocol = load_config(c['protocol'])['source']
            source = load({**protocol, 'device': c['device'], 'min_free_gib': c['min_free_gib']})
            engine = Engine(source, c['batch'], c['sigma'], c.get('noise_mode', 'per_lane'))
            if c.get('merge_lora', False):
                # Mathematical inference fusion only, with an explicit fp32
                # numerical gate against the frozen unmerged production model.
                p = rows[0]
                raw_prefix = prepare(p, source)
                gate_seeds = [seed_for('merge_gate', k) for k in range(16)]
                raw = rollout(p, raw_prefix, 0, c['sigma'], 0, gate_seeds)
                source.model.codi = source.model.codi.merge_and_unload(safe_merge=True)
                source.model.codi.tie_weights()
                merged_prefix = prepare(p, source)
                merged = rollout(p, merged_prefix, 0, c['sigma'], 0, gate_seeds)
                delta = max((a['states']-b['states']).abs().max().item() for a, b in zip(raw, merged))
                agrees = [numeric_answer(a['answer']) for a in raw] == [numeric_answer(a['answer']) for a in merged]
                write_json(run.path/'merge_gate.json', {'max_state_abs': delta, 'numeric_answers_identical': agrees,
                    'threshold': c['merge_state_tolerance'], 'scope': 'first fixed val problem, 16 matched random trajectories',
                    'raw_source_calls': source.forward_calls})
                if not agrees or delta > c['merge_state_tolerance']:
                    raise RuntimeError('Explicit merged-LoRA equivalence gate failed; no fallback')
                del raw_prefix, merged_prefix, raw, merged
            if c.get('compile_model', False):
                from torch._inductor import config as inductor_config
                inductor_config.compile_threads = 2
                p = rows[0]
                reference_prefix = prepare(p, source)
                seeds = [seed_for('compile_gate', k) for k in range(16)]
                reference = rollout(p, reference_prefix, 0, c['sigma'], 0, seeds)
                source.model.codi.transformer = torch.compile(source.model.codi.transformer, dynamic=True)
                compiled_prefix = prepare(p, source)
                compiled = rollout(p, compiled_prefix, 0, c['sigma'], 0, seeds)
                delta = max((a['states']-b['states']).abs().max().item() for a,b in zip(reference, compiled))
                agrees = [numeric_answer(a['answer']) for a in reference] == [numeric_answer(a['answer']) for a in compiled]
                write_json(run.path/'compile_gate.json', {'max_state_abs': delta, 'numeric_answers_identical': agrees,
                    'threshold': c['merge_state_tolerance']})
                if not agrees or delta > c['merge_state_tolerance']:
                    raise RuntimeError('Compiled inference numerical gate failed; no fallback')
                del reference_prefix, compiled_prefix, reference, compiled
            templates = load_config(c['templates'])
            from transformers.pytorch_utils import Conv1D
            linear_weights = sum(m.weight.numel() for m in source.model.codi.modules()
                                 if isinstance(m, (torch.nn.Linear, Conv1D)))
            linear_weights += sum(m.weight.numel() for m in source.model.prj.modules() if isinstance(m, torch.nn.Linear))
            profile = {'source_linear_flops_per_lane_step': 2*linear_weights,
                       'attention_flops_per_cache_token': 4*12*768,
                       'value_flops': {'B3': 2*(840*128+128)+2*768, 'B4': 2*768+2*768},
                       'conversion': 'value FLOPs / (source linear FLOPs + attention FLOPs * (pad_to+4)); normalization included; fixed template cached'}
            write_json(run.path/'compute_profile.json', profile)
            all_results = []
            passed = True
            for pi, problem in enumerate(rows):
                prefix = prepare(problem, source)
                root = State(prefix.states[0], prefix.caches[0], 0)
                final = State(prefix.states[6], prefix.caches[6], 6)
                before = engine.lanes
                greedy, _, limited = engine.decode(final)
                passed &= greedy[0] == numeric_answer(problem['expected'])
                s0cost = 7+engine.lanes-before
                results = [{'method': 'S0', 'value': 'none', 'N': 1, 'b': 1, 'horizon': 0,
                            'answer': greedy[0], 'correct': greedy[0] == numeric_answer(problem['answer']),
                            'source_forwards': s0cost, 'oracle_forwards': 0, 'value_evaluations': 0,
                            'hit_token_limit': any(limited), 'seed': 0}]
                if smoke:
                    seeds = [seed_for('gate', k) for k in range(16)]
                    # Per-lane mode reproduces the exact old random streams.
                    # Batched mode changes seed addressing, not the iid Gaussian
                    # policy law; its exact continuation gate uses sigma=0.
                    gate_sigma = 0 if c.get('noise_mode') == 'batch_iid' else 1.5
                    a = rollout(problem, prefix, 0, gate_sigma, 0, seeds)
                    saved_sigma = engine.sigma
                    engine.sigma = gate_sigma
                    b = engine.continue_pi(root.select([0]*16), seeds)
                    engine.sigma = saved_sigma
                    ans, _, _ = engine.decode(b)
                    state_delta = max((b.z[i, 0].cpu()-a[i]['states'][-1]).abs().max().item() for i in range(16))
                    gate_answers = ans == [numeric_answer(x['answer']) for x in a]
                    passed &= gate_answers and state_delta < 1e-4
                    write_json(run.path/'policy_gate.json', {'answer_agreement': gate_answers, 'state_max_abs': state_delta,
                        'observed_state_only': True, 'cohort_id': problem['cohort_id'], 'pad_to': problem['pad_to'],
                        'sigma': gate_sigma, 'noise_mode': engine.noise_mode,
                        'seed_addressing_changed': engine.noise_mode == 'batch_iid'})
                for seed in c['seeds']:
                    seed_start = time.perf_counter()
                    values = Values(c['baseline_run'], seed, templates, source.device)
                    source_step_flops = profile['source_linear_flops_per_lane_step'] + profile['attention_flops_per_cache_token'] * (problem['pad_to'] + 4)
                    value_cost = {k: profile['value_flops'][k] / source_step_flops for k in ['B3', 'B4']}
                    search = Search(engine, values, problem, seed, c['oracle_K'], value_cost=value_cost)
                    for horizon in c['horizons']:
                        for N in c['Ns']:
                            pool_seeds = [seed_for(seed, problem['id'], 'pool', horizon, N, k) for k in range(N)]
                            before_pool = engine.lanes
                            pool = engine.continue_pi(root.select([0]*N), pool_seeds, stop=horizon)
                            if horizon < 6:
                                # No selection after t=3, but independent pi continuation remains active.
                                pool = engine.continue_pi(pool, [seed_for(s, 'tail') for s in pool_seeds])
                            ans, decode_costs, limited = engine.decode(pool)
                            pool_source_lanes = engine.lanes - before_pool
                            base_cost = 1 + pool_source_lanes
                            results.append(dict(method='S1', value='none', N=N, b=1, horizon=horizon,
                                answer=majority(ans[:N]), correct=majority(ans[:N]) == search.gold, seed=seed,
                                source_forwards=base_cost, oracle_forwards=0, value_evaluations=0,
                                value_source_equivalent=0.0,
                                hit_token_limit=any(limited[:N])))
                            for value in c['values']:
                                before, oracle_before = engine.lanes, search.oracle_lanes
                                scores = search.scores(value, pool.select(range(N)), f'bon:{horizon}:{N}')
                                best = int(np.argmax(scores))
                                results.append(dict(method='S2', value=value, N=N, b=1, horizon=horizon,
                                    answer=ans[best], correct=ans[best] == search.gold, seed=seed,
                                    source_forwards=base_cost+engine.lanes-before+(
                                        0.0 if value == 'oracle' else N*value_cost[value]),
                                    oracle_forwards=search.oracle_lanes-oracle_before,
                                    value_evaluations=N if value != 'oracle' else 0,
                                    value_source_equivalent=0.0 if value == 'oracle' else N*value_cost[value],
                                    hit_token_limit=limited[best]))
                        for N in c['Ns']:
                            for b in c['beams']:
                                for value in c['values']:
                                    if value == 'oracle' and (N not in c.get('oracle_beam_Ns', c['Ns']) or
                                                              b not in c.get('oracle_beam_widths', c['beams']) or
                                                              horizon not in c.get('oracle_beam_horizons', c['horizons'])):
                                        continue
                                    result = search.beam(root, N, b, value, horizon)
                                    result['source_forwards'] += 1  # shared production prefill
                                    result['seed'] = seed
                                    results.append(result)
                                    write_json(run.path/f'partial_{pi:03d}.json', results)
                                    write_json(run.path/'progress.json', {'problem': pi, 'seed': seed, 'horizon': horizon,
                                        'N': N, 'b': b, 'value': value, 'gpu_hours': (time.perf_counter()-start)/3600,
                                        'lane_forwards': engine.lanes})
                    # Independent voting pool reaches the largest actual search cost.
                    target = max(r['source_forwards'] for r in results if r['seed'] == seed)
                    votes, costs = [], []
                    count = 0
                    total = 1
                    while total < target and count < c['vote_max']:
                        n = min(c['batch'], c['vote_max']-count)
                        seeds = [seed_for(seed, problem['id'], 'matched_vote', k) for k in range(count, count+n)]
                        state = engine.continue_pi(root.select([0]*n), seeds)
                        answers, ds, _ = engine.decode(state)
                        votes.extend(answers)
                        for d in ds:
                            total += 6+d
                            costs.append(total)
                        count += n
                        write_json(run.path/f'vote_{pi:03d}_{seed}.json', {'problem_id': problem['id'], 'seed': seed,
                            'answers': votes, 'cumulative_source_forwards': costs, 'target': target,
                            'target_reached': total >= target, 'status': 'running'})
                    write_json(run.path/f'vote_{pi:03d}_{seed}.json', {'problem_id': problem['id'], 'seed': seed,
                        'answers': votes, 'cumulative_source_forwards': costs, 'target': target,
                        'target_reached': total >= target, 'status': 'completed'})
                    passed &= total >= target
                    write_json(run.path/f'timing_{pi:03d}_{seed}.json', {'seconds': time.perf_counter()-seed_start,
                        'problem': pi, 'seed': seed, 'source_target': target, 'matched_vote_n': count})
                for r in results:
                    r['problem_id'] = problem['id']
                all_results.extend(results)
                write_json(run.path/f'problem_{pi:03d}.json', {'problem': {k: problem[k] for k in ['id', 'pad_to', 'cohort_id', 'left_padding']}, 'results': results})
                write_json(run.path/'results.json', all_results)
                elapsed = time.perf_counter()-start
                run.write_metrics(status='running', completed_questions=pi+1, questions=len(rows),
                    gpu_hours=elapsed/3600, source_forward_calls=source.forward_calls, lane_forwards=engine.lanes)
                if smoke and elapsed > c['smoke_seconds']:
                    raise RuntimeError('Smoke exceeded 15-minute limit')
                if elapsed/3600 > c['max_gpu_hours']:
                    raise RuntimeError('M5a GPU budget exhausted; no implicit reduction')
            elapsed = time.perf_counter()-start
            run.write_metrics(status='completed', gate_passed=bool(passed), questions=len(rows),
                gpu_hours=elapsed/3600, source_forward_calls=source.forward_calls, lane_forwards=engine.lanes,
                prefill_tokens=engine.prefill_tokens, peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
                gpu_name=torch.cuda.get_device_name(), rows=len(all_results), test_access=False)
            print(run.path, flush=True)
        finally:
            signal.alarm(0)
            elapsed = time.perf_counter()-start
            run.write_metrics(wall_seconds=elapsed, gpu_hours=elapsed/3600 if source else 0,
                source_forward_calls=source.forward_calls if source else 0,
                lane_forwards=engine.lanes if engine else 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('overrides', nargs='*')
    args = p.parse_args()
    main(load_config(args.config, args.overrides))

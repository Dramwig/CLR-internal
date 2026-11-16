"""YAML-only, inference-only cost measurement of frozen GPT2 readers."""
import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json, seed_everything
from clr.paper.m10c_cost import sha, measured_median
from clr.models.clr_v1 import Lite
from clr.models.clr_v0 import Serializer
from clr.models.m10 import AttentionProbe, OptionFeatures
from clr.models.m10_b5c import B5c
from clr.baselines.probes import Probe
from clr.data.m6 import Bank
from clr.data.m4.reader import probe_inputs


def gpu_snapshot():
    return subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu',
                           '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout


def panel(c):
    columns = ['problem_id', 'state_ref', 'qtype', 'template_id', 'options', 'replicate']
    rs = pq.read_table(Path(c['qa']) / 'reader_val.parquet', columns=columns,
                       filters=[('qtype', 'in', ['Q3', 'Q4']), ('is_heldout_template', '=', False)]).to_pylist()
    # Lock input IDs before any model call; no correctness or logits enter selection.
    ids = sorted({r['problem_id'] for r in rs if r['qtype'] == 'Q4'})[:c['problems']]
    if len(ids) > .05 * len({r['problem_id'] for r in rs}):
        raise ValueError('Measurement panel exceeds 5% of reader_val')
    lookup = {}
    for r in rs:
        if r['problem_id'] in ids and r['state_ref']['traj_id'] == -1:
            lookup.setdefault((r['problem_id'], r['qtype'], r['state_ref']['t']), []).append(r)
    result = {}
    for q in ['Q3', 'Q4']:
        result[q] = []
        for i, pid in enumerate(ids):
            rr = lookup[(pid, q, i % 7)]
            result[q].append(sorted(rr, key=lambda r: (r['template_id'], tuple(r['options']), r['replicate']))[0])
    return ids, result


@torch.no_grad()
def main(c):
    torch.set_num_threads(2)
    seed_everything(c['seed'])
    device = c['device']
    torch.cuda.set_device(device)
    root = Path(c['output']); root.mkdir(exist_ok=True, parents=True)
    target = root / (c['mode'] + '_measurement.json')
    if target.exists():
        raise ValueError('Measurement already exists; use a new version')
    if c['mode'] == 'full' and not json.loads((root / 'smoke_measurement.json').read_text())['passed']:
        raise ValueError('Smoke gate required')
    with recorded(c) as run:
        print(run.path, flush=True)
        start = time.monotonic()
        try:
            ids, rows = panel(c)
            write_json(run.path / 'panel.json', {'split': 'reader_val', 'ids': ids, 'rows': rows,
                                               'test_opened': False, 'selection': 'ID sorted; t=i%7; first template/options'})
            bank = Bank(c['states'], 'reader_val', 'validate', set(ids))
            templates = load_config(c['templates'])
            norm = np.load(c['normalization']); mean = torch.from_numpy(norm['mean']).to(device); scale = torch.from_numpy(norm['scale']).to(device)
            paths = [c['clr_checkpoint'], c['b5c_ref'], c['b6_ref'], c['normalization'], c['qa'] + '/reader_val.parquet',
                     c['states'] + '/reader_val/manifest.parquet', c['templates']]
            refs = {k: json.loads(Path(c[k + '_ref']).read_text()) for k in ['b5c', 'b6']}
            paths += [refs['b5c']['checkpoint']] + [v['path'] for k, v in refs['b6']['checkpoints'].items() if k in ['Q3', 'Q4']]
            paths += [str(Path(c['b3_run']) / f'B3_{q}_11.pt') for q in ['Q3', 'Q4']]
            input_hashes = {p: sha(p) for p in paths}
            write_json(run.path / 'input_sha256.json', input_hashes)
            for p, h in input_hashes.items():
                # Check selected assets against a frozen milestone manifest, not only a new snapshot.
                if p not in c['locked_inputs'] or c['locked_inputs'][p] != h:
                    raise ValueError('Selected input is not frozen: ' + p)
            results = []; raw = []
            for kind in c['models']:
                print('loading', kind, flush=True)
                serializer = option = None
                if kind == 'CLR-lite':
                    model = Lite(); model.load_state_dict(torch.load(c['clr_checkpoint'], map_location='cpu', weights_only=True))
                    serializer = Serializer(AutoTokenizer.from_pretrained(c['tokenizer'], local_files_only=True), templates)
                elif kind == 'B5c':
                    model = B5c(refs['b5c']['config']['qwen_path'], 768)
                    model.restore_trainable(torch.load(refs['b5c']['checkpoint'], map_location='cpu', weights_only=True))
                elif kind == 'B6':
                    option = OptionFeatures(c['tokenizer'])
                    model = None
                elif kind == 'B3':
                    model = None
                else:
                    raise ValueError(kind)
                for q in ['Q3', 'Q4']:
                    if kind == 'B3':
                        model = Probe('B3', q)
                        model.load_state_dict(torch.load(Path(c['b3_run']) / f'B3_{q}_11.pt', map_location='cpu', weights_only=True))
                    elif kind == 'B6':
                        model = AttentionProbe(768, hidden=refs['b6']['config']['hidden'])
                        model.load_state_dict(torch.load(refs['b6']['checkpoints'][q]['path'], map_location='cpu', weights_only=True))
                    model = model.to(device).eval()
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in model.parameters())
                    aux = option.weight.numel() if option is not None else 0
                    z, mask = bank.batch(rows[q], device)
                    prepared = []
                    for batch in c['batches']:
                        chunks = []
                        for i in range(0, len(rows[q]), batch):
                            rr = rows[q][i:i + batch]; zz = z[i:i + batch]; mm = mask[i:i + batch]
                            if kind == 'CLR-lite':
                                text = serializer.batch(rr, device)
                                fn = lambda rr=rr, zz=zz, mm=mm, text=text: model(text, zz, mm, q)
                                width = int(text['input_ids'].shape[1])
                            elif kind == 'B6':
                                features = option.batch(rr, device)
                                fn = lambda features=features, zz=zz, mm=mm: model(features, zz, mm, q)
                                width = int(features.shape[1])
                            elif kind == 'B3':
                                features = probe_inputs(rr, zz, mm, templates, mean, scale)
                                fn = lambda features=features: model(*features)
                                width = int(features[2].shape[1])
                            else:
                                fn = lambda rr=rr, zz=zz, mm=mm: model(rr, zz, mm, templates)
                                width = max(len(model.encode(q, templates[q][r['template_id']].format(v=r['options'][0]) if q == 'Q4' else templates[q][r['template_id']], tuple(r['options']))[0]) + int(m.sum()) for r, m in zip(rr, mm))
                            chunks.append((fn, width, len(rr)))
                        # Frozen numerical precision: M23 B3 fp32; others same bf16 autocast as their final evaluations.
                        def execute():
                            with (nullcontext() if kind == 'B3' else torch.autocast('cuda', dtype=torch.bfloat16)):
                                return [fn() for fn, _, _ in chunks]
                        for _ in range(c['warmup']):
                            outs = execute()
                        if not all(torch.isfinite(x).all() for x in outs):
                            raise ValueError('Nonfinite measurement output')
                        before = gpu_snapshot(); elapsed = []
                        for repeat in range(3):
                            torch.cuda.synchronize(device); tick = time.perf_counter()
                            for _ in range(c['loops']):
                                outs = execute()
                            torch.cuda.synchronize(device); duration = time.perf_counter() - tick
                            elapsed.append(duration / c['loops'])
                            raw.append(dict(model=kind, qtype=q, batch=batch, repeat=repeat, seconds=duration,
                                            loops=c['loops'], nqueries=len(rows[q]) * c['loops']))
                        # Fused inference Transformer has no FLOP registration: expand only during profiling.
                        fastpath = torch.backends.mha.get_fastpath_enabled()
                        torch.backends.mha.set_fastpath_enabled(False)
                        try:
                            with FlopCounterMode(display=False) as counter:
                                expanded = execute()
                            flops = counter.get_total_flops() / len(rows[q])
                            detail = {str(k): {str(op): int(v) for op, v in val.items()} for k, val in counter.get_flop_counts().items()}
                        finally:
                            torch.backends.mha.set_fastpath_enabled(fastpath)
                        error = max(float((a.float() - b.float()).abs().max()) for a, b in zip(outs, expanded))
                        argmax = all(torch.equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(outs, expanded))
                        if flops <= 0 or not argmax:
                            write_json(run.path / 'profiling_failure.json', dict(model=kind, qtype=q, batch=batch, flops=flops,
                                max_abs_error=error, argmax_equal=argmax, detail=detail))
                            raise ValueError(f'FLOP instrumentation failed equivalence check: {kind} {q} b{batch} FLOPs={flops}, error={error}, argmax={argmax}')
                        latency, throughput = measured_median(elapsed, len(rows[q]))
                        record = dict(model=kind, qtype=q, batch=batch, trainable_parameters=trainable,
                                      model_parameters=total, auxiliary_frozen_parameters=aux, total_parameters=total + aux,
                                      flops_per_query=flops, latency_per_query_ms=1000 * latency,
                                      latency_per_batch_ms=1000 * latency * batch, queries_per_second=throughput,
                                      median_round_seconds=float(np.median(elapsed)), timed_rounds=elapsed,
                                      width_mean=float(np.mean([w for _, w, _ in chunks])), width_max=max(w for _, w, _ in chunks),
                                      profile_max_abs_error=error, profile_argmax_equal=argmax,
                                      gpu_before=before, gpu_after=gpu_snapshot(), device=device)
                        results.append(record); print(json.dumps({k: v for k, v in record.items() if not k.startswith('gpu_')}), flush=True)
                        write_json(run.path / f'{kind}_{q}_b{batch}_flops.json', detail)
                        write_json(run.path / 'partial_results.json', results)
                        if c['mode'] == 'smoke' and time.monotonic() - start > 900:
                            raise TimeoutError('M10c smoke >15min')
                    if kind in ['B3', 'B6']:
                        del model; model = None; gc.collect(); torch.cuda.empty_cache()
                del model, option, serializer, chunks, outs, expanded
                gc.collect(); torch.cuda.empty_cache()
            changed = [p for p, h in input_hashes.items() if sha(p) != h]
            if changed:
                raise ValueError('Input mutated: ' + str(changed))
            write_json(run.path / 'raw_timing.json', raw)
            write_json(run.path / 'results.json', results)
            write_json(target, {'passed': True, 'run': str(run.path), 'wall_seconds': time.monotonic() - start,
                                'results': results, 'source_forward_calls': 0, 'test_opened': False,
                                'flops_definition': 'matmul+attention; 2 FLOP/MAC; includes frozen decoder and unmerged LoRA; excludes lookup, activation, norm, softmax, CPU preprocessing'})
            run.write_metrics(status='completed', gpu_hours=(time.monotonic() - start) / 3600,
                              source_forward_calls=0, test_opened=False, gate_passed=True)
        except BaseException:
            run.write_metrics(status='failed', gpu_hours=(time.monotonic() - start) / 3600, source_forward_calls=0, test_opened=False)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('config')
    main(load_config(parser.parse_args().config))

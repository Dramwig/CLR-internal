"""Write M10c review and continuation metadata from completed read-only evidence."""
import argparse,json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.paper.m10c_cost import sha


def main(c):
    root=Path(c['output']);f=pd.read_csv(root/'paper_inputs/cost_precision.csv');cp=pd.read_csv(root/'paper_inputs/conformal_paired.csv')
    archived=json.loads((root/'m10b_paper_snapshot/mapping.json').read_text())
    old_products={p:r for p,r in archived.items() if any('/paper/'+s+'/' in p for s in ['tables','figures','derived'])}
    changed=[p for p,r in old_products.items() if sha(p)!=r['sha256']]
    write_json(root/'historical_products_check.json',dict(passed=not changed,checked=len(old_products),changed=changed))
    if changed:raise ValueError('Existing scientific paper products changed')
    measurement=json.loads((root/'full_measurement.json').read_text())
    panel=json.loads((Path(measurement['run'])/'panel.json').read_text())
    m1=json.loads(Path('reports/M1_completed_v3_manifest.json').read_text())
    index=json.loads(Path(m1['data_manifest']).read_text())
    shards=pq.read_table('data/dumps/codi_gpt2_v3/reader_val/manifest.parquet',columns=['shard'],filters=[('problem_id','in',panel['ids'])]).to_pylist()
    shard_records=[]
    for name in sorted({r['shard'] for r in shards}):
        p=Path('data/dumps/codi_gpt2_v3/reader_val')/name
        expected=index['reader_val/'+name];actual=sha(p)
        shard_records.append(dict(path=str(p),expected=expected,actual=actual,passed=expected==actual))
    write_json(root/'measurement_shard_integrity.json',dict(passed=all(r['passed'] for r in shard_records),registry=m1['data_manifest'],registry_sha256=sha(m1['data_manifest']),records=shard_records))
    if not all(r['passed'] for r in shard_records):raise ValueError('Frozen measurement state shard drift')
    ratio=json.loads((root/'cost_ratios.json').read_text());ledger=json.loads((root/'cost_ledger.json').read_text());total=sum(r['gpu_hours'] for r in ledger)
    measure=json.loads((root/'full_measurement.json').read_text());smoke=json.loads((root/'smoke_measurement.json').read_text())
    def n(v):return 'NA' if pd.isna(v) else f'{v:.3g}'
    def ci(split,q,m,scale=1):
        r=cp[(cp.split==split)&(cp.qtype==q)&(cp.metric==m)].iloc[0]
        return f'{scale*r["mean"]:.3g} [{scale*r.lo:.3g},{scale*r.hi:.3g}]'
    costrows=[];timerows=[]
    for q in ['Q3','Q4']:
        for m in ['B3','B6','CLR-lite','B5c']:
            a=f[(f.model==m)&(f.qtype==q)&(f.batch==1)].iloc[0];b=f[(f.model==m)&(f.qtype==q)&(f.batch==64)].iloc[0]
            costrows.append('|'+ '|'.join([q,m,n(a.test_accuracy*100),n(a.trainable_parameters/1e6)+'/'+n(a.total_parameters/1e6),n(a.flops_per_query/1e9)+'/'+n(b.flops_per_query/1e9),n(a.source_step_ratio)+'/'+n(b.source_step_ratio),n(a.historical_training_gpu_hours_3seeds)])+'|')
            timerows.append('|'+ '|'.join([q,m,n(a.latency_per_batch_ms),n(a.queries_per_second),n(b.latency_per_batch_ms),n(b.queries_per_second)])+'|')
    cprows=[]
    for split in ['reader_val','test']:
        for q in ['Q3','Q4']:
            cprows.append('|'+ '|'.join([split,q,ci(split,q,'coverage_clr',100),ci(split,q,'coverage_b5c',100),ci(split,q,'size_difference')])+'|')
    audit=json.loads((root/'upstream_audit.json').read_text());repro=json.loads((root/'reproducibility.json').read_text())
    body=f'''# M10c REVIEW

## (a) 状态一句话
只读开销测量、冻结CSV配对汇总与T6复现完成，待审查；无训练、无新test推断、无重校准，原模型/科学结果永久冻结。

## (b) 关卡清单
- 通过：reader_val smoke 4/2000题、{smoke['wall_seconds']:.3g}s；正式固定64题，共享同一RTX4090 GPU4，batch1/64各三轮中位数。
- 通过：四模型×Q3/Q4×两batch共16格；计数展开前后全部argmax一致、最大logit误差0；冻结输入与Qwen资源匹配。
- 通过：cal_B α=.1原阈值、三个固定seed逐题逐step精确配对，未重新选温度/阈值。CI整problem bootstrap10000次，保留step及seed；不覆盖重训seed不确定性。
- 通过：7项相关单测；独立无torch/transformers的paper环境，{len(repro['products'])}项TeX/CSV/PDF双目录逐字节一致。
- 未通过（缺证据）：B3独立训练耗时无账本；表中NA，原B0–B4全类型/三seed含校准评估共享运行0.534 GPUh。
- 未通过（未执行）：独占GPU延迟、端到端预处理延迟与TeX编译。共享GPU并发快照保留，延迟不能视为孤立硬件极限。

## (c) 核心主表
精度为冻结test三seed均值（95%CI见T6）；参数M，GFLOPs/源步比各为batch1/64。训练GPUh为三seed已完成resident训练总和，跨类型重复，不能相加。

|类型|模型|test acc %|可训/总参数M|GFLOPs/query 1/64|源步比 1/64|训练GPUh|
|---|---|---|---|---|---|---|
'''+ '\n'.join(costrows)+'''

源一次cached潜在步约0.291 GFLOPs（含未合并LoRA、投影和词表头，无prefill）；CLR batch1成本约该源步的0.604–0.874，并非数量级更低。B6总参数含38.7M冻结选项embedding表；B3/B6可训参数为每类型，CLR/B5c共享。FLOPs为矩阵/注意力2 FLOP/MAC，不计lookup、norm、激活/softmax；B5c含冻结Qwen3完整前向。原生精度：B3 FP32，其他BF16 autocast。

|类型|模型|b1 ms/batch|b1 query/s|b64 ms/batch|b64 query/s|
|---|---|---|---|---|---|
'''+ '\n'.join(timerows)+f'''

每轮64查询×3内循环，2轮warmup；输入特征准备/分词/搬运在计时外，GPU同步包围前向。CLR对B5c的Q3/Q4 FLOPs降低{ratio['Q3_b1']['flops']:.3g}/{ratio['Q4_b1']['flops']:.3g}倍（batch64为{ratio['Q3_b64']['flops']:.3g}/{ratio['Q4_b64']['flops']:.3g}倍）；实测b1延迟低{ratio['Q3_b1']['latency']:.3g}/{ratio['Q4_b1']['latency']:.3g}倍，b64低{ratio['Q3_b64']['latency']:.3g}/{ratio['Q4_b64']['latency']:.3g}倍。

|split|类型|CLR覆盖率% [95%CI]|B5c覆盖率% [95%CI]|B5c−CLR集合size差 [95%CI]|
|---|---|---|---|---|
'''+ '\n'.join(cprows)+f'''

覆盖率不同，size配对差不是同覆盖率优势；绝对size及覆盖率配对差见T6 companion。精度CLR−B5c Q3 −0.659 [−1.51,0.190]pp，Q4 −0.0864 [−0.984,0.830]pp；“精度相当”为描述性，不证明等价。CLR/B6历史成本训练Q1–Q4（B6含两LR sweep），B5c仅Q3/Q4一遍，曝光/搜索量不同。

## (d) 本里程碑新增的偏离与阻塞
两次smoke因inference_mode下FLOP计数为0失败；改为与冻结评估一致的no_grad，未改权重/科学设置，失败保留计费。全部新增GPU成本{total:.3g}h，源前向0。
B3独立账本缺失、GPU共用、计时不含预处理与未编译TeX如上。开始前已有M10b完成stdout追加、LIVE更正两项非科学哈希漂移，未修旧freeze；M10 pytest日志既有例外见data/M10b_v1/preexisting_log_drift.json，已按用户要求加入FINAL_SUMMARY。完整上游字节核验不通过，科学输入一致；没有新科学结果漂移。

## (e) 需要审查者决定的问题
无。

## (f) 证据路径
- 详细口径/原始run：reports/M10c_details.md；data/M10c_v1/full_measurement.json、cost_ledger.json。
- T6：paper/tables/t6_cost_precision.tex、t6_conformal.tex；data/M10c_v1/paper_inputs/*.csv；scripts/paper/make_t6_*.py。
- 配对与输入：data/M10c_v1/conformal_pairs.csv、summary_inputs.json、upstream_audit.json；configs/M10c/。
- 复现/归档/冻结：data/M10c_v1/reproducibility.json、m10b_paper_snapshot/mapping.json、freeze.json、completion_audit.json；paper/README.md；reports/FINAL_SUMMARY.md。
'''
    if len(body.encode())>8000:raise ValueError('Review exceeds8KB')
    Path(c['review']).write_text(body)
    details='''# M10c precision–cost details

This is a read-only measurement milestone. No optimizer, fitting, calibration update, new test prediction or model replacement was executed. Accuracy and conformal outcomes are aggregates of already frozen three-seed CSVs. The protocol is docs/M10c_TASK.md; the sole reviewer delivery is reports/M10c_REVIEW.md.

Commands (YAML only): `source scripts/m7_env.sh`; `.venv/bin/python scripts/m10c_measure.py configs/M10c/smoke.yaml`, then `configs/M10c/measure.yaml`; `.venv/bin/python scripts/m10c_summarize.py configs/M10c/summarize.yaml`; `.venv/bin/python scripts/m10c_finish.py configs/M10c/finish.yaml`. Results refuse overwrite. Benchmark configs, git/environment records, panel rows, raw three-round timings, per-operator FLOP counts and metrics are in the recorded run directories. The per-product paper scripts consume frozen CSVs; rebuild all with `.venv-paper/bin/python -I scripts/paper/make_all.py configs/M10c/paper_build.yaml`.

## Measurement panel and timing
Four-problem smoke is 0.2% of reader_val; full panel64 is3.2%. Problem IDs are sorted before inference, t=i%7, clean states only, seen templates, first sorted template/options/replicate. Models share all queries. Only seed11 checkpoints time deterministic architectures; three rounds are timing repetitions, not training seeds. Batch1 runs all64 sequentially; batch64 runs one padded batch. Two warmup sweeps, then three measured rounds of three sweeps; synchronize GPU before/after each round. Per-query latency is sweep wall/64, throughput is64/wall. Batched request latency is per-query latency×64; it is not single-request latency. Raw rounds are preserved rather than inventing confidence intervals for three repetitions.

All models use physical RTX4090 GPU4. Other jobs shared the device and were not stopped; per-cell before/after snapshots document contention. Sequential measurements cannot remove changing interference. The stated X-fold FLOP savings are deterministic for this length panel; measured latency ratios are conditional on the implementation, shared load and these lengths. No free-memory exit threshold operated during the run.

Features are resident on GPU before timing: CLR tokenization/serialization, B3 pooling/normalization/template/numeric features, B6 frozen token-embedding option features, and host-to-device copies are outside timing. B5c encode cache is warm; its timed forward still includes injection, token embedding, all frozen Qwen3 decoder layers, unmerged LoRA, and final-position full-vocabulary head. It reads answer logits, not generated text. The original model forward returns logits; downstream temperature/conformal postprocessing is excluded for every model. B3 retains FP32, others the same BF16 autocast as frozen evaluations; no quantization or LoRA merging.

## FLOPs and parameters
torch.utils.flop_counter counts matrix products and attention with2 FLOP/MAC. Embedding lookup, normalization, activation, softmax and elementwise/CPU feature preparation are excluded. Native MHA inference fastpath is expanded only for profiling, not timing; every cell has identical logits and argmax before/after expansion. First two smoke attempts returned0 FLOPs in inference_mode despite equal logits; no_grad fixed instrumentation. Failed attempts and costs are retained. B5c profile includes the frozen1.7B decoder and LoRA, not merely the small adapter.

CLR trainable/total17180931 shared across Q1–Q4. B3 active per-type107777. B6 active per-type661249 trainable, plus38682624 frozen ModernBERT option-embedding entries, total39343873 (the table is CPU resident; FLOPs exclude its precomputed mean lookup). B5c trainable18876928, total1739451904, shared across Q3/Q4. These are unique parameter counts, not activation memory. Deploying both B3/B6 types requires two independent probe parameter sets; the frozen B6 table can be shared. CLR/B5c weights must not be summed across qtypes.

The denominator profiles the frozen CODI-GPT2 checkpoint by matrix shapes:147 matrices, unmerged LoRA, projection and vocabulary head,287176704 linear FLOPs. Attention adds36864×(pad_to+2+t), including one cached query position, averaged over the same panel. No source forward was executed, no prompt prefill included. source_profile_reproduced.json is generated by the YAML-driven aggregation and matches the initial static count in source_profile.json; the latter is not presented as a recorded forward. Source checkpoint SHA and matrix shapes are retained.

## Historical training costs
paper_inputs/historical_training.csv retains three seeds and run IDs; training_cost_summary.json gives sums and median seed costs. CLR uses reports/M41_costs.csv completed main/lite train rows; selected checkpoint epoch8 does not mean training stopped there (early stopping ran9 epochs). B6 costs match full seed refs in M10 ledger, include all four qtypes/two learning-rate sweeps, initialization and val. B5c costs match full seed refs in M10b ledger, Q3/Q4 single full pass, loading/validation included. Pure per-type train seconds are also exported where history provides them, but shared startup/validation is not prorated. Historical failures/smokes/evaluation are not called completed-model training; full M10b including them cost17.245090245821153GPUh. Original B3 all-baseline shared run cost0.5343011922869442GPUh; B3 alone is unidentifiable, reported NA. These training workloads/exposures are not matched; no per-type cost is guessed.

## Frozen conformal pairing and uncertainty
Input CLR=data/M10_v1/paper_inputs/conformal_problem.csv, B5c=data/M10b_v1/paper_inputs/B5c_conformal_problem.csv. Use gpt2 Q3/Q4, both reader_val and test, exact(source,seed,split,qtype,problem_id,t) outer join, requiring one selected instance on both sides. No labels, predictions, selected queries, temperatures or cal_B thresholds change. Differences use B5c−CLR. Bootstrap10000 entire problems, preserve allsteps and fixedseed results, micro within t then macro across steps. It quantifies problem uncertainty conditional on the three trained seeds, not training-seed population uncertainty. Reader_val is diagnostic, no fresh exchangeable-data guarantee. Coverage differs, so smaller sets do not prove equal-coverage dominance. Accuracy similarity is descriptive: the pairedCI crosses0; equivalence was not tested.

## Integrity and reproduction
Old paper and FINAL_SUMMARY are byte-preserved in m10b_paper_snapshot before authorized updates. Existing freezes are untouched. upstream_audit.json verifies original scientific assets plus approved archived paper versions; full byte pass is false due only to disclosed pre-existing M10 pytest stdout and M10b completion stdout/LIVE administrative drifts. No attempt was made to fabricate original missing logs or rewrite old hashes. M6 frozen Qwen resource registry is checked post-start (explicitly not called a pre-start snapshot). New paper products rebuild in independent .venv-paper without torch/transformers to two roots byte-for-byte. TeX compilation was not executed because no compiler is installed. Input SHA, config/script hashes and old freeze inventory are indexed by paper/README.md and artifact_manifest.json.

'''
    details+='## Measured cost records\n\n'+f.to_csv(index=False)+'\n## Paired conformal estimates\n\n'+cp.to_csv(index=False)
    Path('reports/M10c_details.md').write_text(details)
    status='M10c只读开销测量与T6/配对conformal汇总完成待审查；无训练/重校准，科学结果永久冻结。唯一审查reports/M10c_REVIEW.md；冻结data/M10c_v1/freeze.json。'
    for p in [Path('AGENTS.md'),Path('docs/STATUS.md')]:
        text=p.read_text();a=text.find('<!-- M10c_LIVE -->');b=text.find('<!-- /M10c_LIVE -->');block='<!-- M10c_LIVE -->\n'+status+'\n<!-- /M10c_LIVE -->'
        p.write_text(text[:a]+block+text[b+len('<!-- /M10c_LIVE -->'):] if a>=0 else text+'\n'+block+'\n')
    write_json('docs/M10c_LIVE.json',dict(milestone='M10c',stage='complete_pending_review',no_training=True,no_recalibration=True,no_new_test_inference=True,
               measurement_run=measure['run'],gpu_hours=total,review=c['review'],freeze=str(root/'freeze.json')))
    with Path('docs/EXPERIMENTS.md').open('a') as out:out.write(f'\n## M10c 只读精度–开销\n固定reader_val64，同卡RTX4090、batch1/64三轮中位；无训练/新test推断/重校准。\nCLR对B5c单查询FLOPs低{ratio["Q3_b1"]["flops"]:.3g}/{ratio["Q4_b1"]["flops"]:.3g}倍，精度CI沿用三seed；conformal整problem bootstrap10000。\n新增GPU成本{total:.3g}h含失败，源前向0；命令configs/M10c/measure.yaml、summarize.yaml、finish.yaml。\n表/配对差/原始三轮/局限见reports/M10c_REVIEW.md；冻结data/M10c_v1/freeze.json。\n')
    with Path('docs/BLOCKERS.md').open('a') as out:out.write('\n## M10c 开销口径与完整性例外\nB3独立训练账本缺失，NA；GPU共享，未测独占/端到端预处理延迟。\n两次smoke FLOP计数为0，no_grad修复且完整计费；trace与精确口径见reports/M10c_details.md。\n启动前已有M10b完成stdout/LIVE哈希漂移，不改旧freeze；科学哈希一致，见data/M10c_v1/upstream_audit.json。\nM10 pytest日志既有例外已按授权加FINAL_SUMMARY；独立paper复现通过，未编译TeX。\n')
    with Path('docs/CHANGELOG.md').open('a') as out:out.write('\n## M10c 只读开销产物\n新增冻结权重推理计数/测速和已有CSV配对汇总，T6及C1成本措辞；原模型/科学结果保持永久冻结。\n旧paper/总结先归档，M10日志漂移与M10b行政文件例外显式披露；未训练或重校准。\n')
    print('Review bytes',len(body.encode()),'FINAL_SUMMARY bytes',len(Path(c['summary']).read_bytes()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');main(load_config(p.parse_args().config))

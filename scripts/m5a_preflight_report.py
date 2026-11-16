"""Produce an explicitly incomplete M5a review, never infer val300 accuracy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json


def read(p):
    return json.loads(Path(p).read_text())


def main(c):
    with recorded(c) as run:
        out = Path(c['output'])
        out.mkdir(parents=True, exist_ok=True)
        full = Path(Path(c['full_smoke_log']).read_text().splitlines()[0]) if c.get('full_smoke_log') else Path(c['full_smoke'])
        proposal = Path(Path(c['proposal_log']).read_text().splitlines()[0])
        for p in [full, proposal]:
            m = read(p/'metrics.json')
            if m['status'] != 'completed' or not m['gate_passed']:
                raise RuntimeError(f'Preflight evidence incomplete: {p}')
        ledger = []
        for p in sorted(Path('runs').glob('20261105*/config.json')):
            cfg = read(p)
            if cfg.get('milestone') != 'M5a' or cfg.get('stage') == 'preflight_report':
                continue
            m = read(p.parent/'metrics.json')
            interrupted = str(p.parent) == c['interrupted_run']
            progress = read(p.parent/'progress.json') if (p.parent/'progress.json').exists() else {}
            ledger.append({'run': str(p.parent), 'version': cfg['version'],
                'status': 'interrupted_exit_143' if interrupted else m['status'],
                'gpu_hours_recorded': m.get('gpu_hours'),
                'gpu_hours_conservative_charge': c['interrupted_gpu_hours_upper'] if interrupted else
                    max(m.get('gpu_hours', 0), m.get('wall_seconds', 0)/3600),
                'source_forward_calls': m.get('source_forward_calls'),
                'lane_forwards': m.get('lane_forwards'),
                'lane_forwards_lower': progress.get('lane_forwards') if interrupted else m.get('lane_forwards'),
                'lane_count_scope': 'merge numerical gate lanes not captured; GPU wall cost includes them' if cfg['version']=='merged128_v2' else 'all instrumented API lanes',
                'final_cost_missing': interrupted})
        charged = sum(r['gpu_hours_conservative_charge'] for r in ledger)
        estimates = {}
        for name, p in [('full_grid', full), ('reduced_oracle_proposal', proposal)]:
            timings = [read(f)['seconds'] for f in sorted(p.glob('timing_*.json'))]
            seconds = sum(timings)/len(timings)
            raw = seconds*c['questions']*len(c['seeds'])/3600
            total = raw*c['margin']+charged+c['overhead_gpu_hours']
            estimates[name] = {'smoke_run': str(p), 'seconds_per_problem_seed': seconds,
                'timing_seconds': timings, 'smoke_questions': read(p/'metrics.json')['questions'],
                'raw_full_gpu_hours': raw, 'estimated_total_gpu_hours': total,
                'margin': c['margin'], 'prior_conservative_gpu_hours': charged,
                'overhead_gpu_hours': c['overhead_gpu_hours'], 'gate_passed': total <= c['budget_gpu_hours'],
                'authorized_for_full': False,
                'limitation': 'Fixed val smoke panel / one timing seed; engineering extrapolation, not a scientific CI'}
        write_json(out/'budget.json', estimates)
        write_json(out/'full_grid_plan.json', estimates['full_grid'])
        write_json(out/'reduced_oracle_plan.json', estimates['reduced_oracle_proposal'])
        write_json(out/'cost_ledger.json', ledger)
        rawrows = read(full/'results.json')
        frozen = read('data/M23_v1a/freeze.json')['sha256']
        references = [str(Path(read(full/'config.json')['baseline_run'])/f'{name}_Q1_{s}.pt')
                      for name in ['B3','B4'] for s in c['seeds']]
        references.append(str(Path(read(full/'config.json')['baseline_run'])/'normalization.npz'))
        verification = {}
        for p in references:
            actual = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            if p not in frozen or actual != frozen[p]:
                raise ValueError('Frozen baseline hash mismatch: '+p)
            verification[p] = {'sha256': actual, 'matches_M23_freeze': True}
        write_json(out/'input_verification.json', verification)
        columns = ['problem_id', 'method', 'value', 'N', 'b', 'horizon', 'seed', 'source_forwards',
                   'oracle_forwards', 'value_evaluations']
        with Path(c['cost_csv']).open('w') as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for row in rawrows:
                w.writerow({k: row[k] for k in columns})
        tests = Path('reports/M5a_tests_final_preflight.log').read_text().strip().splitlines()[-1]
        a, b = estimates['full_grid'], estimates['reduced_oracle_proposal']
        proposed_budget = math.ceil(a['estimated_total_gpu_hours'])
        detail = f'''# M5a 工程预检与预算冲突（未完成）

本文件不报告 val300 搜索准确率，不回答两项研究问题；唯一审查文件为 `reports/M5a_REVIEW.md`。
完整任务、输入/标签/成本口径见 `docs/M5a_TASK.md`。没有打开test数据、生成test预测或查看本次test指标。

## 已完成

实现S0/S1/S2/S3，N4/8/16、beam2/4、sigma1.5、全步/早期范围、B4/B3(Q1)/oracle K16。
复用冻结M3三seed checkpoint及normalization；B3只接收模板0与训练时的dummy选项`['0']`数值特征。
先扰动反馈、源模型前向，再读真实观测；每条分支保留自身KV。只从冻结val manifest读题，入口allowlist在读文件前拒绝其他数据路径。
单位测试：{tests}。覆盖共享策略续算、观测/反馈区分、K16终点退化、合法答案投票与并列、Q1特征、预算规则及val路径隔离。

## 完整网格预算

正确特征/UUID绑定/批量iid采样版本固定{a['smoke_questions']}题×单seed完整网格，每题61个配置，均值{a['seconds_per_problem_seed']:.6f}秒/题/seed（不含一次模型加载与核对），逐题时间{a['timing_seconds']}。
300题×3seed直接外推{a['raw_full_gpu_hours']:.6f}GPUh；加20%裕量、0.05GPUh公共开销和已有预检保守计费{charged:.6f}GPUh，总估算{a['estimated_total_gpu_hours']:.6f}GPUh，上限10。
因此完整预算关卡未通过，没有启动完整val300。固定小面板计时不能代表严谨吞吐置信区间；没有以此伪造三seed研究结果。

预算备选已形成配置`configs/M5a_oracle_full_proposal_v12.yaml`并只做smoke：B4/B3和S1/S2全部原设置不变；仅S3 oracle固定N16、b4且只做全步范围。
其固定{b['smoke_questions']}题×单seed均值{b['seconds_per_problem_seed']:.6f}秒/题/seed，逐题时间{b['timing_seconds']}；直接外推{b['raw_full_gpu_hours']:.6f}GPUh，含同样裕量/已有成本约{b['estimated_total_gpu_hours']:.6f}GPUh。
备选未获授权应用于完整实验；没有擅自删减原任务。另一选择是将原网格预算明确修订至{proposed_budget}GPUh。

## 优化、失败和限制

- 全程源权重fp32、temperature0、同一cohort左padding；没有TF32/降精度、模型替换或N/b/K回退。
- 显式LoRA合并推理核对：16匹配轨迹答案一致，状态最大绝对差0.000175476。不是逐位等价，更不是全val300数值验收；未来完整运行须保留该限制。
- 编译版数值核对通过（max0.000253677），但计算阶段76.713444秒/题/seed，未见吞吐收益；编译开销计入成本，未选择它跑全量。编译缓存触顶/grad模式重编译警告保留。
- 批量iid协议直接调用同一个`clr.policy.RelativeNoise`处理整批z，仍为每lane独立相对高斯。改变seed寻址和物理batch，不能声称重现原M1逐trajectory随机流。单元测试核对带噪一步输出，实际smoke零噪声续算与原policy一致。
- 原逐lane随机流版本与M1的16轨迹状态逐元素一致。所有工程版本都有启动时源码、配置、环境、git/权重hash快照；不是事后补造。
- Q1 dummy特征最初误用全零；已修复并有与冻结训练接口逐元素相等的测试。早期smoke全部作工程记录，不作科研结果；修正版独立smoke通过。
- 一次实际OOM、若干显存预检查失败及一次入口局部import错误均记录traceback。首轮exec返回143，原因未观测，末尾成本缺失；保守计费0.080GPUh，最终源计数只能给下界。
- 早期`merged128_v2`的合并核对发生在lane计数器安装前；其GPU墙钟/物理API调用包含核对，但lane计数不含核对。总lane计数仅下界，不补造最终计数。
- 共享GPU序号有歧义，最终用GPU UUID显式绑定。保留全部资源快照；没有终止其他项目作业。
- 后续预检门槛显式从14GiB调为11GiB，依据已测8.55GiB峰值；设置CUDA allocator expandable_segments以控制缓存碎片。硬件参数调整不缩小源模型/batch/N/b/K。备选删减的oracle网格另外明确标注，未用于全量。

## 待执行的统计与比较

源前向主轴含prefill、潜步骤、答案解码；probe FLOPs换算由smoke的`compute_profile.json`给出；oracle内部所有K16单列且计入主轴。
独立长S1池按每题累计前向成本匹配搜索；同一预算内取最长前缀，残余不足一条轨迹的预算须报告。4096轨迹上限不够时标缺，不声称匹配。
固定预算点128..32768和全部原始配置点均须保留；不按准确率挑N/b/seed。还需生成accuracy–compute曲线以及按题配对10000次bootstrap CI。
需同时说明终点oracle是使用gold的特权0/1选择；中间K16有MC误差和自适应选择偏差，不能称严格数学上界。
比例分母的oracle增益没有可靠正证据时，不能报告“取得上限百分比”。本预检没有足以支持任何收益/比例的结果。

## 证据

- 正确完整网格smoke：`{full}`。
- 备选smoke：`{proposal}`。
- 完整逐配置成本：`{c['cost_csv']}`；不包含accuracy列，避免将1题smoke误作val300结论。
- 预算和成本：`{out}/budget.json`、`{out}/cost_ledger.json`。
- 运行目录都有config/git/environment/source_snapshot/input_sha256/metrics；中断版本缺失项已标识。
'''
        Path(c['detail']).write_text(detail)
        review = f'''# M5a REVIEW

### (a) 状态一句话

未完成：实现与val-only工程smoke通过；完整val300未启动，实测原网格超过10GPUh，需处理预算/范围冲突。

### (b) 关卡清单

- 通过：仅冻结val路径，代表性smoke {a['smoke_questions']}/300题（{100*a['smoke_questions']/300:.3g}%），无本次test访问；全部完成smoke耗时均<900秒。
- 通过：9项最小测试；候选先前向再读观测，K16终点语义、Q1特征与划分入口检查通过。
- 通过：fp32/cohort协议及明确GPU UUID；LoRA合并16轨迹答案一致，状态max差0.000175。批量iid改变seed寻址，未改噪声分布。
- 未通过：原网格预算。{a['seconds_per_problem_seed']:.3g}秒/题/seed→{a['raw_full_gpu_hours']:.3g}GPUh；加裕量/已有成本约{a['estimated_total_gpu_hours']:.3g}>10.0。预检保守计费{charged:.3g}GPUh，含中断0.0800GPUh预留，非精确总数。
- 未通过（未执行）：val300×3seed、按题95%CI、accuracy–compute曲线及两项研究问题。

### (c) 核心主表（最多3张）

尚无完整val300研究指标，不以smoke填充主表。oracle同计算量相对多数投票的增益、B4/B3取得的比例均不可回答；test按本次用户要求不执行。

### (d) 本里程碑新增的偏离与阻塞

预算冲突；一次OOM及显存预检查失败；一次加载前import错误；首轮exit143的最终成本缺失，已保守预留。
早期Q1 dummy特征误用全零已修复，相关smoke不作研究证据。合并推理非逐位一致；编译虽数值通过但更慢，未采用。详见详细报告与逐run记录。

### (e) 需要审查者决定的问题

选择：①保留10GPUh，S3 oracle只做N16/b4的全步搜索（其余B4/B3、S1/S2原网格不变）；smoke外推含裕量/已有成本约{b['estimated_total_gpu_hours']:.3g}GPUh。②保留完整原网格，将预算修订至{proposed_budget}GPUh。备选仅做smoke，未擅自用于全量。

### (f) 证据路径

- 任务：`docs/M5a_TASK.md`；详细报告：`{c['detail']}`。
- 成本CSV：`{c['cost_csv']}`；预算/成本台账：`{out}/`。
- 完整网格smoke：`{full}`；备选smoke：`{proposal}`。
- 测试：`reports/M5a_tests_final_preflight.log`；配置：`configs/M5a_representative_v11.yaml`、`configs/M5a_oracle_full_proposal_v12.yaml`。
'''
        if len(review.encode()) > 8000:
            raise ValueError('Review exceeds 8000 bytes')
        Path(c['review']).write_text(review)
        write_json(out/'report_manifest.json', {'report_run': str(run.path), 'status': 'incomplete_budget_conflict',
            'review_bytes': len(review.encode()), 'full_grid_started': False, 'test_access': False,
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(c['review']), Path(c['detail']), Path(c['cost_csv']), out/'budget.json', out/'cost_ledger.json']}})
        run.write_metrics(status='completed', milestone_complete=False, source_forward_calls=0, gpu_hours=0,
                          budget=estimates, review_bytes=len(review.encode()), test_access=False)
        print(run.path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    main(load_config(a.config, a.overrides))

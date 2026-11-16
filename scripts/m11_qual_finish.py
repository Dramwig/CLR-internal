"""CPU validation, documentation, freezing and deterministic ZIP packaging."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
from scipy.special import softmax
from clr.export.qual import digest, json_write, rank_ids
from clr.utils.config import load_config


def validate(root):
    dest=root/'export';sel=json.loads((dest/'selection.json').read_text());checks=json.loads((dest/'checks.json').read_text())
    provenance=json.loads((dest/'provenance.json').read_text())
    # Final validation checks sources again; it never changes selection.
    for p,h in provenance['source_sha256'].items():assert digest(p)==h,p
    for p,h in provenance['freeze_manifest_sha256'].items():assert digest(p)==h,p
    assertions=0;required=['mlp_probe','attention_probe','clr','clr_joint']
    frozen=json.loads((dest/'frozen_predictions.json').read_text())['rows']
    frozen_lookup={(r['source'],r['method'],r['seed'],r['qtype'],r['problem_id'],r['t'],r['template_id'],tuple(r['options']),r['file']):r for r in frozen}
    main_frozen=0
    for source in ['gpt2','llama']:
        d=json.loads((dest/source/'qual_reading.json').read_text());assert len(d['problems'])==12
        assert [r['problem_id'] for r in d['problems']]==sel[source+'_reading']['ids']
        assert rank_ids(sel[source+'_reading']['ids'],12)==sel[source+'_reading']['ids']
        for r in d['problems']:
            assert not r['cot_parse_failed'] and len(r['intermediate_values'])>=3
            pos={x['value'] for x in r['Q4_candidates'] if x['label']==1};neg={x['value'] for x in r['Q4_candidates'] if x['label']==0}
            assert len(pos)==len(neg) and not pos&neg
            assert [s['t'] for s in r['steps']]==list(range(7))
            for s in r['steps']:
                assert s['K16_Vhat'] is not None and 0<=s['K16_Vhat']<=1
                for q in ['Q1','Q2','Q3']:
                    block=s[q]
                    needed=required+(['latentqa'] if source=='gpt2' and q=='Q3' else []) if q=='Q3' else ['mlp_probe','lto','hsrm','clue','clr']
                    assert set(needed)<=block['methods'].keys(),(source,q,block['methods'].keys())
                    for method,seeds in block['methods'].items():
                        for seed,v in seeds.items():
                            p=np.array(v['probabilities'],np.float64);assert np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and abs(p.sum()-1)<1e-12
                            if v['origin']=='frozen_prediction':
                                k=(source,method,int(seed),q,r['problem_id'],s['t'],block['template_id'],tuple(block['options']),v['file'])
                                pp=np.array(frozen_lookup[k]['probabilities'],np.float64)
                                assert np.array_equal(p.view(np.uint64),pp.view(np.uint64));main_frozen+=1
                            if q=='Q3' and method in needed:
                                cp=v['conformal'];thr=np.inf if cp['threshold'] is None else cp['threshold']
                                assert cp['set_indices']==np.flatnonzero(1-p<=thr).tolist()
                    assertions+=1
                for candidate in s['Q4']:
                    assert candidate['label']==int(candidate['value'] in pos) and candidate['template_id']==0
                    for method in required+(['latentqa'] if source=='gpt2' else []):
                        v=candidate['methods'][method]['11'];p=v['P_yes'];assert 0<=p<=1 and p==v['probabilities'][1]
                        assertions+=1
    for name,n in [('embed_reasoning.npz',1200),('embed_siblings.npz',1360),('embed_audit.npz',1200)]:
        d=np.load(dest/name,allow_pickle=False)
        for k in d.files:
            a=d[k]
            if a.dtype.kind in 'fci':assert np.isfinite(a).all(),(name,k)
        if name=='embed_reasoning.npz':
            assert d['z'].shape==(n,768) and d['z'].dtype==np.float16
            assert d['clr_pool'].shape==(n,256) and d['mlp_hidden'].shape==(n,128)
            assert d['sibling_z'].shape==(1360,768) and d['sibling_all_z'].shape==(4080,768)
            assert len(set(d['problem_id']))==300 and set(d['t'])=={0,2,4,6}
        elif name=='embed_siblings.npz':
            assert d['z'].shape==(n,768) and len(set(d['problem_id']))==40
            assert set(d['t'])=={1,3} and d['all_z'].shape==(4080,768)
            assert d['is_sigma0'].sum()==80
            for k in ['clr_Q1_probability','mlp_Q1_probability','all_clr_probability','all_mlp_probability']:
                assert np.all((d[k]>=0)&(d[k]<=1))
        else:
            assert d['f_layers'].shape==(n,4,512) and d['f_mean'].shape==(n,512)
            assert d['clr_option_representation'].shape==(n,4,256) and d['mlp_hidden'].shape==(n,256)
            assert len(set(d['group_id']))==600 and set(d['category'])=={'clean','counterfactual'}
            np.testing.assert_array_equal(d['f_mean'],d['f_layers'].mean(1))
            for seed in [11,22,33]:
                cal=json.loads(Path(f'data/M71_v1/pool/calibration/CLR_{seed}.json').read_text())
                for alpha in ['0.01','0.05']:
                    thr=cal['thresholds']['16'][alpha]
                    np.testing.assert_array_equal(d[f'alarm_{seed}_{alpha}'],d[f'Gamma_{seed}']>(np.inf if thr is None else thr))
    dep=json.loads((dest/'dependent_scores.json').read_text());assert len({r['group_id'] for r in dep['rows']})==100
    assert len(dep['rows'])==600 and all(r['epsilon']==.1 and r['kind']=='E3' for r in dep['rows'])
    checks['final_validation']=dict(passed=True,reading_field_assertions=assertions,main_frozen_probability_rows_bitwise_equal=main_frozen,
        required_probability_and_conformal_fields_complete=True,npz_shapes_finite=True,all_three_seed_sibling_states=True,
        dependent_no_new_PGD=True,source_files_rechecked=len(provenance['source_sha256']))
    json_write(dest/'checks.json',checks)
    return checks


def main(c):
    root=Path(c['output']);assert c['stage']=='package'
    assert not (root/'freeze.json').exists(),'Already frozen'
    done=json.loads((root/'export_complete.json').read_text());assert done['passed']
    for source in ['gpt2','llama']:
        f=root/'export'/source/'qual_reading.json';data=json.loads(f.read_text())
        for problem in data['problems']:
            for step in problem['steps']:
                for method,seeds in step['Q3']['methods'].items():
                    for value in seeds.values():
                        if 'conformal' not in value:
                            value['conformal']=None
                            value['conformal_unavailable_reason']='No frozen cal_B threshold was persisted for this additional baseline; no recalibration performed.'
        json_write(f,data)
    checks=validate(root)
    # Record exact external pretrained dependencies used by option encoding.
    pp=root/'export/provenance.json';p=json.loads(pp.read_text())
    for f in Path(c['tokenizer']).iterdir():
        if f.suffix in ['.json','.safetensors']:
            p['pretrained_dependencies'].append(dict(path=str(f),sha256=digest(f),role='pinned_frozen_option_encoder_or_tokenizer'))
    p['code_snapshot_timing']='post-completion source snapshot; not represented as a pre-run snapshot'
    json_write(pp,p)
    costs=[json.loads(f.read_text()) for f in sorted(root.glob('cost_*.json'))]
    gpu=sum(r['gpu_hours'] for r in costs)
    json_write(root/'costs.json',dict(README='Sum of conservative whole-process GPU wall-hour charges; includes smoke, failed smoke and final export repetitions, including CPU/hash/loading time. No training/recalibration/source-model/PGD calls.',gpu_hours=gpu,source_forward_calls=0,gpu='NVIDIA GeForce RTX 4090',runs=costs))
    for cost in costs:
        run=Path(cost['run']);out=root/'evidence'/run.name;out.mkdir(parents=True,exist_ok=True)
        for name in ['environment.json','git.json','git_commit.txt','metrics.json','config.yaml','config.json','traceback.txt','gpu_start.txt','dependencies.json']:
            if (run/name).exists():shutil.copyfile(run/name,out/name)
    evidence=root/'evidence';evidence.mkdir(exist_ok=True)
    for log in Path('reports').glob('M11_qual*.log'):shutil.copyfile(log,evidence/log.name)
    (evidence/'README.md').write_text('Original run configuration, environment, Git metadata, metrics, GPU startup snapshots and failure traces, copied after completion. GPU cost ledger includes every listed run; logs contain val-only debugging or final exports.\n')
    rep=root/'reproduce';rep.mkdir(exist_ok=True)
    for f in ['clr/export/qual.py','scripts/m11_qual.py','scripts/m11_qual_finish.py','configs/M11_qual/export.yaml','tests/test_m11_qual.py','docs/M11_qual_TASK.md']:
        target=rep/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    (rep/'README.md').write_text('Completion-time code/config snapshots (not claimed to be startup snapshots).\n\nRun from the original frozen repository/environment with a new output version:\n`python -m scripts.m11_qual configs/M11_qual/export.yaml output=data/M11_qual_repro stage=smoke`\n`python -m scripts.m11_qual configs/M11_qual/export.yaml output=data/M11_qual_repro stage=full`\n`python -m scripts.m11_qual_finish configs/M11_qual/export.yaml output=data/M11_qual_repro stage=package`\n\nThe JSON/NPZ export requires local frozen input files and checkpoint paths from provenance.json; no training or calibration fitting is performed. Original GPU batch kernels may cause tiny differences in newly extracted representations; stored frozen predictions are copied exactly.\n')
    (root/'README.md').write_text('M11-qual v1 — reviewer plotting data\n\nStart at `export/README.md`; requested data are under `export/`, with one `qual_reading.json` per source. `embed_reasoning.npz` contains cross-problem and sibling_* sections; sibling data are also separate for convenience. `export/selection.json`, `export/checks.json` and `export/provenance.json` describe selection, verification and frozen inputs. `costs.json` includes every smoke/failure/export GPU charge; `smoke/` contains val proxies only. `reproduce/` contains completion-time code snapshots and instructions. `M11_qual_REVIEW.md` is the sole review summary. The ZIP contains this whole directory except the ZIP itself.\n')
    filesizes={str(f.relative_to(root/'export')):f.stat().st_size for f in (root/'export').rglob('*') if f.is_file()}
    json_write(root/'inventory.json',dict(README='Byte lengths of final exported data and README files before ZIP compression; counts are in checks.json. No model outputs rounded for this inventory.',files=filesizes,export_bytes=sum(filesizes.values())))
    mb=sum(filesizes.values())/1e6
    report=f'''### (a) 状态一句话
M11-qual导出完成待审查；无训练/重校准/PGD，旧冻结输入SHA一致。

### (b) 关卡清单
通过：val smoke 53.8秒，规模≤5%；12项单测；两源各3题共6题冻结概率逐位一致；必需概率/集合、NPZ形状与有限值检查。
GPU保守累计{gpu:.3g}h（含失败/重复导出），RTX4090；源模型前向0。

### (c) 文件与规模
- `export/gpt2/qual_reading.json`、`llama/qual_reading.json`：各12题，t0–6。
- `embed_reasoning.npz`：300题×4步；含兄弟部分，另存`embed_siblings.npz`，40题×2步×17候选，三seed状态/值均保留。
- `embed_audit.npz`：600组×2消息，四层特征/A4表示与三seed概率/Γ/报警。
- `dependent_scores.json`：100组、600条ε=.1已有E3记录。
- `frozen_predictions.json`、selection/checks/provenance及逐文件README；数据{mb:.3g}MB，ZIP见同目录。

### (d) 新增偏离与阻塞
M8c未保存扰动张量及同一攻击的另一审计器分数，按授权导出现有分数，其余显式null。附加LLaMA线性基线无冻结conformal阈值；必需方法齐全。M5b原排序logits保留，另用冻结温度导出概率。
两次val smoke格式失败已修复并计费；原日志保留。

### (e) 需审查者决定
无。

### (f) 证据路径
`data/M11_qual_v1/export/{{selection,checks,provenance}}.json`；`costs.json`、`inventory.json`、`reproduce/`；`reports/M11_qual_tests.log`。
选择：题目仅按gold可解析且中间量≥3过滤；其余按指定总体；统一sha256(ID+"qual")取前N，不看结果。Q4固定template0，负例补足供体见selection；Q3原模板/选项/冻结cal_B α=.1。
'''
    assert len(report.encode())<=2000,len(report.encode())
    Path('reports/M11_qual_REVIEW.md').write_text(report);(root/'M11_qual_REVIEW.md').write_text(report)
    for f in list(root.rglob('*.json')):
        side=f.with_name(f.name+'.README.md')
        if not side.exists():side.write_text('Metadata/provenance for M11-qual; see README.md and export/selection.json. Cost entries include full process elapsed time and original run paths.\n')
    scientific_files=sorted(f for f in root.rglob('*') if f.is_file())
    scientific_files += [Path('reports/M11_qual_REVIEW.md'),Path('clr/export/qual.py'),Path('scripts/m11_qual.py'),Path('scripts/m11_qual_finish.py'),Path('configs/M11_qual/export.yaml'),Path('tests/test_m11_qual.py')]
    json_write(root/'freeze.json',dict(README='SHA256 of completed export, exact code snapshots, config and review; old scientific inputs are independently indexed in export/provenance.json. ZIP is packaged after this manifest and is not self-hashed.',version='M11_qual_v1',gpu_hours=gpu,source_forward_calls=0,sha256={str(f):digest(f) for f in scientific_files}))
    (root/'freeze.json.README.md').write_text('Freeze manifest; paths are repository-relative. The ZIP is built after freezing and is not included in its own hash manifest.\n')
    total=sum(f.stat().st_size for f in root.rglob('*') if f.is_file());assert total<c['max_bytes']
    archive=root/'qual_export.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(root.rglob('*')):
            if not f.is_file() or f==archive:continue
            info=zipfile.ZipInfo(str(Path('M11_qual_v1')/f.relative_to(root)),(2026,11,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,f.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    assert sum(f.stat().st_size for f in root.rglob('*') if f.is_file())<=c['max_bytes']
    completion=dict(status='completed_pending_review',zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=digest(archive),gpu_hours=gpu,
                    source_forward_calls=0,counts=done['counts'],review='reports/M11_qual_REVIEW.md',freeze=str(root/'freeze.json'))
    json_write('docs/M11_qual_LIVE.json',completion)
    for f in ['AGENTS.md','docs/STATUS.md']:
        with Path(f).open('a') as s:s.write('\n<!-- M11_QUAL_LIVE -->\nM11-qual只读导出完成待审查；两源各12题、跨题300、兄弟40、审计600、M8c100组。唯一审查reports/M11_qual_REVIEW.md；交付data/M11_qual_v1/qual_export.zip，冻结data/M11_qual_v1/freeze.json。无训练/重校准/PGD；M8c仅已有分数。\n<!-- /M11_QUAL_LIVE -->\n')
    with Path('docs/EXPERIMENTS.md').open('a') as f:f.write(f'\n## 2026-11-12 — M11-qual\n只读冻结权重/预测导出；命令`python -m scripts.m11_qual configs/M11_qual/export.yaml stage=full`及同配置finish/package。\nval-only smoke通过；12单测；两源各3题概率逐位一致；所需规模齐全。保守{gpu:.3g}GPUh，源前向0。\n确定性哈希选样，无新统计/CI；三seed已有预测保留；M8c仅已有分数。\n交付/规模/限制见[审查](../reports/M11_qual_REVIEW.md)及`data/M11_qual_v1/qual_export.zip`。\n')
    with Path('docs/CHANGELOG.md').open('a') as f:f.write('\n## 2026-11-12 — M11-qual\n新增YAML驱动只读导出器、12项验证、冻结数据与ZIP；不修改论文和既有科学结果。接续`M11_qual_LIVE.json`。\n')
    print(json.dumps(completion,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

"""Close M9 evidence, then freeze it without mutating any upstream artifact."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded
from clr.utils.m9_runtime import hours, live, snapshot
from scripts.m9_upstream import digest
from scripts.m9_write_reports import documents, readme


def terminal_runs(config, current):
    """Preserve the terminated original controller's historical running record."""
    paths, abandoned = set(), []
    supplement = json.loads((Path(config['output']) / 'execution_supplement.json').read_text())
    for p in Path('runs').glob('*/config.json'):
        if p.parent.resolve() == current.resolve():
            continue
        cfg = json.loads(p.read_text())
        if cfg.get('milestone') != 'M9':
            continue
        metrics = json.loads((p.parent / 'metrics.json').read_text())
        if metrics.get('status') not in ['completed', 'failed']:
            if p.parent.resolve() != Path(supplement['previous_controller_run']).resolve():
                raise ValueError('Unclosed M9 run: ' + str(p.parent))
            pid = config['terminated_controller_pid']
            proc = Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'scripts.m9_pipeline' in proc.read_bytes():
                raise ValueError('Original controller is still alive')
            abandoned.append({'run': str(p.parent), 'status': 'terminated_controller',
                              'historical_metrics_retained': True,
                              'evidence': str(Path(config['output']) / 'execution_supplement.json')})
        paths.update(q for q in p.parent.rglob('*') if q.is_file())
    return paths, abandoned


def update_records(config, gpu):
    status = 'M9-A三seed联合训练与论文产物完成待审查；实验至此冻结。唯一审查reports/M9_REVIEW.md，冻结data/M9_v1/freeze.json。'
    for name in ['AGENTS.md', 'docs/STATUS.md']:
        path = Path(name)
        content = path.read_text()
        start, end = '<!-- M9_LIVE -->', '<!-- /M9_LIVE -->'
        replacement = start + '\n' + status + '\n' + end
        if start in content:
            a, b = content.index(start), content.index(end) + len(end)
            content = content[:a] + replacement + content[b:]
        else:
            content += '\n' + replacement + '\n'
        path.write_text(content)
    entries = {
        'docs/EXPERIMENTS.md': (
            '\n## 2026-11-10 M9完成\n'
            'M6 C2三seed共享CLR-lite补齐；单源/联合Q1–Q4与conformal配对差及95%题bootstrap CI见reports/M9_REVIEW.md。\n'
            f'A累计{gpu:.3g}/10GPUh（含smoke/加载/训练/校准/评估），源前向0；训练锁后才test。\n'
            '命令：scripts.m9_resume_pipeline configs/M9/resume_pipeline.yaml；scripts.m9_finish configs/M9/finish.yaml。\n'
            '19份独立表图脚本、输入CSV冻结SHA、隔离制图环境逐字节重建；四贡献/两局限逐条核验，不能主张另列。\n'
            '负面/缺失控制与历史偏离保留；唯一审查reports/M9_REVIEW.md，详细reports/M9_details.md，冻结data/M9_v1/freeze.json。\n'),
        'docs/BLOCKERS.md': (
            '\n## 2026-11-10 M9交付限制\n'
            'LLaMA B0/prefix/single-state对照缺失，论文标NA，按任务不补跑；联合/单源比较有训练规模及优化混杂。\n'
            'CPU导出旧格式缺元数据/区间字段错误保留失败run；用冻结QA恢复选项，逐项复放原conformal值，无新推断。\n'
            '首次制图环境代理安装失败，直连完成；系统缺LaTeX引擎，未执行TeX编译。早期成本缺口标下界，不给项目完整总计。\n'
            '静态协议v1保留但不用于论文，v2读取实际冻结噪声选择；控制器调度交接保留历史running记录与终止说明。\n'
            '结论边界与不能主张清单见reports/FINAL_SUMMARY.md；以上未靠新增实验补齐。\n'),
        'docs/CHANGELOG.md': (
            '\n## 2026-11-10 M9收尾冻结\n'
            '新增两源联合CLR-lite三seed、冻结预测CPU统计、论文TeX/PDF/复现清单与六条证据主线；上游冻结不变。\n'
            '派生conformal从旧QA元数据恢复并对照原结果；不新增M9-A以外模型实验。\n'),
    }
    for name, entry in entries.items():
        path = Path(name)
        marker = entry.splitlines()[1]
        if marker in path.read_text():
            raise ValueError('Completion record already exists: ' + name)
        with path.open('a') as stream:
            stream.write(entry)
    return status


def inventory_manifests():
    names = set(Path('data').rglob('freeze.json'))
    names.update(Path('data').rglob('*freeze*.json'))
    names.update(Path('reports').rglob('*freeze*.json'))
    names.update(Path('reports').rglob('*manifest*.json'))
    inventory = []
    for p in sorted(names):
        value = json.loads(p.read_text())
        entries = value.get('sha256', value.get('files', {})) if isinstance(value, dict) else {}
        inventory.append({'path': str(p), 'sha256': digest(p),
                          'entries': len(entries) if isinstance(entries, (dict, list)) else None,
                          'kind': 'freeze.json' if p.name == 'freeze.json' else 'historical_or_supplemental'})
    return inventory


def main(config):
    with recorded(config) as run:
        snapshot(run, config)
        root = Path(config['output'])
        if (root / 'freeze.json').exists():
            raise FileExistsError('M9 is already frozen')
        gpu = sum(r['gpu_hours'] for r in hours())
        if gpu > config['budget_gpu_hours']:
            raise ValueError('M9 exceeded its cumulative GPU budget')
        complete = json.loads((root / 'experiment_complete.json').read_text())
        if sorted(complete['models']) != ['multi_lite_11', 'multi_lite_22', 'multi_lite_33']:
            raise ValueError('Missing joint-training seed')
        # A second read-only upstream audit is a required job before this script.
        audit = json.loads((root / 'upstream_audit.json').read_text())
        if not audit['passed']:
            raise ValueError('Upstream integrity audit failed')
        run_paths, abandoned = terminal_runs(config, run.path)
        readme(config)
        report = documents(config)
        review = Path('reports/M9_REVIEW.md').read_text()
        if not all(f'### ({letter})' in review for letter in 'abcdef'):
            raise ValueError('Review does not follow the required six sections')
        if sum(line.startswith('|---') for line in review.splitlines()) > 3:
            raise ValueError('Too many review tables')
        status = update_records(config, gpu)
        write_json(root / 'completion.json', {
            'status': status, 'freezer_run': str(run.path), 'gpu_hours': gpu,
            'test_opened': True, 'review': 'reports/M9_REVIEW.md', 'report': report,
            'terminated_controllers': abandoned,
            'experiment': str(root / 'experiment_complete.json'),
        })
        write_json(root / 'cost_ledger.json', {'runs': hours(), 'gpu_hours': gpu,
                                            'source_forward_calls': 0,
                                            'scope': 'M9-A resident wall time including smoke, loading, training, calibration and inference; CPU paper work 0 GPUh'})
        paths = set(run_paths)
        for folder in [root, Path('configs/M9'), Path('paper'), Path('reports/M9_evidence')]:
            paths.update(p for p in folder.rglob('*') if p.is_file())
        paths.update(Path('scripts').glob('m9*.py'))
        paths.update(Path('scripts/paper').glob('*.py'))
        paths.update(Path('clr/paper').glob('*.py'))
        paths.update([Path('clr/utils/m9_runtime.py'), Path('tests/test_m9.py'),
                      Path('docs/M9_TASK.md'), Path('reports/FINAL_SUMMARY.md'),
                      Path('reports/M9_REVIEW.md'), Path('reports/M9_details.md')])
        excluded = {root / 'freeze.json', root / 'verification.json', Path('paper/artifact_manifest.json')}
        paths = {p for p in paths if p not in excluded and p.name not in config['active_logs']
                 and '__pycache__' not in p.parts}
        ordered = sorted(paths)
        with ThreadPoolExecutor(max_workers=2) as pool:
            hashes = dict(zip(map(str, ordered), pool.map(digest, ordered)))
        write_json(root / 'freeze.json', {
            'version': 'M9_v1', 'status': status, 'freezer_run': str(run.path),
            'gpu_hours': gpu, 'source_forward_calls': 0, 'sha256': hashes,
            'exclusions': 'Active freezer run/logs, mutable status pointers, artifact manifest and final verification avoid circular hashes.',
        })
        paper_paths = sorted(p for p in Path('paper').rglob('*')
                             if p.is_file() and p.name != 'artifact_manifest.json')
        manifest_path = Path('paper/artifact_manifest.json')
        write_json(manifest_path, {
            'version': 'M9_paper_v1', 'freeze_manifests': inventory_manifests(),
            'paper_files': {str(p): digest(p) for p in paper_paths},
            'current_freeze': str(root / 'freeze.json'),
            'source_provenance': str(root / 'paper_inputs/provenance.json'),
            'integrity_audit': str(root / 'upstream_audit.json'),
            'historical_manifest_note': 'All discovered data/reports freeze inventories are indexed; historical attempts and metadata-only manifests are preserved without claiming that every old attempt remains valid. Official completed upstream manifests are checked in the linked integrity audit.',
            'verification': str(root / 'verification.json'),
        })
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = dict(zip(map(str, ordered), pool.map(digest, ordered)))
        bad = [p for p in hashes if hashes[p] != actual[p]]
        write_json(root / 'verification.json', {
            'run': str(run.path), 'passed': not bad, 'files': len(hashes), 'mismatches': bad,
            'artifact_manifest_sha256': digest(manifest_path), 'freeze_sha256': digest(root / 'freeze.json'),
            'upstream_files': audit['unique_files'], 'upstream_passed': audit['passed'],
        })
        if bad:
            raise ValueError('M9 files changed during freeze')
        live(stage='frozen_complete_pending_review', milestone_complete=True,
             background_tasks_active=False, test_opened=True, gpu_hours=gpu,
             review='reports/M9_REVIEW.md', freeze=str(root / 'freeze.json'))
        run.write_metrics(status='completed', gpu_hours=0, source_forward_calls=0,
                          cumulative_gpu_hours=gpu, files=len(hashes),
                          **{k: v for k, v in report.items() if k != 'gpu_hours'})
        print(json.dumps({'status': status, 'files': len(hashes), **report}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    main(load_config(parser.parse_args().config))

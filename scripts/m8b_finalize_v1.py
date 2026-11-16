"""Validate the locked matrix, build evidence, and freeze M8b without model selection."""
import sys
sys.dont_write_bytecode = True
import argparse, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.audit.features import sha256
from clr.audit.m8b_runtime import snapshot, Cost, total_cost, live
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def verify_matrix(root, spec, c):
    expected = {(kind, eps, d, seed, 1, step)
                for kind in ['E3'] for eps in c['epsilons']
                for d in c['designs'] for seed in c['reader_seeds'] for step in c['steps']}
    expected |= {(kind,eps,None,None,0.,step) for kind in ['E1','E2']
                 for eps in c['epsilons'] for step in c['steps']}
    groups = {}; references = sorted(root.glob(spec['pattern']))
    assert len(references) == spec['parts'], references
    for refpath in references:
        ref = json.loads(refpath.read_text()); seen = set()
        with (Path(ref['run'])/'results.jsonl').open() as handle:
            for line in handle:
                r = json.loads(line); gid = r['group_id']; seen.add(gid)
                key = (r['kind'],r['epsilon'],r['attack_design'],r['attack_seed'],r['lambda'],r['step_budget'])
                bucket = groups.setdefault(gid,set()); assert key not in bucket
                bucket.add(key)
                assert r['max_relative_norm'] <= r['epsilon']*(1+1e-5)
                if r['kind'] == 'E2': assert r['original_feature_difference']['fp16_bitwise_equal']
                assert len(r['qa']) == 12
        assert len(seen) == ref['groups']
        metrics = json.loads((Path(ref['run'])/'metrics.json').read_text())
        assert metrics['status'] == 'completed'
    assert len(groups) == spec['groups'], len(groups)
    assert all(keys == expected for keys in groups.values())
    if spec['split'] == 'test':
        assert set(groups) == set(json.loads(Path('data/M8a_v1/test500_ids.json').read_text())['group_ids'])
    return {'groups':len(groups),'rows_per_group':len(expected),'parts':len(references),'passed':True}


def main(c):
    with recorded(c) as run:
        snapshot(run,c); cost = Cost(run,c,gpu=False); root = Path(c['output'])
        try:
            assert not (root/'freeze.json').exists()
            lock = json.loads((root/'final_lock.json').read_text())
            for path,digest in lock['sha256'].items(): assert sha256(path) == digest, path
            audit = {s['split']:verify_matrix(root,s,c) for s in c['matrix_specs']}
            write_json(root/'matrix_verification.json',audit)
            for job in c['finalization_jobs']:
                with Path(job['log']).open('x') as handle:
                    subprocess.run([sys.executable,'-m',job['module'],job['config']],stdout=handle,stderr=subprocess.STDOUT,check=True)
            review = Path('reports/M8b_REVIEW.md'); text = review.read_text()
            assert len(review.read_bytes()) <= 8000
            assert all(f'### ({letter})' in text for letter in 'abcdef')
            assert sum(line.startswith('|---') for line in text.splitlines()) <= 3
            assert json.loads((root/'upstream_final.json').read_text())['passed']
            assert total_cost() <= 25
            result = json.loads((root/'completion.json').read_text())
            result['status'] = 'completed_pending_review'; result['matrix_verification'] = str(root/'matrix_verification.json')
            result['freeze'] = str(root/'freeze.json'); write_json(root/'completion.json',result)
            status = 'M8b四种设计/三seed、固定500组E1/E2/E3评估与统计完成待审查；唯一审查reports/M8b_REVIEW.md，冻结data/M8b_v1/freeze.json。'
            for filename in ['AGENTS.md','docs/STATUS.md']:
                p = Path(filename); content = p.read_text(); marker = '<!-- M8B_LIVE -->'; ending = '<!-- /M8B_LIVE -->'
                if marker in content:
                    start = content.index(marker); end = content.index(ending,start)
                    content = content[:start]+marker+'\n'+status+'\n'+content[end:]
                else: content += '\n'+marker+'\n'+status+'\n'+ending+'\n'
                p.write_text(content)
            with Path('docs/EXPERIMENTS.md').open('a') as handle:
                handle.write('\n## 2026-11-09 M8b 设计比较完成\nR1随机问集、R2随机读取、R3全层与R4接收方状态；三seed，test500/PGD val20，clean QA val200/test500。\n命令configs/M8b/final_*.yaml→configs/M8b/statistics_final.yaml；问答、FPR、E1/E2/E3失效及10000次组bootstrap 95%CI见reports/M8b_REVIEW.md。\n全部失败/额外训练/加载及孤儿成本上界计入25GPUh；完整负面结果、k/ε/步数曲线、时延和复现证据见reports/M8b_details.md。\n冻结data/M8b_v1/freeze.json；上游62363项最终SHA复核一致，不更改M8a/M71/M7/M6。\n')
            with Path('docs/BLOCKERS.md').open('a') as handle:
                handle.write('\n## 2026-11-09 M8b 完成限制\nR2/R4语义读取未达近完美；新适配器的训练负例修复、初始化与f维度均可能影响设计比较，负面结果完整保留。\nR1均值NLL规则不满足独立单问veto指数定理；有限PGD/EOT与成功早停只给经验失效，不能证明最坏情况鲁棒性。\nE1/E2针对原f盲区；R4需先消费消息，不能在消费前拒收。共享GPU时延、CPU/GPU舍入修复及孤儿run成本缺证据见reports/M8b_REVIEW.md。\n')
            with Path('docs/CHANGELOG.md').open('a') as handle:
                handle.write('\n## 2026-11-09 M8b冻结\n新增鲁棒读取设计、三seed适配器、公开期望校准、锁定自适应评估及审查证据；上游冻结只读，完整协议docs/M8b_TASK.md。\n')
            paths = set()
            for folder in [root,Path('configs/M8b'),Path('reports/M8b_evidence')]:
                paths.update(p for p in folder.rglob('*') if p.is_file())
            paths.update(p for p in Path('reports').glob('M8b*') if p.is_file())
            paths.update(Path('clr').rglob('*m8b*.py')); paths.update(Path('scripts').glob('m8b*.py'))
            paths.update([Path('tests/test_m8b.py'),Path('docs/M8b_TASK.md')])
            orphan = Path(json.loads((root/'orphan_cost_upper.json').read_text())['run']).resolve()
            for p in Path('runs').glob('*/config.json'):
                if p.parent.resolve() == run.path.resolve(): continue
                cfg = json.loads(p.read_text())
                if cfg.get('milestone') != 'M8b': continue
                met = json.loads((p.parent/'metrics.json').read_text())
                assert met['status'] in ['completed','failed'] or p.parent.resolve() == orphan, p.parent
                paths.update(q for q in p.parent.rglob('*') if q.is_file())
            excluded = {'freeze.json','verification.json','finalize_v1.log'}
            paths = {p for p in paths if p.name not in excluded}
            with ThreadPoolExecutor(max_workers=2) as executor:
                digests = list(executor.map(sha256,sorted(paths)))
            frozen = {str(p.resolve()):h for p,h in zip(sorted(paths),digests)}
            write_json(root/'freeze.json',{'version':'M8b_v1','status':'completed_pending_review',
                'freezer_run':str(run.path),'gpu_hours_conservative':total_cost(),
                'artifact_bytes':sum(Path(p).stat().st_size for p in frozen),'sha256':frozen})
            with ThreadPoolExecutor(max_workers=2) as executor:
                current = list(executor.map(sha256,[Path(p) for p in frozen]))
            bad = [p for (p,h),actual in zip(frozen.items(),current) if h != actual]
            write_json(root/'verification.json',{'passed':not bad,'files':len(frozen),'mismatches':bad,'run':str(run.path)})
            assert not bad, bad
            for path,digest in lock['sha256'].items(): assert sha256(path) == digest, path
            cost.update(status='completed')
            live(stage='frozen_complete_pending_review',background_tasks_active=False,milestone_complete=True,
                 review=str(review),freeze=str(root/'freeze.json'),verification=str(root/'verification.json'),
                 gpu_hours_conservative=total_cost(),finalizer_run=str(run.path))
            print('M8b completed and frozen',len(frozen),'files',flush=True)
        except Exception:
            cost.update(status='failed'); live(stage='finalize_failed',milestone_complete=False,run=str(run.path)); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('config'); a=p.parse_args(); main(load_config(a.config))

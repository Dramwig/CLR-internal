"""Complete evidence and freeze only completed, budgeted M8c results."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.audit.features import sha256
from clr.audit.m8c_runtime import snapshot,Cost,total_cost,live
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output'])
        try:
            assert not (root/'freeze.json').exists()
            lock=json.loads((root/'final_lock.json').read_text())
            for p,h in lock['sha256'].items():assert sha256(p)==h,p
            for job in c['jobs']:
                with Path(job['log']).open('x') as f:subprocess.run([sys.executable,'-m',job['module'],job['config']],stdout=f,stderr=subprocess.STDOUT,check=True)
            assert total_cost()<=c['budget_gpu_hours']
            review=Path('reports/M8c_REVIEW.md');txt=review.read_text()
            assert len(review.read_bytes())<=8000 and all(f'### ({x})' in txt for x in 'abcdef')
            assert sum(l.startswith('|---') for l in txt.splitlines())<=3
            assert json.loads((root/'upstream_final.json').read_text())['passed']
            status=c['status_statement']
            for file in ['AGENTS.md','docs/STATUS.md']:
                p=Path(file);content=p.read_text();start='<!-- M8C_LIVE -->';end='<!-- /M8C_LIVE -->'
                a=content.index(start);b=content.index(end,a)
                p.write_text(content[:a]+start+'\n'+status+'\n'+content[b:])
            with Path('docs/EXPERIMENTS.md').open('a') as f:
                budget=json.loads((root/'budget_lock.json').read_text())
                f.write(f"\n## 2026-11-09 M8c 依赖设置鲁棒性\nshort direct-lookup准入val200：有/无/错配93.0/27.0/26.0%；R4三seed各≥4.224M曝光，D0冻结权重重定阈值。\n最终矩阵由configs/M8c/final_*.yaml与minimum*.yaml驱动；三seed均值/10000组bootstrap CI与次要项进度见reports/M8c_REVIEW.md。\n独立{c['budget_gpu_hours']}GPUh硬限，含加载/失败/中断上界；主E3使用冻结test500哈希序前缀{budget['test_groups']}组，不按test选模型或λ。\n冻结data/M8c_v1/freeze.json；上游M71/M8a/M8b SHA不变。\n")
            with Path('docs/BLOCKERS.md').open('a') as f:
                f.write('\n## 2026-11-09 M8c 交付限制\n21GPUh预算内实际λ网格、最小扰动样本规模与E1/E2未执行范围均见reports/M8c_REVIEW.md及budget_lock.json，未把部分网格冒充全量。\n二分为有限PGD搜索，不是认证最小值；未成功/原已目标/截尾及独立报告检测另列，条件比值不能代表所有episode。\n依赖门槛仅验证群体依赖；R4在消费消息后检测，conformal clean阈值不保证自适应攻击下的检测率。\n')
            with Path('docs/CHANGELOG.md').open('a') as f:f.write('\n## 2026-11-09 M8c\n新增消息依赖direct-lookup、曝光匹配R4、新校准、锁定PGD/二分、三seed统计与审查证据；未修改上游冻结。\n')
            result=json.loads((root/'completion.json').read_text());result.update(freezer_run=str(run.path),freeze=str(root/'freeze.json'),status=status);write_json(root/'completion.json',result)
            paths=set()
            for folder in [root,Path('configs/M8c'),Path('reports/M8c_evidence')]:paths.update(p for p in folder.rglob('*') if p.is_file())
            paths.update(p for p in Path('reports').glob('M8c*') if p.is_file())
            paths.update(Path('clr').rglob('*m8c*.py'));paths.update(Path('scripts').glob('m8c*.py'));paths.update([Path('tests/test_m8c.py'),Path('docs/M8c_TASK.md'),Path('configs/M8c_gate.yaml')])
            for p in Path('runs').glob('*/config.json'):
                cfg=json.loads(p.read_text())
                if cfg.get('milestone')!='M8c' or p.parent.resolve()==run.path.resolve():continue
                met=json.loads((p.parent/'metrics.json').read_text());assert met['status'] in ['failed','completed'],p.parent
                paths.update(q for q in p.parent.rglob('*') if q.is_file())
            paths={p for p in paths if p.name not in ['freeze.json','verification.json','finalize.log']}
            ordered=sorted(paths)
            with ThreadPoolExecutor(max_workers=2) as pool:digests=list(pool.map(sha256,ordered))
            hashes={str(p.resolve()):h for p,h in zip(ordered,digests)}
            write_json(root/'freeze.json',{'version':'M8c_v1','status':status,'freezer_run':str(run.path),'gpu_hours_conservative':total_cost(),'sha256':hashes})
            with ThreadPoolExecutor(max_workers=2) as pool:current=list(pool.map(sha256,ordered))
            bad=[str(p) for p,h,now in zip(ordered,digests,current) if h!=now]
            write_json(root/'verification.json',{'run':str(run.path),'passed':not bad,'files':len(ordered),'mismatches':bad});assert not bad
            cost.update(status='completed');live(stage='frozen_budgeted_results_pending_review',milestone_complete=c['all_requested_settings_completed'],budgeted_plan_complete=True,background_tasks_active=False,test_opened=True,review=str(review),freeze=str(root/'freeze.json'),gpu_hours_conservative=total_cost())
            print(status,'files',len(ordered),'GPUh',total_cost(),flush=True)
        except BaseException:cost.update(status='failed');live(stage='finalize_failed',milestone_complete=False);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

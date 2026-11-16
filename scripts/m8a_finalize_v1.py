"""Wait for the locked suite, report, freeze and verify; no model decisions."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.audit.features import sha256
from clr.audit.m8a_runtime import snapshot,Cost,total_cost,live
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output'])
        try:
            while True:
                complete=[(root/f'full_part{i}.json').exists() for i in range(c['parts'])]
                live(stage='fixed_test_matrix_running',background_tasks_active=True,completed_parts=sum(complete),total_parts=c['parts'])
                if all(complete):break
                cost.check();time.sleep(30)
            subprocess.run([sys.executable,'scripts/m8a_report_v1.py',c['report_config']],check=True)
            result=json.loads((root/'analysis_v1/completion.json').read_text())
            review=Path('reports/M8a_REVIEW.md');text=review.read_text()
            assert len(review.read_bytes())<=8000 and all(f'### ({x})' in text for x in 'abcdef')
            assert sum(line.startswith('|:---') for line in text.splitlines())==3
            assert result['groups']==500 and result['rows']==62000 and total_cost()<=25
            # Preserve task-specific record length limits and avoid changing
            # any upstream frozen report/code/weights.
            status='M8a固定500组评估与统计完成待审查；P0通过，三seed/E0–E4齐全；唯一审查reports/M8a_REVIEW.md，冻结data/M8a_v1/freeze.json。'
            for filename in ['AGENTS.md','docs/STATUS.md']:
                p=Path(filename);s=p.read_text();start=s.index('<!-- M8_LIVE -->');end=s.index('<!-- /M8_LIVE -->',start)
                s=s[:start]+'<!-- M8_LIVE -->\n'+status+'\n'+s[end:];p.write_text(s)
            with Path('docs/EXPERIMENTS.md').open('a') as f:
                f.write('\n## 2026-11-08 M8a 固定测试完成\nP0公开期望通过，无需重训/重定阈值；E0–E4固定500组三seed，全部ε/λ/50/200预算，实际早停步数单列。命令configs/M8a/full_{0,1,2,3}.yaml→configs/M8a/report_v1.yaml；完整均值/95%组CI、负面结果及成本见reports/M8a_REVIEW.md和reports/M8a_details.md。M7/M6/M7.1冻结只读。\n')
            with Path('docs/BLOCKERS.md').open('a') as f:
                f.write('\n## 2026-11-08 M8a 完成限制\nA1训练负例未覆盖e+2，seed33置信度及校准门限异常有cal_B证据；E2只保证实际fp16 reader输入逐位一致，FP32舍入残差明示。文献为full-prefill适配，未复现原latent-thought拓扑/PCA/RePS。有限PGD/4面板EOT/早停不构成鲁棒保证；正文、逐组CI及运行证据见reports/M8a_REVIEW.md。\n')
            paths=set()
            for base in [root,Path('configs/M8a'),Path('reports/M8a_evidence')]:
                paths.update(p for p in base.rglob('*') if p.is_file())
            paths.update(p for p in Path('reports').glob('M8a*') if p.is_file())
            paths.update(Path('clr').rglob('*m8*.py'));paths.update(Path('scripts').glob('m8a*.py'))
            paths.update(Path('tests').glob('test_m8*.py'));paths.add(Path('docs/M8a_TASK.md'))
            for config in Path('runs').glob('*/config.json'):
                if config.parent.resolve()==run.path.resolve():continue
                cfg=json.loads(config.read_text())
                if cfg.get('milestone') in ['M8','M8a']:
                    m=json.loads((config.parent/'metrics.json').read_text());assert m['status'] in ['completed','failed']
                    paths.update(p for p in config.parent.rglob('*') if p.is_file())
            paths={p for p in paths if p.name not in ['freeze.json','verification.json','finalize_v1.log']}
            frozen={str(p.resolve()):sha256(p) for p in sorted(paths)}
            assert not (root/'freeze.json').exists()
            write_json(root/'freeze.json',{'version':'M8a_v1','status':'completed_pending_review','freezer_run':str(run.path),
                'artifact_bytes':sum(Path(p).stat().st_size for p in frozen),'gpu_hours':total_cost(),'sha256':frozen})
            def check(item):
                path,digest=item;return path if sha256(path)!=digest else None
            with ThreadPoolExecutor(max_workers=2) as executor:bad=[p for p in executor.map(check,frozen.items()) if p]
            write_json(root/'verification.json',{'passed':not bad,'files':len(frozen),'mismatches':bad,'run':str(run.path)})
            assert not bad,bad
            # Check upstream manifests after all downstream work as well.
            lock=json.loads((root/'final_lock.json').read_text())
            for p,digest in lock['sha256'].items():assert sha256(p)==digest,p
            cost.update(status='completed')
            live(stage='frozen_complete_pending_review',background_tasks_active=False,milestone_complete=True,
                freeze=str(root/'freeze.json'),verification=str(root/'verification.json'),gpu_hours=total_cost(),review=str(review),
                report_bytes=review.stat().st_size,finalizer_run=str(run.path))
            print('M8a completed and frozen',len(frozen),'files',flush=True)
        except Exception:
            cost.update(status='failed');live(stage='finalize_failed',run=str(run.path),milestone_complete=False);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

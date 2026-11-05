"""Diagnose first, persist conclusion, then candidate gates. Never opens test."""
import argparse,json,time,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
    root=Path(c['output']);dc=load_config(c['diagnostic_config']);start=time.monotonic()
    while not all(Path(p).exists() for p in dc['diagnostic_gates'].values()):
        if time.monotonic()-start>c['wait_seconds']:raise TimeoutError('Diagnostic prerequisite missing; inspect runs, no fallback')
        time.sleep(10)
    subprocess.run(['.venv/bin/python','scripts/m41_diagnostic_report.py',c['diagnostic_config']],check=True)
    Path('docs/STATUS.md').open('a').write('\nM4.1 A诊断完成，根因结论已写reports/M4.1_REVIEW.md(b)，完整证据data/M41_v1/diagnosis_complete.json。继续B两候选学习能力关卡；尚未进入C。\n')
    def worker(variant,configs):
        for cfg in configs:
            with Path(f'reports/M41_gate_{Path(cfg).stem}.log').open('w') as log:subprocess.run(['.venv/bin/python','scripts/m41_learnability.py',cfg],check=True,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(worker,v,configs) for v,configs in c['candidate_configs'].items()]
        for f in fs:f.result()
    gates={}
    for variant,configs in c['candidate_configs'].items():
        for path in configs:
            cfg=load_config(path);name=f"{variant}_{cfg['task']}_{cfg['qtype']}";gates[name]=json.loads(Path(cfg['output'],f"gate_{name}_{cfg['seed']}.json").read_text())
    passed=all(x['passed'] for x in gates.values());write_json(root/'gate_lock.json',{'both_variants_passed':passed,'gates':gates,'model_sha256':hashlib.sha256(Path('clr/models/clr_v1.py').read_bytes()).hexdigest(),'test_opened':False})
    Path('docs/STATUS.md').open('a').write(f'\nM4.1 B关卡完成，both_variants_passed={passed}；见data/M41_v1/gate_lock.json。C只有两者全部通过才可启动。\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
 try:main(load_config(a.config,a.overrides))
 except Exception as e:
    Path('docs/BLOCKERS.md').open('a').write(f'\n## M4.1 诊断接续错误\n{type(e).__name__}: {e}；不自动进入C，检查reports/M41_gate_controller.log。\n');raise

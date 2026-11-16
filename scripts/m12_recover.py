"""Wait for existing A4 workers, then reuse completed receipts and continue M12."""
import argparse,json,time,subprocess,tarfile
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.m12_runtime import digest

def live(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'

def main(c):
    with recorded({**c,'stage':'CPU_recovery_controller'}) as run:
        source=[Path(__file__),Path('scripts/m12_pipeline.py'),Path('configs/M12/recovery_pipeline.yaml')]
        write_json(run.path/'startup_code_sha256.json',{str(p):digest(p) for p in source})
        with tarfile.open(run.path/'startup_source.tar.gz','w:gz') as tar:
            for p in source:tar.add(p)
        recovery=json.loads(Path('data/M12_ablation/recovery.json').read_text())
        pids=[recovery['original_pipeline_pid'],recovery['retry_pid']]
        while any(live(pid) for pid in pids):
            write_json(run.path/'progress.json',{'waiting_pids':[pid for pid in pids if live(pid)],'test_opened':False})
            time.sleep(30)
        for seed in [11,22,33]:
            receipt=Path(f'data/M12_ablation/trained_A4_lite_{seed}.json')
            if not receipt.exists():raise RuntimeError(f'A4 seed{seed} lacks completed receipt; no silent fallback')
            r=Path(json.loads(receipt.read_text())['run'])
            if json.loads((r/'metrics.json').read_text())['status']!='completed':raise RuntimeError('A4 receipt is not completed')
        if Path('data/M12_ablation/test_access.json').exists():raise RuntimeError('Unexpected test access before recovery lock')
        with Path('reports/M12_evidence/controller_recovered.log').open('w') as log:
            p=subprocess.Popen(['.venv/bin/python','-m','scripts.m12_pipeline',c['recovery_pipeline_config']],stdout=log,stderr=subprocess.STDOUT)
            write_json('data/M12_ablation/recovered_controller.json',{'pid':p.pid,'config':c['recovery_pipeline_config'],'controller_run':str(run.path),'reused_training':'A1-A4 all completed receipts; no retraining'})
            code=p.wait()
            if code:raise RuntimeError(f'Recovered pipeline failed with exit code {code}')
        run.write_metrics(status='completed',gpu_hours=0,source_forward_calls=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args();main(load_config(args.config))

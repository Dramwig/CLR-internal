"""Run CPU paper closure after all authorized M9-A inference has completed."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import subprocess
import time
from pathlib import Path
import yaml


def main(config):
    root = Path(config['output'])
    while not (root / 'experiment_complete.json').exists():
        if not Path(f'/proc/{config["experiment_controller_pid"]}').exists():
            raise RuntimeError('Experiment controller ended before experiment_complete.json')
        time.sleep(20)
    for job in config['jobs']:
        log = Path(job['log'])
        if log.exists():
            raise FileExistsError(log)
        print('starting', job['module'], job['config'], flush=True)
        command = [job.get('python', sys.executable), '-u', '-m', job['module'], job['config']]
        if 'script' in job:
            command = [job['python'], '-I', job['script'], job['config']]
        with log.open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('completed', job['module'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    main(yaml.safe_load(Path(parser.parse_args().config).read_text()))

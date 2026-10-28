"""Audit previously collected public text evidence; never load model weights."""
import argparse
import hashlib
import json
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded


def main(config):
    with recorded(config) as run:
        if config['weights_allowed']:
            raise ValueError('This metadata audit cannot authorize weights')
        manifest = json.loads(Path(config['manifest']).read_text())
        for path, expected in manifest['files'].items():
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Evidence digest mismatch: {path}')
        (run.path / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
        (run.path / 'report.md').write_text(Path(config['report']).read_text())
        (run.path / 'entrypoint.py').write_text(Path(__file__).read_text())
        run.write_metrics(status='completed', evidence_files=len(manifest['files']),
                          kind='metadata_evidence_audit', gpu_hours=0,
                          source_forward_calls=0, weights_downloaded=False,
                          independent_accuracy_measured=False)
        print(run.path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    main(load_config(args.config, args.overrides))

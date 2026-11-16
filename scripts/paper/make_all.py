"""One command rebuilds every paper artifact using its own script and YAML."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml


def main(c):
    with tempfile.TemporaryDirectory(prefix='clr-paper-configs-') as folder:
        for item in c['artifacts']:
            config = yaml.safe_load(Path(item['config']).read_text())
            config['output_root'] = c['output_root']
            p = Path(folder) / Path(item['config']).name
            p.write_text(yaml.safe_dump(config, sort_keys=False))
            subprocess.run([sys.executable, *(['-I'] if c.get('isolated') else []), item['script'], str(p)], check=True)
            print(config['artifact'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    main(yaml.safe_load(Path(p.parse_args().config).read_text()))

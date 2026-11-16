"""CPU audit of frozen problem prefixes and original tokenizer cohort widths."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.m6_runtime import snapshot, live
from clr.utils.records import recorded


def audit_contexts(parent, plan, original, encode):
    # Original official cohorts retain members removed from the final splits.
    cohorts = defaultdict(dict)
    for rows in parent['splits'].values():
        for row in rows:
            cohorts[row['cohort_id']][row['id']] = row['question']
    for row in original:
        cohorts[f'M0_official_test:{row["index"] // 128}'][row['problem_id']] = row['question']
    widths = {key: max(len(encode(q.strip().replace('  ', ' '))) for q in rows.values())
              for key, rows in cohorts.items()}
    counts = {}
    for split, rows in plan['splits'].items():
        full = parent['splits'][split]
        if [r['id'] for r in rows] != [r['id'] for r in full[:len(rows)]]:
            raise ValueError('M6 problem IDs differ from frozen prefix: ' + split)
        for row, old in zip(rows, full):
            key = row['cohort_id']
            if key != old['cohort_id'] or row['question'] != old['question']:
                raise ValueError('Original input/cohort changed: ' + row['id'])
            width = widths[key]
            length = len(encode(row['question'].strip().replace('  ', ' ')))
            if row['pad_to'] != width or row['left_padding'] != width - length:
                raise ValueError('Original cohort tokenizer width differs: ' + row['id'])
        counts[split] = len(rows)
    return {'problem_counts': counts, 'original_cohorts': len(cohorts),
            'mismatches': 0, 'prediction_fields_used': False,
            'scope': 'post-start input metadata audit; no source calls, predictions or protocol changes'}


def main(c):
    with recorded(c) as run:
        print(run.path, flush=True)
        start = time.monotonic()
        snapshot(run, c)
        protocol = load_config(c['protocol'])
        tokenizer = AutoTokenizer.from_pretrained(protocol['source']['base_path'], local_files_only=True)
        paths = {name: Path(c[name]) for name in ['parent_plan', 'plan', 'original_contexts']}
        values = {name: json.loads(path.read_text()) for name, path in paths.items()}
        result = audit_contexts(values['parent_plan'], values['plan'], values['original_contexts'], tokenizer.encode)
        result['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
        result['tokenizer_path'] = protocol['source']['base_path']
        write_json(run.path / 'audit.json', result)
        live(context_metadata_audit=str(run.path), context_width_mismatches=0)
        run.write_metrics(status='completed', gpu_hours=0, source_forward_calls=0,
                          wall_seconds=time.monotonic() - start, summary=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    main(load_config(a.config, a.overrides))

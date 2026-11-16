"""Predeclared lambda sensitivity; select candidates using attack-side scores only."""
import sys
sys.dont_write_bytecode = True
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.audit.m8b_statistics import select_candidate, group_ci
from clr.audit.m8b_runtime import snapshot, Cost
from clr.utils.config import load_config
from clr.utils.idea_a import write_json
from clr.utils.records import recorded


def summarize(frame, keys, metrics):
    rows = []
    for key, part in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row['groups'] = part.group_id.nunique()
        row['seeds'] = part.seed.nunique()
        for metric in metrics:
            m, lo, hi = group_ci(part.groupby('group_id')[metric].mean())
            row.update({metric: m, metric+'_lo': lo, metric+'_hi': hi})
        rows.append(row)
    return pd.DataFrame(rows)


def main(c):
    with recorded(c) as run:
        snapshot(run, c); cost = Cost(run, c, gpu=False)
        root = Path(c['output']); out = root/c['sensitivity_name']
        out.mkdir(exist_ok=False)
        grouped = defaultdict(list); clean = {}; references = []
        for split, specs in c['sensitivity_sources'].items():
            for spec in specs:
                refs = sorted(root.glob(spec['pattern']))
                assert refs, spec
                for p in refs:
                    ref = json.loads(p.read_text()); references.append(str(p))
                    with (Path(ref['run'])/'results.jsonl').open() as handle:
                        for line in handle:
                            r = json.loads(line)
                            if r['kind'] != 'E3' or r['epsilon'] != c['sensitivity_epsilon']:
                                continue
                            if spec.get('clean_reference', False):
                                clean[split, r['group_id']] = r['clean_audit']
                            key = (split, r['group_id'], r['attack_design'], r['attack_seed'], r['step_budget'])
                            grouped[key].append(r)
        allowed = {s:set(json.loads(Path(p).read_text())['group_ids'])
                   for s,p in c.get('sensitivity_group_ids',{}).items()}
        rows = []
        for (split, gid, design, seed, budget), rs in grouped.items():
            if split in allowed and gid not in allowed[split]: continue
            assert sorted(r['lambda'] for r in rs) == sorted(c['sensitivity_lambdas'])
            cases = [(str(r['lambda']), r) for r in rs]
            cases.append(('combined_selection', select_candidate(rs)))
            for label, r in cases:
                for k in ([1,4,16,64] if design == 'R1' else [16]):
                    audit = r['audit'][f'{design}:{seed}:{k}']
                    original = clean[split, gid][f'{design}:{seed}:{k}']
                    for alpha in [.01, .05]:
                        flag = audit[str(alpha)]
                        rows.append({'split':split, 'group_id':gid, 'design':design,
                                     'seed':seed, 'steps':budget, 'k':k, 'alpha':alpha,
                                     'candidate_set':label, 'selected_lambda':r['lambda'],
                                     'failure':r['changed_to_target'] and not flag,
                                     'detected':flag, 'target_achieved':r['target_achieved'],
                                     'clean_FPR':original[str(alpha)]})
        frame = pd.DataFrame(rows)
        for split, count in c['expected_split_groups'].items():
            assert frame[frame.split == split].group_id.nunique() == count
            if split in allowed:
                assert set(frame[frame.split == split].group_id) == allowed[split]
        frame.to_csv(out/'lambda_group_metrics.csv.gz', index=False)
        keys = ['split','design','steps','k','alpha','candidate_set']
        summarize(frame, keys, ['failure','detected','target_achieved','clean_FPR']).to_csv(out/'lambda_summary.csv',index=False)
        # Step curves use the locked main lambda=1 protocol, with actual steps separately reported.
        main_frame = pd.read_csv(root/c['analysis_name']/'group_metrics.csv.gz')
        for alpha in [.01,.05]:
            fig, ax = plt.subplots(figsize=(6,4))
            for design in c['designs']:
                p = main_frame[(main_frame.split=='test') & (main_frame.kind=='E3') &
                               (main_frame.design==design) & (main_frame.k==16) &
                               (main_frame.alpha==alpha) & (main_frame.epsilon==c['sensitivity_epsilon'])]
                points = p.groupby('steps').failure.mean()
                if len(points): ax.plot(points.index,points.values,'o-',label=design)
            ax.set(xlabel='PGD step budget (early success allowed)',ylabel='Robust failure rate',ylim=(0,1))
            if ax.lines: ax.legend()
            fig.tight_layout(); fig.savefig(out/f'steps_failure_{alpha}.png',dpi=180)
            fig.savefig(out/f'steps_failure_{alpha}.pdf'); plt.close(fig)
        write_json(out/'completion.json',{'run':str(run.path),'references':references,
                   'groups':{s:int(frame[frame.split==s].group_id.nunique()) for s in c['sensitivity_sources']},
                   'selection':'target achieved, then lowest attack-side selection NLL, then target logp; report outcomes excluded',
                   'main_table':'lambda=1 fixed; combined selection is supplementary only'})
        cost.update(status='completed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('config'); a = p.parse_args()
    main(load_config(a.config))

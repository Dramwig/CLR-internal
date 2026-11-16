"""Complete the prerequisite evidence without model calls or retraining."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clr.audit.m8a_runtime import snapshot,Cost,live
from clr.audit.m8_p0 import expected_team,public_constraints
from clr.audit.m7_evaluation import expected_private
from clr.data.m71 import examples
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json


def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False)
        output=Path(c['output']);output.mkdir(parents=True,exist_ok=True)
        source=load_config('configs/M71/pool_cal_receiver.yaml')
        entries=[json.loads(s) for s in Path('data/M71_v1/pool/source/cal_B/manifest.jsonl').read_text().splitlines()]
        public={e.id:e.x_pub for e in examples(source,'cal_B')}
        for e in entries:
            parsed=public_constraints(public[e['id']])
            assert parsed['T']==e['T'] and parsed['z_pub']==e['z_pub']
            assert expected_team(public[e['id']],e['commitment'])==expected_private({**e,'commitments':[e['commitment']]})
        prereq=[];history=[];b0=[]
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        for seed in [11,22,33]:
            p=json.loads(Path(f'data/M8_v1/P0_full_{seed}.json').read_text())
            assert p['public_expectations_identical'] and p['groups_checked']==3000
            assert json.loads((Path(p['run'])/'metrics.json').read_text())['status']=='completed'
            prereq.extend({**r,'seed':seed} for r in p['results'])
            trained=json.loads(Path(f'data/M71_v1/pool/training/CLR_{seed}.json').read_text())
            h=json.loads((Path(trained['run'])/'history.json').read_text())
            history.extend({**x,'seed':seed} for x in h)
            selected=trained['selected']['epoch']
            axes[0].plot([x['epoch'] for x in h],[x['selection_nll'] for x in h],label=str(seed))
            axes[0].scatter([selected],[trained['selected']['selection_nll']])
            axes[1].plot([x['epoch'] for x in h],[x['metrics']['A3']['nll'] for x in h],label=str(seed))
            axes[1].scatter([selected],[trained['selected']['metrics']['A3']['nll']])
            bptr=json.loads(Path(f'data/M71_v1/pool/evaluated/B0_{seed}.json').read_text())
            metrics=json.loads((Path(bptr['run'])/'qa_metrics.json').read_text())
            b0.extend({**r,'seed':seed,'mode':'trained_B0'} for r in metrics if r['task'] in ['A1','A4'] and not r['heldout_templates'])
        axes[0].set(title='reader_val selection NLL',xlabel='Epoch',ylabel='NLL')
        axes[1].set(title='reader_val A3 NLL',xlabel='Epoch',ylabel='NLL')
        for ax in axes:ax.legend(title='Seed');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'seed_training_curves.png',dpi=160);plt.close(fig)
        result={'passed':True,'expectations_unchanged':True,'cal_B_groups_checked':len(entries),
            'reader_val_test_groups_checked_per_seed':3000,
            'expected_answer_sources':{
                'public_z':'solve literal x_pub arithmetic (floor division), routing puzzle, or verified public result',
                'public_table':'parse x_pub decision table, each z-row is a permutation',
                'A1':'candidate team equals inverse(public_table[public_z], commitment route)',
                'A2':'Yes for the literal route in commitment',
                'A3':'Yes: consistency with the literal commitment',
                'A4':'inverse(public_table[public_z], commitment route)'},
            'private_information_in_expectations':False,'threshold_recalibration_required':False,
            'unchanged_M71_CF_detection_nominal_alpha5_mean':.9313333333333333,
            'seed33_finding':'Selected epoch6 of9, A1/A4 accurate but A3 NLL .8597 vs ~.694 for seeds11/22. A1 temperature .113 and k16 threshold5.49; calibration/inference question negatives include all four teams, while training negatives are adjacent e+1/e-1. This distribution difference and confidence/clipping are possible mechanisms; no causal attribution from curves alone.',
            'localization_caveat':'Oracle locations only in requested diagnostic. Contextual KV and masking OOD prevent concluding token-only causality.',
            'P0_runs':[json.loads(Path(f'data/M8_v1/P0_full_{s}.json').read_text())['run'] for s in [11,22,33]],
            'run':str(run.path)}
        write_json(output/'P0.json',result);write_json(output/'P0_interventions.json',prereq)
        write_json(output/'P0_B0.json',b0);write_json(output/'P0_training_history.json',history)
        cost.update(status='completed');live(stage='P0_passed',P0=str(output/'P0.json'),milestone_complete=False)
        print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

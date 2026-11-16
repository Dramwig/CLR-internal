"""Predeclared whole-variant selection using three-seed val NLL only."""
import sys
sys.dont_write_bytecode=True
import argparse,json,shutil
from pathlib import Path
from clr.audit.m8b_runtime import Cost,snapshot,live
from clr.audit.features import sha256
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.utils.config import load_config

def main(c):
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c,gpu=False);root=Path(c['output']);variants={};picked={}
        for design in ['R2','R3']:
            variants[design]={}
            for folder in ['training','warm_training_v2']:
                refs=[json.loads((root/folder/f'{design}_{seed}.json').read_text()) for seed in c['reader_seeds']]
                variants[design][folder]={'mean_val_nll':sum(r['best_val_nll'] for r in refs)/len(refs),'refs':refs}
            picked[design]='warm_training_v2' if variants[design]['warm_training_v2']['mean_val_nll']<variants[design]['training']['mean_val_nll'] else 'training'
        train=root/'selected_training';cal=root/'selected_calibration';train.mkdir(exist_ok=False);cal.mkdir(exist_ok=False)
        for d in ['R1','R2','R3','R4']:
            for s in c['reader_seeds']:
                if d!='R1':
                    src=root/(picked[d] if d in picked else 'training')/f'{d}_{s}.json';ref=json.loads(src.read_text())
                    assert ref['body_frozen'] and ref['train_groups']==4000 and ref['val_groups']==200
                    write_json(train/f'{d}_{s}.json',{**ref,'selected_from':str(src),'selection_rule':'whole R2/R3 variant by mean val NLL; no test'})
                if d!='R1' and (d not in picked or picked[d]=='training'):
                    src=root/'calibration'/f'{d}_{s}.json';ref=json.loads(src.read_text());write_json(cal/f'{d}_{s}.json',{**ref,'reused_from':str(src),'reused_sha256':sha256(src)})
        write_json(root/'variant_selection.json',{'run':str(run.path),'selected_variants':picked,'variants':variants,'rule':'choose v2 iff 3-seed mean best val NLL is lower; preserve all seeds','test_opened':False})
        cost.update(status='completed');live(stage='variants_selected',selected_variants=picked)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

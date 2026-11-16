"""Per-design clean calibration with receiver-visible expected answers only."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m8b_reader import SourceBank,new_model,public_questions,rng_seed
from clr.audit.m71_reader import Serializer
from clr.audit.m71_storage import load_bundle
from clr.audit.features import FeatureSequence
from clr.audit.m8b_features import as_source
from clr.audit.m7_evaluation import fit_temperature
from clr.data.m7 import threshold
from clr.audit.m8b_runtime import Cost,snapshot,live
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.records import recorded
from clr.utils.config import load_config

@torch.no_grad()
def logits(model,serializer,rows,seq,device):
    source=as_source(seq);source={k:v.to(device) for k,v in source.items()};out={}
    for task in sorted({r['task'] for r in rows}):
        ii=[i for i,r in enumerate(rows) if r['task']==task];rr=[rows[i] for i in ii]
        with torch.autocast('cuda',dtype=torch.bfloat16):ls=model(serializer.batch(rr,device),{k:v.expand(len(ii),*v.shape[1:]) for k,v in source.items()},task)
        out.update({i:x for i,x in zip(ii,ls.float().cpu())})
    return [out[i] for i in range(len(rows))]

def old_sequences(split):
    entries=[json.loads(l) for l in Path(f'data/M71_v1/pool/source/{split}/manifest.jsonl').open()]
    return {e['id']:FeatureSequence(**load_bundle(e['file'])['messages'][0]) for e in entries}

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output']);serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True));folder=root/'source'
            for design in c['designs']:
                train=SourceBank(folder/'cal_A',design) if design!='R1' else None
                bank=SourceBank(folder/'cal_B',design) if design!='R1' else None
                r1z=old_sequences('cal_B') if design=='R1' else None
                entries=bank.entries if bank else [json.loads(l) for l in (folder/'cal_B'/'manifest.jsonl').open()]
                for seed in c['reader_seeds']:
                    if c.get('skip_completed',False) and (root/c.get('calibration_folder','calibration')/f'{design}_{seed}.json').exists():continue
                    cost.check();ref=json.loads((root/c.get('training_folder','training')/f'{design}_{seed}.json').read_text()) if design!='R1' else None
                    model=new_model(seed,design,c['device'],ref['checkpoint'] if ref else None).eval().requires_grad_(False)
                    if design=='R1':temp=json.loads(Path(f'data/M71_v1/pool/calibration/CLR_{seed}.json').read_text())['temperatures']
                    else:
                        if design=='R2':train.refresh(rng_seed(seed,'cal_A'),c['device'] if c.get('randomization_device')=='gpu' else None)
                        rows=train.rows(0,True);ls={t:[] for t in ['A1','A4']};ys={t:[] for t in ls}
                        # Each message contributes one balanced yes/no pair and one choice.
                        for mid in train.z:
                            rr=[r for r in rows if r['message_id']==mid];ll=logits(model,serializer,rr,train.z[mid],c['device'])
                            for r,l in zip(rr,ll):ls[r['task']].append(l.numpy());ys[r['task']].append(r['label'])
                        temp={t:fit_temperature(np.stack(ls[t]),ys[t]) for t in ls}
                    scores={str(k):[] for k in ([1,4,16,64] if design=='R1' else [16])}
                    if design=='R2':bank.refresh(rng_seed(seed,'cal_B'),c['device'] if c.get('randomization_device')=='gpu' else None)
                    for i,e in enumerate(entries):
                        cost.check();seq=r1z[e['id']] if design=='R1' else bank.z[e['id']]
                        rows=public_questions(e,rng_seed(seed,'report',e['id']),max(map(int,scores)),design=='R1' and not c.get('matched_questions',False))
                        ll=logits(model,serializer,rows,seq,c['device']);nll=[min(float(-torch.log_softmax(l/temp[r['task']],-1)[r['label']]),27.6310211159) for r,l in zip(rows,ll)]
                        for k in scores:scores[k].append(float(np.mean(nll[:int(k)])))
                        if i%100==0:cost.update(status='calibrating',design=design,seed=seed,groups_completed=i+1)
                    thresholds={k:{str(a):threshold(v,a) for a in [.01,.05]} for k,v in scores.items()}
                    dest=root/c.get('calibration_folder','calibration');dest.mkdir(exist_ok=True)
                    write_json(dest/f'{design}_{seed}.json',{'temperatures':temp,'thresholds':thresholds,'scores':scores,'run':str(run.path),'cal_A_groups':300,'cal_B_groups':len(entries),'source_random_seed_rule':'sha256(seed,cal_A/cal_B,mid)','public_expectations_only':True,'randomization_device':c.get('randomization_device','cpu')})
                    print(design,seed,thresholds,'seconds',round(time.monotonic()-cost.start,1),flush=True)
                    del model
                del train,bank,r1z
            cost.update(status='completed');live(stage='calibration_completed',run=str(run.path))
        except Exception:
            cost.update(status='failed');live(stage='calibration_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

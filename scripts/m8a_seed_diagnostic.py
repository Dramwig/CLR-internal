"""Cal_B-only diagnosis of the frozen seed33 confidence/negative-pool shift."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m71_reader import Bank,Serializer
from clr.audit.m71_evaluation import predict
from clr.audit.m8a_runtime import snapshot,Cost
from clr.data.m7 import TEAMS,TEMPLATES
from clr.models.m71 import Reader
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything

def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            bank=Bank('data/M71_v1','cal_B');rows=[];types=[]
            for entry in bank.entries:
                for candidate in range(4):
                    team=entry['e'][0]
                    rows.append({'episode_id':entry['id'],'group_id':entry['group_id'],'message_id':entry['id'],'task':'A1',
                        'question_text':TEMPLATES['A1'][0].format(fact=TEAMS[candidate]),'options':['No','Yes'],'label':int(candidate==team)})
                    types.append((candidate-team)%4)
            serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True));out=[]
            for seed in [11,22,33]:
                train=json.loads(Path(f'data/M71_v1/pool/training/CLR_{seed}.json').read_text())
                cal=json.loads(Path(f'data/M71_v1/pool/calibration/CLR_{seed}.json').read_text())
                model=Reader('CLR').to(c['device']);model.load_state_dict(torch.load(train['checkpoint'],weights_only=True,map_location='cpu'));model.eval()
                raw=np.asarray(predict(model,bank,rows,serializer,c));logits=torch.tensor(raw/cal['temperatures']['A1'])
                logp=logits.log_softmax(1).numpy();yy=np.array([r['label'] for r in rows]);nll=-logp[np.arange(len(yy)),yy]
                for gap in range(4):
                    use=np.asarray(types)==gap
                    out.append({'seed':seed,'gap_mod4':gap,'n':int(use.sum()),'accuracy':float((raw.argmax(1)[use]==yy[use]).mean()),
                        'NLL':float(nll[use].mean()),'clipped_rate':float((nll[use]>=27.631021).mean())})
                del model;cost.update();cost.check()
            write_json(Path(c['output'])/'seed_confidence_diagnostic.json',{'run':str(run.path),'split':'cal_B','results':out,'model_or_threshold_changes':False})
            cost.update(status='completed');print(json.dumps(out),flush=True)
        except Exception:cost.update(status='failed');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

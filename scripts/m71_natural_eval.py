"""Post-lock natural A3 distributions, excluded from selection/calibration."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer
from clr.audit.m71_reader import Bank,Serializer
from clr.audit.m71_runtime import Cost,snapshot,assert_final_lock
from clr.audit.m71_evaluation import predict
from clr.data.m7 import TEMPLATES
from clr.models.m71 import Reader
from clr.utils.config import load_config
from clr.utils.idea_a import seed_everything,write_json
from clr.utils.records import recorded


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);root=Path(c['output']);rep=c['representation'];opened=False
        try:
            assert_final_lock(c)
            training=json.loads((root/rep/'training'/f'CLR_{c["seed"]}.json').read_text())
            calibration=json.loads((root/rep/'calibration'/f'CLR_{c["seed"]}.json').read_text())
            model=Reader('CLR').to(c['device']);model.load_state_dict(torch.load(training['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            output=[]
            for split in ['reader_val','test']:
                opened|=split=='test'
                if split=='test':cost.update(test_opened=True)
                bank=Bank(root,split,representation=rep)
                folder=root/rep/'source'/split
                records=[json.loads(line) for line in (folder/'natural_commitment.jsonl').open()]
                rows=[];metadata=[]
                for record in records:
                    for mode in ['direct','cot512']:
                        text=record[mode]['commitment']
                        for tid in range(6):
                            rows.append({'message_id':record['id'],'task':'A3','question_text':TEMPLATES['A3'][tid].format(commitment=text),
                                         'options':['No','Yes'],'label':0})
                            metadata.append({'split':split,'id':record['id'],'group_id':record['group_id'],
                                             'seed':c['seed'],'mode':mode,'model_correct':record[mode]['correct'],'template_id':tid})
                raw=predict(model,bank,rows,serializer,c)
                for m,x in zip(metadata,raw):output.append({**m,'P_consistent':float(softmax(x/calibration['temperatures']['A3'])[1])})
                cost.update(test_opened=split=='test')
            with (run.path/'natural_probabilities.jsonl').open('x') as f:
                for row in output:f.write(json.dumps(row)+'\n')
            destination=root/rep/'natural';destination.mkdir(exist_ok=True)
            write_json(destination/f'CLR_{c["seed"]}.json',{'run':str(run.path),'diagnostic_only':True,'test_opened':True})
            cost.update(status='completed',test_opened=True,source_forward_calls=0)
        except Exception:
            cost.update(status='failed',test_opened=opened);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args()
    main(load_config(a.config,a.overrides))

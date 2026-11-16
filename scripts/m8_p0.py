"""M8 prerequisite diagnostics on frozen readers; YAML only, no attacks."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from clr.audit.m71_reader import Bank,Serializer
from clr.audit.m71_evaluation import predict,consistency_rows,views
from clr.audit.m7_evaluation import expected_private
from clr.audit.m71_storage import save_bundle
from clr.audit.m8_p0 import expected_team,public_constraints,donor_map,value_records,record_intervention,summarize_groups
from clr.audit.m8_runtime import snapshot,Cost,live
from clr.data.m71 import examples
from clr.data.m7 import counterfactual
from clr.models.m71 import Reader
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything

class InterventionBank:
    def __init__(self,bank,mode,locations):
        self.bank,self.mode,self.locations=bank,mode,locations
        self.donors=donor_map(bank.entries)
    def batch(self,rows,device):
        if self.mode=='shuffle':return self.bank.batch([{**r,'message_id':self.donors[r['message_id']]} for r in rows],device)
        if self.mode=='zero':
            b=self.bank.batch(rows,device);b['values'].zero_();return b
        if self.mode=='full':return self.bank.batch(rows,device)
        # Use a shallow bank to keep original arrays immutable.
        local=object.__new__(Bank);local.z={}
        kind,operation=self.mode.split('_')
        for r in rows:
            mid=r['message_id']
            if mid not in local.z:local.z[mid]=record_intervention(self.bank.z[mid],self.locations[mid][kind],operation)
        return Bank.batch(local,rows,device)

def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);live(stage='P0_'+c['mode'],run=str(run.path),milestone_complete=False)
        try:
            root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
            source_config=load_config('configs/M71/pool_cal_receiver.yaml')
            serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            tr=json.loads(Path(f'data/M71_v1/pool/training/CLR_{c["seed"]}.json').read_text())
            model=Reader('CLR').to(c['device']);model.load_state_dict(torch.load(tr['checkpoint'],map_location='cpu',weights_only=True));model.eval()
            results=[];expectations=[]
            for split in c['splits']:
                cost.test_opened|=split=='test'
                bank=Bank('data/M71_v1',split,limit=c.get('groups'))
                ee=list(examples(source_config,split,c.get('groups')))
                lookup={e.id:e for e in ee};locations={}
                for entry in bank.entries:
                    e=lookup[entry['id']];pub=public_constraints(e.x_pub)
                    # Public text is the only input to the alternative expectation.
                    inferred=expected_team(e.x_pub,entry['commitment'])
                    old=expected_private({**entry,'commitments':[entry['commitment']]})
                    if pub!={'T':entry['T'],'z_pub':entry['z_pub']} or inferred!=old:
                        raise ValueError('Frozen expectations differ from receiver-visible public text')
                    expectations.append({'split':split,'group_id':entry['group_id'],'expected_e':inferred,'identical':True})
                    for ex in [e,counterfactual(e)]:
                        locations[ex.id]={'value':value_records(ex),'critical':[x+1 for x in ex.meta['critical_record_indices']]}
                        if not locations[ex.id]['value']:raise ValueError('No literal value records')
                rows=[r for r in bank.rows if r['task'] in ['A1','A4']]
                for mode in ['full','zero','shuffle','value_only','value_remove','critical_only','critical_remove']:
                    use=rows if mode in ['full','zero','shuffle'] else [r for r in rows if r['task']=='A4']
                    raw=predict(model,InterventionBank(bank,mode,locations),use,serializer,c)
                    grouped=summarize_groups(use,raw)
                    save_bundle({'groups':grouped},run.path/f'{split}_{mode}.pt.gz')
                    for task in ['A1','A4']:
                        selected=[g for g in grouped if g['task']==task]
                        if not selected:continue
                        for category in ['clean','counterfactual','both']:
                            values=[g['accuracy'] for g in selected if category=='both' or g['category']==category]
                            results.append({'split':split,'mode':mode,'task':task,'category':category,'accuracy':float(np.mean(values)),'groups':len(bank.entries)})
                    print(split,mode,'completed',len(use),flush=True);cost.update();cost.check()
                del bank
            write_json(run.path/'expectations.json',expectations);write_json(run.path/'results.json',results)
            target=root/f'P0_{c["mode"]}_{c["seed"]}.json'
            write_json(target,{'run':str(run.path),'results':results,'public_expectations_identical':True,'groups_checked':len(expectations),'test_opened':cost.test_opened})
            cost.update(status='completed');live(stage='P0_'+c['mode']+'_completed',run=str(run.path))
        except Exception:
            cost.update(status='failed');live(stage='P0_failed',run=str(run.path));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

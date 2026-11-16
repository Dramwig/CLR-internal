"""Fixed 4-training-group overfit diagnostic; does not select main settings/checkpoints."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from clr.models.m7 import Reader
from clr.audit.m7_reader import Bank,Serializer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.audit.m7_runtime import snapshot,Cost

def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c)
        try:
            bank=Bank(c['training_fixture'],'reader_train');serializer=Serializer(AutoTokenizer.from_pretrained(c['tokenizer_path'],local_files_only=True))
            for kind in ['CLR','B3']:
                model=Reader(kind).to(c['device']);m,s=bank.normalization();model.mean.copy_(m);model.scale.copy_(s)
                opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=0.)
                rng=np.random.default_rng(c['seed']);history=[]
                for epoch in range(1,c['epochs']+1):
                    model.train()
                    for task in ['A1','A2','A3','A4']:
                        rs=[r for r in bank.rows if r['task']==task];order=rng.permutation(len(rs))
                        for start in range(0,len(rs),c['batch_size']):
                            cost.check();batch=[rs[i] for i in order[start:start+c['batch_size']]]
                            z,mask=bank.batch(batch,c['device']);text=serializer.batch(batch,c['device'])
                            y=torch.tensor([r['label'] for r in batch],device=c['device']);opt.zero_grad(set_to_none=True)
                            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(text,z,mask,task);loss=torch.nn.functional.cross_entropy(logits.float(),y)
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                    if epoch%10==0:
                        model.eval();acc={}
                        with torch.no_grad():
                            for task in ['A1','A2','A3','A4']:
                                rs=[r for r in bank.rows if r['task']==task];correct=0
                                for start in range(0,len(rs),c['batch_size']):
                                    batch=rs[start:start+c['batch_size']];z,mask=bank.batch(batch,c['device'])
                                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(serializer.batch(batch,c['device']),z,mask,task)
                                    y=torch.tensor([r['label'] for r in batch],device=c['device']);correct+=int((logits.argmax(1)==y).sum())
                                acc[task]=correct/len(rs)
                        history.append({'epoch':epoch,'train_accuracy':acc});print(kind,epoch,acc,flush=True)
                        write_json(run.path/(kind+'_history.json'),history);cost.update()
                        if min(acc.values())>=c['threshold']:break
                write_json(run.path/(kind+'_result.json'),{'passed':min(acc.values())>=c['threshold'],'train_accuracy':acc,'epochs':epoch,'groups':4})
                del model,opt
            cost.update(status='completed',source_forward_calls=0,test_opened=False)
            write_json(Path(c['output'])/'learnability.json',{'run':str(run.path)})
        except Exception:
            cost.update(status='failed');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

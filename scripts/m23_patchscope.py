"""B5a input-embedding injection into frozen CODI-GPT2, never original problem text."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.sources.codi_v2 import load
from clr.data.questions.loader import StateStore
from clr.data.questions.core import seed
from clr.calib.m23 import temperature,metrics

def prompt(row,templates):
    text=templates[row['qtype']][row['template_id']]
    if row['qtype']=='Q4':text=text.format(v=row['options'][0])
    if row['qtype']=='Q3':text+='\n'+'\n'.join(f'{chr(65+i)}: {v}' for i,v in enumerate(row['options']))+'\nReply with A, B, C, or D.'
    else:text+='\nReply with Yes if correct or true, otherwise No.'
    return text+'\nAnswer:'

@torch.inference_mode()
def read(source,store,rows,templates,batch_size=8):
    outputs=[];tok=source.tokenizer;emb=source.model.get_embd(source.model.codi,source.model.model_name)
    verbalizers=[' A',' B',' C',' D'] if rows[0]['qtype']=='Q3' else [' No',' Yes']
    ids=[tok.encode(s,add_special_tokens=False) for s in verbalizers]
    if any(len(i)!=1 for i in ids):raise ValueError('Verbalizer must be a single token')
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];tokens=[tok.encode(prompt(r,templates),add_special_tokens=False) for r in batch];width=max(map(len,tokens))+1
        inputs=torch.full((len(batch),width),source.model.pad_token_id,dtype=torch.long,device=source.device);mask=torch.zeros_like(inputs)
        for i,(r,seq) in enumerate(zip(batch,tokens)):
            pad=width-len(seq)-1;inputs[i,pad]=tok.eos_token_id;inputs[i,pad+1:]=torch.tensor(seq,device=source.device);mask[i,pad:]=1
        e=emb(inputs)
        for i,(r,seq) in enumerate(zip(batch,tokens)):
            ref=r['state_ref'];e[i,width-len(seq)-1]=store.state(r['problem_id'],ref['traj_id'],ref['t']).to(source.device)
        pos=(mask.cumsum(1)-1).clamp(min=0)
        logits=source._forward(inputs_embeds=e,attention_mask=mask,position_ids=pos,use_cache=False).logits[:,-1,[i[0] for i in ids]]
        outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs)

def main(c):
 with recorded(c) as run:
    start=time.monotonic();torch.set_num_threads(2)
    if c['mode']=='full' and json.loads(Path(c['smoke_run'],'metrics.json').read_text())['status']!='completed':raise ValueError('B5a smoke required')
    source=load({**load_config(c['protocol'])['source'],'device':c['source_device'],'min_free_gib':c['source_min_free_gib']})
    templates=load_config(c['templates']);root=Path(c['output'])/c['mode'];stats=[]
    for qtype in ['Q1','Q2','Q3','Q4']:
        cal=pq.read_table(root/'cal.parquet',filters=[('qtype','=',qtype),('cal_part','=','cal_A'),('is_heldout_template','=',False)]).to_pylist()
        cal_logits=read(source,StateStore(c['root'],'cal'),cal,templates)
        def target(rows):
            if qtype=='Q3':return np.eye(4)[[r['label'] for r in rows]]
            y=np.array([r['label_soft'] for r in rows]);return np.stack([1-y,y],1)
        temp=temperature(cal_logits,target(cal))
        for split in ['reader_val','test']:
            store=StateStore(c['root'],split)
            rows=pq.read_table(root/f'{split}.parquet',filters=[('qtype','=',qtype)]).to_pylist()
            # identical inputs share a forward result, labels remain independent.
            keys={};unique=[];mapping=[]
            for r in rows:
                ref=r['state_ref'];key=(r['problem_id'],ref['traj_id'],ref['t'],r['template_id'],tuple(r['options']))
                if key not in keys:keys[key]=len(unique);unique.append(r)
                mapping.append(keys[key])
            logits=read(source,store,unique,templates)[mapping];y=target(rows)
            for held in [False,True]:
                pred=[]
                for i,r in enumerate(rows):
                    if r['is_heldout_template']==held:pred.append({'problem_id':r['problem_id'],'t':r['state_ref']['t'],'kind':r['state_ref']['kind'],'template_id':r['template_id'],'label':r['label'],'label_soft':r['label_soft'],'logits':logits[i].tolist(),'temperature':temp})
                pq.write_table(pa.Table.from_pylist(pred),run.path/f'B5a_{qtype}_0_{split}_{held}.parquet',compression='zstd')
                for t in range(7):
                    for kind in ['clean','traj']:
                        ii=[i for i,r in enumerate(rows) if r['is_heldout_template']==held and r['state_ref']['t']==t and r['state_ref']['kind']==kind]
                        if ii:stats.append({'baseline':'B5a','qtype':qtype,'seed':0,'split':split,'t':t,'kind':kind,'heldout':held,'n':len(ii),'temperature':temp,'raw':metrics(logits[ii],y[ii],qtype=='Q3'),'calibrated':metrics(logits[ii]/temp,y[ii],qtype=='Q3')})
            write_json(run.path/'statistics.json',stats);run.write_metrics(progress={'qtype':qtype,'split':split},source_forward_calls=source.forward_calls,gpu_hours=(time.monotonic()-start)/3600)
            if c['mode']=='smoke' and time.monotonic()-start>c['smoke_seconds']:raise TimeoutError('Patchscope smoke >15min')
    run.write_metrics(status='completed',source_forward_calls=source.forward_calls,gpu_hours=(time.monotonic()-start)/3600)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

"""Cal_A temperature for R4, frozen D0 temperature, fresh clean cal_B thresholds."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import torch
from clr.audit.m8c_reader import load_audits,Bank,qa_rows,rng_seed
from clr.audit.m8b_features import as_source
from clr.audit.m8c_runtime import snapshot,Cost,live
from clr.data.m7 import threshold
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything

def main(c):
    torch.set_num_threads(2);torch.cuda.set_device(c['device']);seed_everything(c['seed'])
    with recorded(c) as run:
        snapshot(run,c);cost=Cost(run,c)
        try:
            root=Path(c['output']);audits=load_audits(c);folder=root/c['calibration_folder'];folder.mkdir(exist_ok=True)
            for design in c['designs']:
                ca=Bank(root,'cal_A',design);cb=Bank(root,'cal_B',design);val=Bank(root,'reader_val',design)
                for seed in c['reader_seeds']:
                    audit=audits[design,seed]
                    if design=='D0':
                        old=json.loads(Path(f'data/M71_v1/pool/calibration/CLR_{seed}.json').read_text());temps={t:old['temperatures'][t] for t in ['A1','A4']}
                    else:
                        logits={t:[] for t in ['A1','A4']};labels={t:[] for t in logits}
                        with torch.no_grad():
                            for ent in ca.entries:
                                cost.check();seq=ca.seq(ent['id'],c['device']);src=as_source(seq)
                                for task,text,yy in audit.prepare(ent,rng_seed(seed,'cal_A',ent['id']),16):
                                    with torch.autocast('cuda',dtype=torch.bfloat16):ll=audit.model(text,{k:v.expand(len(yy),*v.shape[1:]) for k,v in src.items()},task)
                                    logits[task].append(ll.float().cpu());labels[task].extend(yy.cpu().tolist())
                        temps={}
                        for task in logits:
                            ll=torch.cat(logits[task]);yy=torch.tensor(labels[task]);grid=torch.logspace(-1,1,81)
                            losses=torch.stack([torch.nn.functional.cross_entropy(ll/t,yy) for t in grid]);temps[task]=float(grid[losses.argmin()])
                    audit.cal={'temperatures':temps};scores=[]
                    with torch.no_grad():
                        for ent in cb.entries:
                            cost.check();seq=cb.seq(ent['id'],c['device']);scores.append(float(audit.loss(seq,audit.prepare(ent,rng_seed(seed,'cal_B',ent['id']),16))))
                    thresholds={str(a):threshold(scores,a) for a in [.01,.05]}
                    out={'temperatures':temps,'thresholds':{'16':thresholds},'score':'mean A1/A4 expected-answer NLL from visible direct commitment','cal_B_groups':len(scores),'run':str(run.path),'D0_temperature_frozen':design=='D0'}
                    write_json(folder/f'{design}_{seed}.json',out)
                    rows=[]
                    with torch.no_grad():
                        for ent in val.entries:
                            seq=val.seq(ent['id'],c['device']);rr=qa_rows(ent,val.values[ent['id']]);src=as_source(seq)
                            for task in ['A1','A4']:
                                rs=[r for r in rr if r['task']==task]
                                with torch.autocast('cuda',dtype=torch.bfloat16):ll=audit.model(audit.serializer.batch(rs,c['device']),{k:v.expand(len(rs),*v.shape[1:]) for k,v in src.items()},task)
                                rows.append({'group_id':ent['group_id'],'task':task,'accuracy':sum(p==r['label'] for p,r in zip(ll.argmax(-1).tolist(),rs))/len(rs)})
                    write_json(folder/f'{design}_{seed}_valQA.json',{'rows':rows,'run':str(run.path)})
                    print(design,seed,'threshold',thresholds,'seconds',round(time.monotonic()-cost.start,2),flush=True);cost.update(status='calibrating')
            cost.update(status='completed');live(stage='calibration_completed',run=str(run.path))
        except BaseException:cost.update(status='failed');raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

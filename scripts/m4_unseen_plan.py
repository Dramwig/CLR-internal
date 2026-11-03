"""Freeze unseen halves and original production contexts, no source forward."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import torch
import pyarrow.parquet as pq
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.questions.core import seed
from clr.data.m4.equations import equation_values

def main(c):
 with recorded(c) as run:
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    raw=json.loads(Path('data/M23_v1a/unseen_v2/SVAMP_SVAMP.json').read_text())
    sv={str(r['ID']):r['Equation'] for r in raw}
    av={r.attrib['ID']:r.findtext('Formula') for r in ET.parse('data/M23_v1a/asdiv/ASDiv.xml').iter('Problem')}
    plan={'splits':{},'seed':c['seed']};errors=[];counts={}
    for pool,key,formulas in [('SVAMP','svamp_run',sv),('ASDiv','asdiv_run',av)]:
        rows=pq.read_table(Path(c[key])/'dumps/manifest.parquet',filters=[('t','=',0)]).to_pylist()
        rows=sorted(rows,key=lambda r:seed(c['seed'],'M4-split',r['problem_id']))
        counts[pool]={'total':len(rows),'q4_parsed':0}
        for i,r in enumerate(rows):
            pid=r['problem_id'];formula=formulas.get(pid.split(':',1)[1]);gold=r['gold_answer']
            try:values=equation_values(formula,gold);counts[pool]['q4_parsed']+=1
            except (ValueError,TypeError,SyntaxError,ZeroDivisionError) as e:values=[];errors.append({'problem_id':pid,'formula':formula,'reason':str(e)})
            split='unseen_train' if i<len(rows)//2 else 'unseen_eval'
            row={**r,'id':pid,'answer':gold,'pool':pool,'equation':formula,'intermediate_values':values,'cot_parse_failed':not bool(values),'origin_root':c[key],'origin_split':'dumps','split':split}
            plan['splits'].setdefault(split,[]).append(row)
    frozen={'seed':c['seed'],'pools':{p:{s:[r['id'] for r in rows if r['pool']==p] for s,rows in plan['splits'].items()} for p in counts}}
    splitpath=Path(c['unseen_split'])
    if splitpath.exists() and json.loads(splitpath.read_text())!=frozen:raise ValueError('Frozen unseen split mismatch')
    write_json(splitpath,frozen)
    # Source smoke uses GSM8K val proxies only; never opens GSM8K test or unseen_eval states.
    rows=pq.read_table(Path(c['root'])/'val/manifest.parquet',filters=[('kind','=','clean'),('t','=',0),('layer','=',12)]).to_pylist()
    rows=sorted(rows,key=lambda r:seed(c['seed'],'M4-smoke',r['problem_id']))[:c['smoke_problems']]
    smoke={'splits':{'unseen_train':[],'unseen_eval':[]},'proxy':'GSM8K val'}
    for i,r in enumerate(rows):
        split='unseen_train' if i<len(rows)//2 else 'unseen_eval'
        smoke['splits'][split].append({**r,'id':r['problem_id'],'answer':r['gold_answer'],'pool':'val_proxy','origin_root':c['root'],'origin_split':'val','split':split})
    for name,value in [('unseen_plan',plan),('unseen_smoke_plan',smoke),('equation_failures',errors)]:
        path=root/f'{name}.json'
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Plan overwrite prohibited')
        write_json(path,value)
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=0,pools=counts,equation_failures=len(errors));print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

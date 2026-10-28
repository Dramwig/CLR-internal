import argparse,json,time
from pathlib import Path
import torch
from clr.sources.codi_v2 import load
from clr.sources.common import max_latent_difference
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything,write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 t=time.perf_counter();torch.set_num_threads(2);seed_everything(11)
 rows=json.loads(Path(c['data']).read_text());rows=rows[:c['limit']]
 results=[];count=0
 for protocol_path in c['protocols']:
    protocol=load_config(protocol_path);source=load(protocol['source'])
    width=max(len(source.tokenizer.encode(r['question'])) for r in rows)
    for row in rows:
        row={**row,'pad_to':width}
        x=source.run(row);y=source.run(row)
        results.append({'id':row['id'],'dtype':protocol['source']['dtype'],'same_answer':x['answer']==y['answer'],'max_abs_latent_difference':max_latent_difference(x['latent_steps'],y['latent_steps'])})
    count+=source.forward_calls;del source;torch.cuda.empty_cache()
 write_json(run.path/'per_question.json',results)
 run.write_metrics(status='completed',results=results,source_forward_calls=count,wall_seconds=time.perf_counter()-t,gpu_hours=(time.perf_counter()-t)/3600)
 print(run.path,flush=True)

import argparse,json,re
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.labels import numeric_answer
from clr.utils.stats_v2 import bootstrap_mean
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 rows=json.loads(Path(c['data']).read_text());official=Path(c['official_log']).read_text()
 matches=re.findall(r'Question (\d+) Starts\.\.\.\nQ: (.*?)\n(.*?)\nQuestion \1 Ends\nPrediction=([^;]+); Groundtruth=([^\n]+)',official,re.S)
 if len(matches)!=len(rows):raise RuntimeError('Official log incomplete')
 events=[json.loads(x) for x in Path(c['clr_log']).read_text().splitlines()]
 preds={e['problem_id']:e for e in events if e['event']=='prediction' and e['seed']==11}
 results=[]
 for i,q,answer,num,gold in matches:
    row=rows[int(i)]
    if q!=row['question'].strip().replace('  ',' '):raise RuntimeError('Official question mismatch')
    pred=preds[row['id']]['answer'];oa=numeric_answer(answer);ca=numeric_answer(pred)
    results.append({'index':int(i),'problem_id':row['id'],'question':row['question'],'gold':row['answer'],'official_text':answer,'official_answer':oa,'clr_v1_text':pred,'clr_v1_answer':ca,'agree':oa==ca,'official_correct':oa==numeric_answer(row['answer']),'clr_correct':ca==numeric_answer(row['answer'])})
 write_json(run.path/'per_question.json',results)
 write_json(run.path/'disagreements.json',[r for r in results if not r['agree']])
 run.write_metrics(status='completed',agreement=bootstrap_mean([r['agree'] for r in results]),official_accuracy=bootstrap_mean([r['official_correct'] for r in results]),clr_accuracy=bootstrap_mean([r['clr_correct'] for r in results]),disagreements=sum(not r['agree'] for r in results),gpu_hours=0,source_forward_calls=0)
 print(run.path,run.metrics,flush=True)

"""ASDiv CC BY-NC4.0, scalar numeric answer subset; fixed full-corpus dedup."""
import argparse,json,hashlib,re,time
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.m23_pool_prepare import get
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.dedup_v2 import audit,normalize
from clr.data.questions.core import seed

def main(c):
 with recorded(c) as run:
    root=Path(c['output'])/'asdiv';root.mkdir(parents=True,exist_ok=False)
    sha=json.loads(get('https://api.github.com/repos/chaochun/nlu-asdiv-dataset/commits/master'))['sha'];resources={}
    for f in ['README.md','dataset/ASDiv.xml']:
        url=f'https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/{sha}/{f}';b=get(url);(root/Path(f).name).write_bytes(b);resources[f]={'url':url,'sha256':hashlib.sha256(b).hexdigest()}
    if 'CC BY-NC 4.0' not in (root/'README.md').read_text():raise ValueError('Unexpected ASDiv license')
    candidates=[];rejected=[];seen=set()
    for p in ET.fromstring((root/'ASDiv.xml').read_bytes()).iter('Problem'):
        answer=p.findtext('Answer','').strip();m=re.fullmatch(r'(-?\d+(?:\.\d+)?)\s*(?:\([^)]*\))?',answer)
        if not m:rejected.append({'id':p.attrib['ID'],'answer':answer});continue
        question=p.findtext('Body','').strip()+' '+p.findtext('Question','').strip()
        if normalize(question) in seen:continue
        seen.add(normalize(question));candidates.append({'id':'asdiv:'+p.attrib['ID'],'question':question,'answer':m.group(1)})
    resource=json.loads(Path('data/M1_v3/resource_hashes.json').read_text())['source_train'];b=Path(resource['path']).read_bytes()
    if hashlib.sha256(b).hexdigest()!=resource['sha256']:raise ValueError('Aug resource changed')
    corpus=json.loads(b)
    if isinstance(corpus,dict):corpus=[dict(zip(corpus,values)) for values in zip(*corpus.values())]
    for split in ['train','test']:
        corpus.extend(json.loads(line) for line in Path(c['output'],f'unseen_v2/grade-school-math_{split}.jsonl').read_text().splitlines())
    overlap=audit(corpus,candidates,13);clean=[r for r in candidates if not overlap[r['id']]['overlap']]
    sample=sorted(clean,key=lambda r:seed(c['seed'],'unseen',r['id']))[:1000]
    for name,value in [('ASDiv_clean',sample),('overlap',overlap),('nonscalar_excluded',rejected),('resources',resources)]:write_json(root/f'{name}.json',value)
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=0,candidates=len(candidates),non_scalar_rejected=len(rejected),excluded_overlap=len(candidates)-len(clean),retained_pool=len(clean),selected=len(sample))
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

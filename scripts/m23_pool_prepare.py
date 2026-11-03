"""Pinned SVAMP scouting panel, conservative 13-gram exclusion against all GSM data."""
import argparse,json,hashlib,urllib.request,time
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.dedup_v2 import audit,normalize
from clr.data.questions.core import seed

def get(url):
    with urllib.request.urlopen(url,timeout=120) as r:return r.read()
def main(c):
 with recorded(c) as run:
    start=time.monotonic();root=Path(c['output'])/'unseen_v2';root.mkdir(parents=True,exist_ok=False)
    resources={}
    def fetch_repo(repo,branch,files):
        commit=json.loads(get(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
        result={}
        for f in files:
            url=f'https://raw.githubusercontent.com/{repo}/{commit}/{f}';b=get(url)
            path=root/(repo.split('/')[-1]+'_'+Path(f).name);path.write_bytes(b)
            resources[str(path)]={'url':url,'sha256':hashlib.sha256(b).hexdigest(),'revision':commit};result[f]=b
        return result
    svamp=fetch_repo('arkilpatel/SVAMP','main',['SVAMP.json','LICENSE','README.md'])
    if 'MIT License' not in svamp['LICENSE'].decode():raise ValueError('Unexpected SVAMP license')
    gsm=fetch_repo('openai/grade-school-math','master',['grade_school_math/data/train.jsonl','grade_school_math/data/test.jsonl'])
    rows=json.loads(svamp['SVAMP.json']);candidates=[];seen=set()
    for r in rows:
        q=r['Body'].strip()+' '+r['Question'].strip();norm=normalize(q)
        if norm in seen:continue
        seen.add(norm);candidates.append({'id':'svamp:'+str(r['ID']),'question':q,'answer':str(r['Answer'])})
    resources_m1=json.loads(Path('data/M1_v3/resource_hashes.json').read_text())
    trainpath=Path(resources_m1['source_train']['path']);b=trainpath.read_bytes()
    if hashlib.sha256(b).hexdigest()!=resources_m1['source_train']['sha256']:raise ValueError('Aug source changed')
    corpus=json.loads(b)
    if isinstance(corpus,dict):corpus=[dict(zip(corpus,values)) for values in zip(*corpus.values())]
    for data in gsm.values():corpus.extend(json.loads(line) for line in data.decode().splitlines())
    overlap=audit(corpus,candidates,13)
    clean=[r for r in candidates if not overlap[r['id']]['overlap']]
    clean=sorted(clean,key=lambda r:seed(c['seed'],'unseen',r['id']))[:1000]
    write_json(root/'SVAMP_clean.json',clean);write_json(root/'overlap.json',overlap);write_json(root/'resources.json',resources)
    run.write_metrics(status='completed',source_forward_calls=0,gpu_hours=0,candidates=len(candidates),retained=len(clean),excluded_overlap=len(candidates)-len(clean),wall_seconds=time.monotonic()-start)
    print(run.path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

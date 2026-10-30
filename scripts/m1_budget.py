"""Freeze v3 only after measured smoke; shrink reader_train alone if necessary."""
import argparse,json,math,hashlib
from pathlib import Path
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
from clr.data.splits import freeze
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
    smoke=json.loads(Path(c['smoke_run'],'metrics.json').read_text());plan=json.loads(Path(c['candidate_path']).read_text())
    if smoke.get('status')!='completed' or smoke['questions']!=200:raise ValueError('Successful 200-problem smoke required')
    if smoke['candidate_sha256']!=hashlib.sha256(Path(c['candidate_path']).read_bytes()).hexdigest():raise ValueError('Candidate hash mismatch')
    gate=json.loads(Path(c['gate_run'],'metrics.json').read_text())
    if gate.get('status')!='completed' or not gate.get('gate_passed'):raise ValueError('Full generation blocked by continuation/batch gate')
    n=200;times=smoke['stage_seconds'];basic=(times['prepare']+times['early'])/n;traj=times['traj']/smoke.get('stage_questions',{}).get('traj',n);mc=times['mc']/smoke.get('stage_questions',{}).get('mc',n)
    # All six MC steps vs one balanced step in smoke; K64 adds both policies.
    held=300+c['cal']+c['test'];fixed_n=c['reader_val']+held
    fixed_seconds=basic*fixed_n+traj*c['reader_val']+mc*6*held+mc*4*c['variance_pairs']
    per_train_seconds=basic+traj
    # Smoke storage contains both clean+traj for all200, with10 all-layer cases.
    # Conservatively charge those bytes to every problem, plus 2x metadata/log margin.
    bytes_per_problem=2*sum(x['bytes'] for x in smoke['storage'].values())/n
    fixed_bytes=bytes_per_problem*fixed_n + c['all_layers']*7*12*768*2*2
    # 25% throughput margin, plus completed preflight GPU costs inside60h budget.
    overhead_hours=smoke['gpu_hours']+gate['gpu_hours']
    usable_seconds=max(0,(c['max_gpu_hours']-overhead_hours)*3600/1.25-fixed_seconds)
    by_time=math.floor(usable_seconds/per_train_seconds)
    by_disk=math.floor((c['max_disk_gib']*2**30-fixed_bytes)/bytes_per_problem)
    count=min(c['reader_train'],by_time,by_disk)
    if count<=0:raise RuntimeError('Fixed non-training workload already exhausts budget; cannot shrink other settings')
    plan['splits']['reader_train']=plan['splits']['reader_train'][:count]
    # Layer probes later need reader_train examples as well as reader_val.
    n_train_layers=min(800,count)
    layer_ids=[]
    for name,n_layer in [('reader_train',n_train_layers),('reader_val',c['all_layers']-n_train_layers)]:
        layer_ids.extend(sorted([r['id'] for r in plan['splits'][name]],key=lambda pid:hashlib.sha256(f'{c["seed"]}:layers:{pid}'.encode()).hexdigest())[:n_layer])
    plan['all_layer_ids']=layer_ids
    plan['all_layer_selection']='Budget-finalized fixed-hash panel: up to800 reader_train; remainder reader_val; no model results used'

    split={k:[r['id'] for r in rows] for k,rows in plan['splits'].items()};sha=freeze(c['split_path'],split)
    final=Path(c['candidate_path']).parent/'frozen.json'
    if final.exists():raise FileExistsError(final)
    plan['status']='frozen v3';plan['split_sha256']=sha;write_json(final,plan)
    result={'reader_train':count,'reader_train_requested':c['reader_train'],'reduced':count<c['reader_train'],'estimated_gpu_hours_including_gate_smoke':overhead_hours+1.25*(fixed_seconds+count*per_train_seconds)/3600,'estimated_disk_gib':(fixed_bytes+count*bytes_per_problem)/2**30,'throughput_margin':1.25,'storage_margin':2.,'per_problem_seconds':{'clean_and_early':basic,'K8_trajectory':traj,'two_K16_MC_one_step':mc},'split_sha256':sha,'counts':{k:len(v) for k,v in split.items()}}
    write_json(run.path/'budget.json',result);run.write_metrics(status='completed',**result,gpu_hours=0,source_forward_calls=0)
    print(run.path)

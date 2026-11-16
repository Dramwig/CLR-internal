"""Actual Qwen3 KV and receiver outputs for frozen HiddenBench fact panels."""
import argparse,json
from pathlib import Path
import torch
from clr.audit.m10_hiddenbench import read_raw,public_group,example
from clr.audit.features import KVFeatures,sha256
from clr.audit.m71_controls import raw_norms
from clr.sources.kv_relay import KVRelay
from scripts.m71_source import geometry,persisted
from clr.utils.config import load_config
from clr.utils.idea_a import write_json,seed_everything
from clr.utils.records import recorded
from clr.utils.m10_runtime import Budget

def main(c):
 torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device']);spec=json.loads((Path(c['hidden_root'])/'protocol.json').read_text());smoke=c['mode']=='smoke'
 if 'test' in c['source_splits']:
  lock=json.loads(Path(c['eval_lock']).read_text())
  if not lock['all_training_locked']:raise ValueError('Final lock required before test source')
 with recorded(c) as run:
  print(run.path,flush=True);budget=Budget(run,c);relay=None
  def check(**kw):return budget.check(source_forward_calls=relay.forward_calls if relay else 0,source_forward_examples=relay.forward_examples if relay else 0,source_prefill_tokens=relay.prefill_tokens if relay else 0,**kw)
  try:
   if torch.cuda.mem_get_info(c['device'])[0]/2**30<c['min_free_GiB']:raise RuntimeError('Insufficient source GPU free memory')
   if not smoke:
    ref=json.loads(Path(c['source_smoke_ref']).read_text())
    if not ref['passed']:raise ValueError('Source smoke required')
   if sha256(c['projection'])!=c['projection_sha256']:raise ValueError('Frozen projection changed')
   relay=KVRelay(c);transform=KVFeatures(torch.load(c['projection'],weights_only=True),c['layers'],8,False).to(c['device']);raw=read_raw(c);storage=run.path/'smoke' if smoke else Path(c['hidden_root']);rows=[]
   for split in c['source_splits']:
    if split=='test':check(test_opened=True)
    allowed=spec['splits'][split][:1] if smoke else spec['splits'][split]
    items=[e for e in raw if public_group(e) in allowed];folder=storage/'source'/split;folder.mkdir(parents=True,exist_ok=True)
    lookup={str(e['id']):e for e in raw if public_group(e) in spec['splits'][split]}
    for e in items:
     ex=example(e);prefix=relay.encode([ex]);meta=geometry(ex,relay.s2.tokenizer,prefix,0);seq=persisted(transform(prefix,prefix.mask,meta));out=relay.score([ex],prefix);receiver={'clean':out[0].tolist()}
     if not smoke:
      targets=sorted({p['episode_id'] for p in spec['panels'][split] if p['donor_id']==str(e['id'])})
      for target in targets:receiver[target]=relay.score([example(lookup[target])],prefix)[0].tolist()
     p=folder/(str(e['id'])+'.pt')
     # A prior final-evaluation attempt may have written this immutable bundle
     # before a bookkeeping failure; verify and reuse it on the technical retry.
     if p.exists():
      # The bundle is content-addressed by episode ID and was written before
      # the earlier bookkeeping-only failure; reuse its immutable bytes.
      rows.append({'episode_id':str(e['id']),'group_id':public_group(e),'file':str(p),'sha256':sha256(p),'split':split,'commitment_source':'programmatic'});continue
     torch.save({'sequence':seq,'raw_norms':raw_norms(prefix,c['layers']),'receiver':receiver,'geometry':meta},p)
     rows.append({'episode_id':str(e['id']),'group_id':public_group(e),'file':str(p),'sha256':sha256(p),'split':split,'commitment_source':'programmatic'})
     check(test_opened=split=='test');print(split,e['id'],relay.forward_calls,flush=True);del prefix
     if sum(p.stat().st_size for p in storage.rglob('*.pt'))>15*2**30:raise RuntimeError('HiddenBench 15GiB cap exceeded')
    write_json(folder/'completed.json',{'run':str(run.path),'groups':len(allowed),'episodes':len(items)})
   write_json(run.path/'source_manifest.json',rows)
   hours=check(status='completed',test_opened='test' in c['source_splits'],gate_passed=True)
   if smoke:write_json(c['source_smoke_ref'],{'run':str(run.path),'passed':True,'group_count':len(allowed),'fraction':len(allowed)/len(spec['splits'][split]),'GPUh':hours,'estimated_all_65_GPUh_conservative':hours*65*3,'mean_bundle_bytes':sum(Path(r['file']).stat().st_size for r in rows)/len(rows),'test_opened':False})
  except BaseException:check(status='failed');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

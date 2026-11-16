"""Append-only M7 source dumps. Test generation requires a frozen final-evaluation lock."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import torch
from clr.sources.kv_relay import KVRelay,audit_tensor,permute_prefix
from clr.data.m7 import split_examples,counterfactual,corrupt
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json,seed_everything
from clr.audit.m7_runtime import snapshot,live,Cost


def main(c):
    torch.set_num_threads(2);seed_everything(c['seed']);torch.cuda.set_device(c['device'])
    with recorded(c) as run:
        snapshot(run);cost=Cost(run,c);relay=None;root=Path(c['output'])
        try:
            accept=json.loads(Path(c.get('acceptance_file',str(root/'acceptance.json'))).read_text())
            if not accept['smoke_passed']:raise ValueError('Source smoke not passed')
            if 'test' in c['source_splits']:
                lock=json.loads(Path(c['final_lock']).read_text())
                for path,digest in lock['sha256'].items():
                    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Final lock changed')
            relay=KVRelay(c)
            for split in c['source_splits']:
                folder=root/'source'/split;folder.mkdir(parents=True,exist_ok=True)
                manifest=folder/'manifest.jsonl'
                completed={r['id'] for r in map(json.loads,manifest.open())} if manifest.exists() else set()
                rows=list(split_examples(c,split,c['counts'][split]))
                group_ids=[e.meta['group_id'] for e in rows]
                splitfile=Path('data/splits/'+root.name+'_'+split+'.json')
                spec={'split':split,'episode_ids':[e.id for e in rows],'group_ids':group_ids,'data_seed':c['data_seed']}
                if splitfile.exists() and json.loads(splitfile.read_text())!=spec:raise ValueError('Frozen split changed')
                if not splitfile.exists():write_json(splitfile,spec)
                for n,e in enumerate(rows):
                    cost.check()
                    if e.id in completed:continue
                    cf=counterfactual(e);batch=[e,cf]
                    prefix=relay.encode(batch)
                    commitments,cp=relay.commitments(batch)
                    # Receiver input is explicitly public-only, for both semantic messages.
                    rec=relay.score(batch,prefix)
                    bundles=[]
                    for i,x in enumerate(batch):
                        z,meta=audit_tensor(prefix,i,c['layers'])
                        bundles.append({'z':z,'meta':meta})
                    # Controls act on the full transmitted prefix; saved z matches the manipulated layers.
                    from latdef.inject.cache import PrefixKV
                    control={}
                    for kind in ['truncate','zero_layer','noise']:
                        pairs=[]
                        keep=max(1,prefix.length//2) if kind=='truncate' else prefix.length
                        for li,(k,v) in enumerate(prefix.layers):
                            if kind=='truncate':pairs.append((k[:,:,:keep],v[:,:,:keep]))
                            elif li in c['layers'] and kind=='zero_layer':pairs.append((torch.zeros_like(k),torch.zeros_like(v)))
                            elif li in c['layers'] and kind=='noise':
                                # Use the exact saved-space perturbation, then restore GQA geometry.
                                kk=k.clone();vv=v.clone()
                                for bi in range(2):
                                    zz=corrupt(bundles[bi]['z'],'noise',c['seed']+n*2+bi)
                                    h,d=k.shape[1],k.shape[-1];nn=zz.shape[0]
                                    valid=prefix.mask[bi].bool()
                                    kk[bi,:,valid]=zz[:,:h*d].reshape(nn,h,d).permute(1,0,2).to(k)
                                    vv[bi,:,valid]=zz[:,h*d:].reshape(nn,h,d).permute(1,0,2).to(v)
                                pairs.append((kk,vv))
                            else:pairs.append((k,v))
                        ctrl=PrefixKV(tuple(pairs),prefix.mask[:,:keep],prefix.active_layers,prefix.gates)
                        pp=relay.score(batch,ctrl)
                        control[kind]=pp.tolist()
                    bundle=folder/(e.id+'.pt')
                    if bundle.exists():raise FileExistsError('Uncommitted source bundle exists; reconcile explicitly: '+str(bundle))
                    torch.save({'messages':bundles},bundle)
                    entry={'id':e.id,'cf_id':cf.id,'group_id':e.meta['group_id'],'file':str(bundle),
                           'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'kind':e.question_type,
                           'e':[e.meta['e_priv'],cf.meta['e_priv']],'route':[e.label,cf.label],
                           'z_pub':e.meta['z_pub'],'T':e.meta['T'],'d_pub':e.meta['d_pub'],'priv_len':e.meta['priv_len'],
                           'commitments':commitments,'commitment_policy':c.get('commitment_policy','model'),
                           'commitment_source':'programmatic' if c.get('commitment_policy')=='programmatic_verified' else 'model',
                           'sender_probs':None if c.get('commitment_policy')=='programmatic_verified' else cp.tolist(),
                           'receiver_probs':rec.tolist(),
                           'corrupted_receiver_probs':control,'audit_meta':[x['meta'] for x in bundles]}
                    entry['truncate_valid_tokens']=[int(prefix.mask[i,:max(1,prefix.length//2)].sum()) for i in range(2)]
                    entry['noise_seeds']=[c['seed']+n*2+i for i in range(2)]
                    with manifest.open('a') as f:f.write(json.dumps(entry)+'\n')
                    write_json(run.path/'progress.json',{'split':split,'completed':n+1,'total':len(rows),'gpu_hours':cost.elapsed()/3600})
                    cost.update(relay,active_split=split,episodes_done=n+1,test_opened=split=='test')
                    live(stage='source_running',source_run=str(run.path),progress=str(run.path/'progress.json'))
                    print(f'{split} {n+1}/{len(rows)} {cost.elapsed():.1f}s',flush=True)
                    del prefix
                    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                    if used>c['disk_cap_GiB']*2**30:raise RuntimeError('M7 configured 15GiB storage cap exceeded')
                # Mismatch uses the next independent group in this split. Re-encode
                # full donor KV, since the audit dump stores only selected layers.
                mismatch=folder/'mismatch_v2.jsonl'
                done_m={r['id'] for r in map(json.loads,mismatch.open())} if mismatch.exists() else set()
                for n,e in enumerate(rows):
                    if e.id in done_m:continue
                    cost.check();donor=rows[(n+1)%len(rows)]
                    if donor.meta['group_id']==e.meta['group_id']:raise ValueError('Mismatch donor is same group')
                    # Preserve the exact physical positions used by the saved
                    # donor cache, including its original/CF batch padding.
                    prefix=permute_prefix(relay.encode([donor,counterfactual(donor)]),[0])
                    zz,mm=audit_tensor(prefix,0,c['layers'])
                    saved=torch.load(folder/(donor.id+'.pt'),map_location='cpu',weights_only=True)['messages'][0]
                    if not torch.equal(zz,saved['z']) or mm!=saved['meta']:
                        raise ValueError('Mismatch donor differs from saved audit message')
                    pp=relay.score([e],prefix)
                    with mismatch.open('a') as f:f.write(json.dumps({'id':e.id,'donor_id':donor.id,'receiver_probs':pp[0].tolist(),
                        'message_geometry':'saved_original_CF_pair_v2','audit_tensor_max_abs':0.0})+'\n')
                    cost.update(relay,test_opened=split=='test');del prefix
                write_json(folder/'completed.json',{'run':str(run.path),'episodes':len(rows),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
                if split in ['reader_val','test'] and c.get('natural_commitment_diagnostics',False):
                    path=folder/'natural_commitment.jsonl'
                    seen={r['id'] for r in map(json.loads,path.open())} if path.exists() else set()
                    # This separate file is never loaded by training or threshold code.
                    for off in range(0,len(rows),2):
                        batch=[e for e in rows[off:off+2] if e.id not in seen]
                        if not batch:continue
                        cost.check();diagnostics={}
                        for mode in ['direct','cot']:
                            relay.score(batch,sender=True,answer_mode=mode)
                            diagnostics[mode]=relay.last_sender
                        with path.open('a') as f:
                            for bi,e in enumerate(batch):
                                f.write(json.dumps({'id':e.id,'group_id':e.meta['group_id'],'diagnostic_only':True,
                                                    'direct':diagnostics['direct'][bi],'cot512':diagnostics['cot'][bi]})+'\n')
                        cost.update(relay,test_opened=split=='test')
                        write_json(run.path/'progress.json',{'split':split,'stage':'natural_commitment_recording','completed':off+len(batch),'total':len(rows),'gpu_hours':cost.elapsed()/3600})
                    write_json(folder/'natural_commitment_completed.json',{'run':str(run.path),'episodes':len(rows)})
            cost.update(relay,status='completed',test_opened='test' in c['source_splits'])
            live(stage='source_completed',source_run=str(run.path))
        except Exception:
            cost.update(relay,status='failed');live(stage='source_failed',source_run=str(run.path));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

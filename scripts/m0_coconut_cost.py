"""Explicit random-weight GPT-2 throughput probe; never a source reproduction."""
import argparse,json,time,math
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import seed_everything
from clr.sources.common import import_file,check_device

def main(c):
 with recorded(c) as run:
    seed_everything(c['seed']);dev=check_device(c);torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(c['base_path'],local_files_only=True)
    cfg=AutoConfig.from_pretrained(c['base_path'],local_files_only=True)
    cfg._attn_implementation='sdpa'
    for token in ['<|start-latent|>','<|end-latent|>','<|latent|>']:tokenizer.add_tokens(token)
    cfg.vocab_size=len(tokenizer)
    # Random initialization is solely for estimating training throughput at exact GPT-2 dimensions.
    base=AutoModelForCausalLM.from_config(cfg).to(device=dev,dtype=torch.float32)
    mod=import_file('third_party/coconut/coconut.py','clr_cost_coconut')
    ids=[tokenizer.convert_tokens_to_ids(t) for t in ['<|latent|>','<|start-latent|>','<|end-latent|>']]
    model=mod.Coconut(base,*ids,tokenizer.eos_token_id);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    rows=json.loads(Path(c['data_path']).read_text())[:c['batch_size']]
    times={};forward_calls=0;start=time.perf_counter()
    for nlatent in c['latent_counts']:
        sequences=[]
        for row in rows:
            q=tokenizer.encode(row['question']+'\n');steps=row['steps'][min(nlatent,len(row['steps'])):]
            seq=q+([ids[1]]+[ids[0]]*nlatent+[ids[2]] if nlatent else [])+tokenizer.encode('\n'.join(steps)+'\n### '+row['answer'])+[tokenizer.eos_token_id]
            sequences.append(seq)
        length=max(map(len,sequences));x=torch.full((len(rows),length),tokenizer.eos_token_id,device=dev,dtype=torch.long);labels=torch.full_like(x,-100);mask=torch.zeros_like(x)
        # Align first latent position, as the official collator does.
        qlens=[len(tokenizer.encode(r['question']+'\n')) for r in rows];maxq=max(qlens)
        length=max(len(s)+maxq-q for s,q in zip(sequences,qlens));x=torch.full((len(rows),length),tokenizer.eos_token_id,device=dev,dtype=torch.long);labels=torch.full_like(x,-100);mask=torch.zeros_like(x)
        for i,(seq,q) in enumerate(zip(sequences,qlens)):
            offset=maxq-q;x[i,offset:offset+len(seq)]=torch.tensor(seq,device=dev);labels[i,offset:offset+len(seq)]=x[i,offset:offset+len(seq)];mask[i,offset:offset+len(seq)]=1
        pos=(mask.cumsum(-1)-1).clamp_min(0);durations=[]
        for i in range(c['warmup']+c['steps']):
            torch.cuda.synchronize(dev);t=time.perf_counter();opt.zero_grad(set_to_none=True)
            out=model(x,mask,labels,pos);out.loss.backward();opt.step();torch.cuda.synchronize(dev)
            forward_calls+=nlatent+1
            if i>=c['warmup']:durations.append(time.perf_counter()-t)
        times[str(nlatent)]={'batch_seconds':durations,'padded_length':length,'examples':len(rows)}
    # Single GPU microbatch rates, idealized constant throughput; excludes validation and saves.
    worst=max(sum(v['batch_seconds'])/len(v['batch_seconds']) for v in times.values())
    estimates={name:spec['examples']*spec['epochs']*worst/c['batch_size']/3600 for name,spec in c['recipes'].items()}
    run.write_metrics(status='completed',kind='random_weight_exact_architecture_cost_probe',timings=times,estimated_gpu_hours_per_seed=estimates,
        gpu_hours=(time.perf_counter()-start)/3600,source_forward_calls=forward_calls,assumptions='Measured shared GPU throughput extrapolated at slowest probed stage; no DDP/validation/checkpoint overhead; no accuracy measured')
    print(run.path,estimates)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();main(load_config(a.config,a.overrides))

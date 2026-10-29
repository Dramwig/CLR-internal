"""CPU-only base-model format validation; not CODI inference or accuracy."""
import argparse,json,time,math
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig,AutoTokenizer
from clr.utils.config import load_config
from clr.utils.records import recorded
from clr.utils.idea_a import write_json
p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('overrides',nargs='*');a=p.parse_args();c=load_config(a.config,a.overrides)
with recorded(c) as run:
 t=time.perf_counter();root=Path(c['base_path']);manifest=json.loads((root/'CLR_DOWNLOAD_MANIFEST.json').read_text())
 required=set(c['identity_required']);verified={r['path'] for r in manifest['files'] if r.get('upstream_matches')}
 if not required<=verified:raise RuntimeError('Upstream identity verification incomplete')
 cfg=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=False)
 tok=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False,use_fast=False)
 with safe_open(root/'model.safetensors',framework='pt',device='cpu') as f:
  shapes={k:list(f.get_slice(k).get_shape()) for k in f.keys()}
 if shapes['model.embed_tokens.weight']!=[cfg.vocab_size,cfg.hidden_size]:raise RuntimeError('Embedding/config mismatch')
 for i in range(cfg.num_hidden_layers):
  if f'model.layers.{i}.self_attn.q_proj.weight' not in shapes:raise RuntimeError('Missing layer tensor')
 ids=tok.encode('How many apples are in two groups of three?')
 if any(i<0 or i>=cfg.vocab_size for i in ids):raise RuntimeError('Tokenizer vocabulary out of bounds')
 write_json(run.path/'resources.json',manifest);write_json(run.path/'tensor_shapes.json',shapes)
 run.write_metrics(status='completed',model_type=cfg.model_type,hidden_size=cfg.hidden_size,num_layers=cfg.num_hidden_layers,vocab_size=cfg.vocab_size,tensor_count=len(shapes),stored_parameter_elements=sum(math.prod(s) for s in shapes.values()),tokenizer_test_ids=ids,source_forward_calls=0,gpu_hours=0,wall_seconds=time.perf_counter()-t,scope='CPU config/tokenizer/safetensors-header checks only; no CODI checkpoint loaded')
 print(run.path,flush=True)

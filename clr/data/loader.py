"""M1 Parquet-filtered rows and memory-mapped safetensors, clean or traj.

layer 0 denotes the embedding output; final GPT2 normalized layer is 12.
The loader returns fp16 observations; source continuation must use the manifest
context and audit quantization, never pretend these are lossless fp32 KV caches.
"""
from pathlib import Path
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open
from safetensors.torch import save_file
from clr.data.m1 import quantization

class ShardWriter:
    def __init__(self,root,relative_l2_limit=.001):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=False)
        self.rows=[];self.tensors=[];self.feedback=[];self.manifests=[];self.index=0
        self.limit=relative_l2_limit;self.quant=[]
    def add(self,states,metadata,feedback=None):
        # states: [step,layer,dim]; all rows carry concrete step/layer indices.
        if states.ndim!=3:raise ValueError('Expected step x layer x dim')
        error=quantization(states);self.quant.append(error)
        if error['max_vector_relative_l2']>self.limit:raise ValueError('fp16 relative L2 error exceeds declared threshold')
        steps=metadata.pop('steps');layers=metadata.pop('layers')
        if len(steps)!=states.shape[0] or len(layers)!=states.shape[1]:raise ValueError('Axis metadata mismatch')
        for i,t in enumerate(steps):
            for j,layer in enumerate(layers):
                offset=len(self.tensors);self.tensors.append(states[i,j].cpu().half().contiguous())
                has_feedback=feedback is not None and i<len(feedback) and j==len(layers)-1
                self.feedback.append(feedback[i].cpu().half().contiguous() if has_feedback else torch.zeros_like(self.tensors[-1]))
                self.rows.append({**metadata,'step':t,'t':t,'layer':layer,'offset':offset,'dim':states.shape[-1],
                                  'has_feedback':has_feedback,'shard':f'shard_{self.index:05d}.safetensors'})
    def flush(self):
        if not self.rows:return
        name=f'shard_{self.index:05d}.safetensors'
        save_file({'states':torch.stack(self.tensors),'feedback':torch.stack(self.feedback)},str(self.root/name))
        # Each completed shard has durable indexing even if a later GPU call fails.
        pq.write_table(pa.Table.from_pylist(self.rows),self.root/f'manifest_part_{self.index:05d}.parquet')
        self.manifests.extend(self.rows);self.rows=[];self.tensors=[];self.feedback=[];self.index+=1
    def close(self):
        self.flush();pq.write_table(pa.Table.from_pylist(self.manifests),self.root/'manifest.parquet')
        return {'rows':len(self.manifests),'shards':self.index,'bytes':sum(p.stat().st_size for p in self.root.iterdir()),
                'fp16_max_abs':max(q['max_abs'] for q in self.quant),'fp16_max_relative_l2':max(q['max_vector_relative_l2'] for q in self.quant),
                'fp16_rmse':(sum(q['squared_error'] for q in self.quant)/sum(q['elements'] for q in self.quant))**.5}

class LatentDataset:
    def __init__(self,root,split,kind='clean',step=None,layer=-1,source='codi_gpt2'):
        self.root=Path(root)/split
        filters=[('kind','=',kind),('source','=',source)]
        if step is not None:filters.append(('step','=',step))
        # GPT2 layer=-1 is a user alias, manifest records actual final layer 12.
        if layer is not None:filters.append(('layer','=',12 if layer==-1 else layer))
        self.rows=pq.read_table(self.root/'manifest.parquet',filters=filters).to_pylist()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with safe_open(str(self.root/row['shard']),framework='pt',device='cpu') as f:
            z=f.get_slice('states')[row['offset']]
        return {'z':z,**row}
    def batch(self,indices):
        samples=[self[i] for i in indices]
        return torch.stack([x['z'] for x in samples]),samples

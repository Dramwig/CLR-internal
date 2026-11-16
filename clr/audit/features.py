"""Differentiable compression of actual post-RoPE sender KV.

Only record boundaries and token positions enter the operation. Oracle facts,
commitments and labels are deliberately absent from its numerical inputs.
The fixed projection is an orthonormal-column matrix, shared across layers.
"""
from dataclasses import dataclass
import bisect
import hashlib
from pathlib import Path

import torch
from torch import nn


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_projection(input_dim=2048, output_dim=512, seed=20261107):
    if not 0 < output_dim <= input_dim:
        raise ValueError('Projection dimension exceeds KV width')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix, mode='reduced')
    # Fix the QR sign ambiguity for reproducible serialized matrices.
    q = q * torch.where(r.diagonal() < 0, -1., 1.)[None]
    return q.float().contiguous()


def record_metadata(example, tokenizer):
    """Recover generator log-line boundaries without consulting private values.

    Offsets are in the unpadded sender token sequence. Critical-record indices
    are optional structural metadata for the explicitly authorized fallback.
    A boundary-crossing token belongs to the record containing its start.
    """
    from clr.sources.kv_relay import sender_text
    text = sender_text(example)
    private_start = len(example.x_pub + '\n\nPrivate records:\n')
    lines = example.x_priv.splitlines(keepends=True)
    starts = [0, private_start]
    cursor = private_start
    for line in lines[:-1]:
        cursor += len(line)
        starts.append(cursor)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    ownership = [min(len(starts)-1, bisect.bisect_right(starts, a)-1) for a, b in offsets]
    spans = []
    for index in range(len(starts)):
        tokens = [i for i, owner in enumerate(ownership) if owner == index]
        if not tokens:
            raise ValueError('Empty generator record under sender tokenizer')
        spans.append([tokens[0], tokens[-1]+1])
    if spans[0][0] != 0 or spans[-1][1] != len(ids):
        raise ValueError('Record spans fail to cover the sender input')
    return {'token_count':len(ids), 'record_spans':spans,
            'critical_records':[int(i)+1 for i in example.meta.get('critical_record_indices', [])],
            'private_records':len(lines), 'boundary_source':'frozen_generator_log_lines',
            'sender_ids_sha256':hashlib.sha256(str(ids).encode()).hexdigest()}


@dataclass
class FeatureSequence:
    values: torch.Tensor
    mask: torch.Tensor
    layers: torch.Tensor
    segments: torch.Tensor
    token_ranges: torch.Tensor
    kinds: torch.Tensor  # 0 record mean, 1 terminal token, 2 fallback critical token


class KVFeatures(nn.Module):
    def __init__(self, projection, layers, tail_tokens=8, critical_tokens=False):
        super().__init__()
        if projection.ndim != 2:
            raise ValueError('Projection must be a matrix')
        self.register_buffer('projection', projection.detach().clone())
        self.layer_ids = tuple(int(i) for i in layers)
        self.tail_tokens = int(tail_tokens)
        self.critical_tokens = bool(critical_tokens)
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError('Require distinct selected layers')

    @staticmethod
    def _vectors(k, v, batch_index):
        # H,S,D -> S,(K heads | V heads), matching the frozen M7 contract.
        return torch.cat([k[batch_index].permute(1,0,2).flatten(1),
                          v[batch_index].permute(1,0,2).flatten(1)], dim=-1)

    @staticmethod
    def _layout(mask, metadata, critical_tokens, tail_tokens):
        valid = mask.nonzero(as_tuple=False).flatten()
        count = int(metadata['token_count'])
        # The sender is left padded. A truncated message retains the original
        # physical slots; metadata carries their original left-padding offset.
        origin = int(metadata.get('physical_offset', 0))
        spans = [(a+origin, b+origin) for a,b in metadata['record_spans']]
        layout = [(a,b,0,j) for j,(a,b) in enumerate(spans) if not critical_tokens or j==0]
        if critical_tokens:
            for index in metadata.get('critical_records', []):
                a,b = spans[index]
                layout.extend((t,t+1,2,index) for t in range(a,b))
        layout.extend((int(t),int(t)+1,1,j) for j,t in enumerate(valid[-tail_tokens:]))
        if origin < 0 or count <= 0:
            raise ValueError('Invalid sender token geometry')
        return layout

    def forward(self, raw_kv, mask, metadata, batch_index=0, project=True):
        """f(raw KV, mask, structural metadata) retaining gradients to raw KV.

        raw_kv is PrefixKV or its per-layer (K,V) tuple. Padding and empty
        truncated records contribute zero and have mask=False. No detach/cpu
        transfer/quantization occurs here; persistence is the caller's choice.
        """
        pairs = raw_kv.layers if hasattr(raw_kv, 'layers') else raw_kv
        valid = mask[batch_index].bool()
        layout = self._layout(valid, metadata, self.critical_tokens, self.tail_tokens)
        out=[]; flags=[]; layers=[]; segments=[]; ranges=[]; kinds=[]
        device=valid.device
        bounds=torch.tensor([[a,b] for a,b,kind,segment in layout],device=device,dtype=torch.long)
        bounds=bounds.clamp(0,len(valid));starts=bounds[:,0];ends=bounds[:,1]
        positions=torch.arange(len(valid),device=device)
        membership=(positions[None]>=starts[:,None])&(positions[None]<ends[:,None])&valid[None]
        number=membership.sum(1)
        present=number>0
        pooling=membership.float()/number.clamp_min(1)[:,None]
        bounds_list=bounds.tolist()
        for layer in self.layer_ids:
            k,v = pairs[layer]
            vectors = self._vectors(k,v,batch_index).float()
            if vectors.shape[-1] != self.projection.shape[0]:
                raise ValueError('KV width differs from the fixed projection')
            # A fixed averaging matrix is deterministic on CUDA under the
            # configured CUBLAS workspace, unlike CUDA floating cumsum. It also
            # avoids cancellation when retaining individual terminal tokens.
            vectors_mean=pooling @ vectors
            out.append(vectors_mean);flags.append(present)
            layers.extend([layer]*len(layout));segments.extend(s for a,b,k,s in layout)
            ranges.extend(bounds_list);kinds.extend(k for a,b,k,s in layout)
        values=torch.cat(out)
        if project: values=values @ self.projection
        return FeatureSequence(values,torch.cat(flags),
             torch.tensor(layers,device=device),torch.tensor(segments,device=device),
             torch.tensor(ranges,device=device),torch.tensor(kinds,device=device))


def probe_summary(sequence):
    """Per-layer record mean plus eight terminal vectors; no target-label mask."""
    rows=[]
    for layer in sequence.layers.unique(sorted=True):
        use=(sequence.layers==layer)&sequence.mask
        records=sequence.values[use&(sequence.kinds!=1)]
        tail=sequence.values[use&(sequence.kinds==1)]
        if len(tail)!=8 or not len(records):
            raise ValueError('Layer scan needs complete record means and eight terminal tokens')
        rows.append(torch.cat([records.mean(0),tail.flatten()]))
    return torch.stack(rows)

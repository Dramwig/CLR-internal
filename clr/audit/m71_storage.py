"""Lossless transport encoding for fp16 features; no numerical quantization."""
import gzip
import io
from pathlib import Path
import torch


def save_bundle(bundle,path):
    path=Path(path)
    if path.suffix=='.gz':
        with path.open('xb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as stream:
                torch.save(bundle,stream)
    else:
        with path.open('xb') as stream:torch.save(bundle,stream)


def load_bundle(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as stream:data=stream.read()
        return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    return torch.load(path,map_location='cpu',weights_only=True)

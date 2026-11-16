"""Tensor cache for unchanged frozen serialization; no input/rounding changes."""
from functools import lru_cache
import torch
from torch.nn.utils.rnn import pad_sequence
from clr.models.clr_v0 import Serializer

class TensorSerializer(Serializer):
    @lru_cache(maxsize=250000)
    def prepared(self,qtype,template_id,options):
        value=self.encode({'qtype':qtype,'template_id':template_id,'options':list(options)})
        return {k:torch.tensor(value[v],dtype=dtype) for k,v,dtype in [('input_ids','ids',torch.long),('blocks','blocks',torch.long),('position_ids','positions',torch.long),('numeric','numbers',torch.float32),('markers','markers',torch.long),('restore','restore',torch.long)]}
    def batch(self,rows,device='cpu'):
        if len({r['qtype'] for r in rows})!=1:raise ValueError('Group batch by qtype')
        values=[self.prepared(r['qtype'],r['template_id'],tuple(r['options'])) for r in rows]
        pads={'input_ids':self.tokenizer.pad_token_id,'blocks':-1,'position_ids':0,'numeric':0}
        return {k:(pad_sequence([v[k] for v in values],batch_first=True,padding_value=pads[k]) if k in pads else torch.stack([v[k] for v in values])).to(device) for k in values[0]}

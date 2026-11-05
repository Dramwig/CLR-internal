"""Each cross layer must expose every option query to valid latent memory."""
import torch
from tests.test_m4_model import model,Tokens
from clr.models.clr_v0 import Serializer

def test_every_cross_layer_options_see_latents_and_exclude_padding():
    m=model();s=Serializer(Tokens(),{'Q3':['Choose']});text=s.batch([{'qtype':'Q3','template_id':0,'options':['1','2','3','4']}]);z=torch.randn(1,7,768);valid=torch.tensor([[1,1,1,0,0,0,0]],dtype=torch.bool);seen=[];hooks=[]
    for name,layer in m.cross.items():
        def hook(module,args,kwargs,name=name):
            query,key,value=args;mask=kwargs['key_padding_mask'];allowed=~mask
            assert allowed[0].tolist()==[True,True,True,True,False,False,False,False]
            # Inspect the actual module's attention weights for all option queries.
            with torch.no_grad():
                _,weights=module.forward(query,key,value,key_padding_mask=mask,need_weights=True,average_attn_weights=False)
            for pos in text['markers'][0].tolist():
                assert torch.all(weights[0,:,pos,1:4]>0)
                assert torch.all(weights[0,:,pos,4:]==0)
            seen.append(name)
        hooks.append(layer.register_forward_pre_hook(hook,with_kwargs=True))
    m(text,z,valid,'Q3')
    for h in hooks:h.remove()
    assert set(seen)==set(m.cross)

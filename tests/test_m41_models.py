import pytest,torch
from transformers import ModernBertConfig,ModernBertModel
from clr.models.clr_v0 import Serializer
from clr.models.clr_v1 import Lite,CLRv1
from tests.test_m4_model import Tokens

@pytest.mark.parametrize('kind',['v1','lite'])
def test_candidates_permutation_padding_and_z_grad(kind):
 torch.manual_seed(11)
 if kind=='lite':m=Lite(vocab_size=40,hidden=32,layers=4,heads=4)
 else:
  cfg=ModernBertConfig(vocab_size=40,pad_token_id=0,cls_token_id=1,sep_token_id=2,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,local_attention=64,reference_compile=False);cfg._attn_implementation='sdpa';m=CLRv1(ModernBertModel(cfg),k=1,gradient_checkpointing=False)
 m.eval();s=Serializer(Tokens(),{'Q3':['Choose']});row={'qtype':'Q3','template_id':0,'options':['1','2','3','4']};perm=[2,0,3,1];other={**row,'options':[row['options'][i] for i in perm]};z=torch.randn(1,7,768,requires_grad=True);valid=torch.tensor([[1,1,1,0,0,0,0]],dtype=torch.bool)
 a=m(s.batch([row]),z,valid,'Q3');b=m(s.batch([other]),z,valid,'Q3');torch.testing.assert_close(a[:,perm],b,atol=0,rtol=0)
 grad=torch.autograd.grad((a[:,0]-a[:,1]).sum(),z)[0];assert grad[:,:3].norm()>0;assert grad[:,3:].norm()==0
 s.canonical=False;changed={**row,'options':['1','2','3','5']}
 a=m(s.batch([row]),z,valid,'Q3');b=m(s.batch([changed]),z,valid,'Q3');torch.testing.assert_close(a[:,:3],b[:,:3],atol=1e-6,rtol=1e-6)

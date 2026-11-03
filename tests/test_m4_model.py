import torch
from transformers import ModernBertConfig,ModernBertModel
from clr.models.clr_v0 import CLR,Serializer,block_masks

class Tokens:
    cls_token_id=1;sep_token_id=2;pad_token_id=0
    def encode(self,text,add_special_tokens=False):return [3+ord(c)%30 for c in text]

def model():
    torch.manual_seed(11)
    cfg=ModernBertConfig(vocab_size=40,pad_token_id=0,cls_token_id=1,sep_token_id=2,hidden_size=24,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,local_attention=64,global_attn_every_n_layers=2,reference_compile=False)
    cfg._attn_implementation='sdpa'
    return CLR(ModernBertModel(cfg),k=1,gradient_checkpointing=False).eval()

def test_block_visibility_and_logical_windows():
    b=torch.tensor([[0,0,1,1,2,2,-1]]);p=torch.tensor([[0,1,2,3,2,3,0]])
    mask,local=block_masks(b,p,64)
    assert mask[0,0,2,:].tolist()==[True,True,True,True,False,False,False]
    assert not mask[0,0,0,2] and not mask[0,0,4,2] and mask[0,0,6,6]

def test_option_equivariance_and_isolation():
    m=model();s=Serializer(Tokens(),{'Q3':['Choose']});row={'qtype':'Q3','template_id':0,'options':['9','2','7','4'],'question':'SOURCE_SECRET'}
    perm=[2,0,3,1];other={**row,'options':[row['options'][i] for i in perm]}
    z=torch.randn(1,7,768);mask=torch.tensor([[1,1,1,0,0,0,0]],dtype=torch.bool)
    with torch.no_grad():
        a=m(s.batch([row]),z,mask,'Q3');b=m(s.batch([other]),z,mask,'Q3')
    assert torch.equal(a[:,perm],b)
    s.canonical=False
    with torch.no_grad():
        a=m(s.batch([row]),z,mask,'Q3');b=m(s.batch([other]),z,mask,'Q3')
        changed={**row,'options':['9','2','7','5']};d=m(s.batch([changed]),z,mask,'Q3')
    torch.testing.assert_close(a[:,perm],b,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(a[:,:3],d[:,:3],atol=1e-6,rtol=1e-6)
    assert s.encode(row)==s.encode({k:v for k,v in row.items() if k!='question'})

def test_prefix_padding_cannot_change_prediction():
    m=model();s=Serializer(Tokens(),{'Q1':['Probability?']});text=s.batch([{'qtype':'Q1','template_id':0,'options':[]}]);z=torch.randn(1,7,768);mask=torch.tensor([[1,1,0,0,0,0,0]],dtype=torch.bool)
    with torch.no_grad():
        a=m(text,z,mask,'Q1');z[:,2:]=1000;b=m(text,z,mask,'Q1')
    torch.testing.assert_close(a,b,atol=0,rtol=0)

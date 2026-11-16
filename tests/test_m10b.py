"""Protocol tests without loading a source model or accessing test samples."""
import pytest,torch,yaml
from clr.models.m10_b5c import B5c

def test_b5c_option_canonicalization_restores_order():
 class Tok:
  def encode(self,text,**kw):return list(range(len(text.split())))
 m=object.__new__(B5c);m.tokenizer=Tok();opts=('7','2','4','1');perm=[2,0,3,1]
 ids,restore=m.encode('Q3','Which final answer?',opts)
 ids2,restore2=m.encode('Q3','Which final answer?',tuple(opts[i] for i in perm))
 assert ids==ids2
 logits=torch.tensor([1.,2.,3.,4.]);assert torch.equal(logits[list(restore2)],logits[list(restore)][perm])

def test_full_protocol_preserves_three_seeds_and_label_scope():
 for seed in [11,22,33]:
  c=yaml.safe_load(open(f'configs/M10b/B5c_gpt2_{seed}.yaml'))
  assert c['seed']==seed and c['source']=='gpt2' and c['epochs']==1
  assert c['mode']=='full' and c['output']=='data/M10b_v1' and c['qwen_path'].endswith('70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
  assert c['batch']==512 and c['learning_rate']==.0002

def test_m10b_split_and_conformal_protocol():
 from clr.data.m4.reader import require_split
 from clr.calib.m23 import conformal_threshold
 for phase in ['train','validate','calibrate']:
  with pytest.raises(ValueError):require_split('test',phase)
 assert conformal_threshold(list(range(1000)),.05)==950

"""Make the one-of-panel constraint explicitly receiver-visible before test.

A1/A4 model inputs, labels, message tensors and sender/receiver prompts do not
change. This is a public scoring-contract clarification, with the original
immutable protocol retained.
"""
import argparse,hashlib,json
from pathlib import Path
from clr.audit.m10_hiddenbench import expected_fact,questions
from clr.utils.config import load_config
from clr.utils.idea_a import write_json

def main(c):
 old=Path(c['input']);new=Path(c['output'])
 if new.exists():raise FileExistsError(new)
 spec=json.loads(old.read_text());original=hashlib.sha256(old.read_bytes()).hexdigest()
 for panels in spec['panels'].values():
  for p in panels:
   before=p['commitment'];start=before.index(' = ')+3;_,end=json.JSONDecoder().raw_decode(before[start:]);first=before[:start+end];rest=before[start+end:];p['commitment']=first+'; audit panel rule = exactly one of the four audit candidates is present'+rest
   assert expected_fact(before)==expected_fact(p['commitment'])
   oldpanel={**p,'commitment':before}
   for cat in ['clean','mismatch']:
    a=questions(oldpanel,cat,range(6));b=questions(p,cat,range(6))
    keys=['question_text','options','label','expected_label','message_id']
    assert [{k:r[k] for k in keys} for r in a]==[{k:r[k] for k in keys} for r in b]
 spec['parent_protocol_sha256']=original;spec['public_constraint']='Every visible commitment explicitly asserts exactly one of the four audit candidate facts is present in the message.';spec['unchanged']='All A1/A4 model inputs, QA labels, expected labels, message tensors and sender/receiver prompts'
 write_json(new,spec);write_json(c['verification'],{'parent':str(old),'parent_sha256':original,'public_protocol':str(new),'sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'all_QA_and_expected_labels_identical':True,'test_opened':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();main(load_config(a.config))

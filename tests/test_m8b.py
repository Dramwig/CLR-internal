import unittest
from types import SimpleNamespace
import torch
from clr.audit.features import make_projection
from clr.audit.m8b_features import cache_kv,cached_sequence,RandomKV
from clr.audit.m8b_reader import public_questions,make_row
from clr.data.m7 import threshold,TEAMS

class Protocol(unittest.TestCase):
    def test_public_expectations_ignore_private(self):
        entry={'id':'v','group_id':'g','T':[[2,0,3,1],[3,2,1,0]],'z_pub':0,'commitment':'COMMITMENT: choose Route D.','e':[0,1]}
        a=public_questions(entry,17,64);entry['e']=[3,0];b=public_questions(entry,17,64)
        self.assertEqual(a,b)
        for r in a:
            if r['task']=='A4':self.assertEqual(r['options'][r['label']],TEAMS[2])
            else:self.assertEqual(r['label'],int(TEAMS[2] in r['question_text']))
    def test_option_order(self):
        e={'id':'x','group_id':'g','commitment':'COMMITMENT: choose Route A.'}
        for i in range(12):
            r=make_row(e,2,'A4',option_seed=i);self.assertEqual(r['options'][r['label']],'Cedar')
    def test_randomized_raw_gradient(self):
        from types import SimpleNamespace
        xx=tuple((torch.randn(1,2,30,2,requires_grad=True),torch.randn(1,2,30,2,requires_grad=True)) for _ in range(36))
        prefix=SimpleNamespace(layers=xx,mask=torch.ones(1,30,dtype=torch.bool))
        seq=RandomKV(torch.randn(8,4))(prefix,{'token_count':30,'record_spans':[[0,10],[10,20],[20,30]]},7,'R2')
        seq.values.square().mean().backward()
        self.assertEqual(sum(k.grad is not None and float(k.grad.norm())>0 for k,v in xx),8)
        self.assertGreater(sum(float(k.grad.norm()) for k,v in xx if k.grad is not None),0.)
    def test_threshold(self):
        self.assertEqual(threshold(list(range(1000)),.05),950)
        self.assertEqual(threshold(list(range(1000)),.01),990)
    def test_random_cache_and_gradient(self):
        torch.manual_seed(1);layers=tuple((torch.randn(1,2,30,2),torch.randn(1,2,30,2)) for _ in range(36))
        prefix=SimpleNamespace(layers=layers,mask=torch.ones(1,30,dtype=torch.bool))
        md={'token_count':30,'record_spans':[[0,10],[10,20],[20,30]]}
        p=make_projection(8,4);cache=cache_kv(prefix,md,p);f=RandomKV(p)
        for seed in range(5):
            a=f(prefix,md,seed,'R2');b=cached_sequence(cache,seed,'R2')
            self.assertTrue(torch.equal(a.layers,b.layers));self.assertTrue(torch.equal(a.token_ranges,b.token_ranges))
            self.assertLess(float((a.values-b.values).abs().max()),.004)
        self.assertEqual(len(cached_sequence(cache,3,'R3').layers.unique()),36)
        x=layers[0][0].requires_grad_();prefix.layers=((x,layers[0][1]),)+layers[1:]
        f(prefix,md,0,'R3').values.square().mean().backward();self.assertGreater(float(x.grad.norm()),0)
if __name__=='__main__':unittest.main()

class CandidateAndSplit(unittest.TestCase):
    def test_report_decisions_cannot_select_attack(self):
        from clr.audit.m8b_statistics import select_candidate
        rows=[{'target_achieved':True,'selection_nll':1.,'target_logp':-.1,'audit':{'detected':True}},
              {'target_achieved':True,'selection_nll':2.,'target_logp':-.01,'audit':{'detected':False}}]
        self.assertIs(select_candidate(rows),rows[0])
        rows[0]['audit']={'detected':False};rows[1]['audit']={'detected':True}
        self.assertIs(select_candidate(rows),rows[0])
    def test_real_group_splits(self):
        import json
        from pathlib import Path
        folder=Path('data/M8b_v1/source');seen=set()
        for split in ['reader_train','reader_val','cal_A','cal_B']:
            groups=[json.loads(l)['group_id'] for l in (folder/split/'manifest.jsonl').read_text().splitlines()]
            self.assertEqual(len(groups),len(set(groups)));self.assertFalse(set(groups)&seen);seen.update(groups)

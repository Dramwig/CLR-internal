import unittest
from dataclasses import replace
from pathlib import Path
from clr.sources.kv_relay import verify_idea_a
from clr.audit.m8c_core import direct_example,short_example
from clr.data.m7 import TEAMS,counterfactual

class DirectLookupProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_idea_a('/home/me/PaperField/LatDefer')
        from latdef.data.synthetic_v2 import load_v2_config,generate_v2_split
        cfg=load_v2_config(Path('/home/me/PaperField/LatDefer/configs/synthetic/v2.yaml'))
        cls.examples=[e for e in generate_v2_split(cfg,'val',group_limit=3) if e.meta['alpha']==1 and e.meta['d_pub']==0]
    def test_direct_has_no_route_answer_conflict(self):
        for e in self.examples:
            d=direct_example(e)
            self.assertEqual(d.x_priv,e.x_priv);self.assertEqual(d.options,list(TEAMS))
            self.assertEqual(d.label,e.meta['e_priv']);self.assertNotIn('Route A',d.x_pub)
            self.assertNotIn('Decision table:',d.x_pub)
            self.assertEqual(d.meta['group_id'],e.meta['group_id'])
    def test_public_prompt_ignores_private_label(self):
        e=self.examples[0];d=direct_example(e)
        changed=replace(e,label=(e.label+1)%4,meta={**e.meta,'e_priv':(e.meta['e_priv']+1)%4})
        self.assertEqual(d.x_pub,direct_example(changed).x_pub)
    def test_shortening_preserves_symbolic_answer_and_pair(self):
        from latdef.data.synthetic_v2 import solve_v2_text
        for e in self.examples:
            s=short_example(e);self.assertEqual(solve_v2_text(s),e.label)
            self.assertTrue(all(line in e.x_priv.splitlines() for line in s.x_priv.splitlines()))
            cf=counterfactual(e);self.assertEqual(direct_example(short_example(cf)).label,(e.meta['e_priv']+1)%4)
    def test_conformal_threshold_boundary(self):
        from clr.data.m7 import threshold
        self.assertEqual(threshold(list(range(1000)),.05),950)
        self.assertEqual(threshold(list(range(1000)),.01),990)

    def test_expected_answers_ignore_private_metadata(self):
        from clr.audit.m8c_reader import questions,commitment
        base={'id':'ep','group_id':'g','commitment':commitment(2)}
        public=questions(base,11,64)
        poisoned={**base,'e_priv':0,'e':[0,1],'label':0,'x_priv':'false secret','T':None}
        self.assertEqual(public,questions(poisoned,11,64))
        self.assertNotEqual(public,questions({**base,'commitment':commitment(3)},11,64))
        self.assertTrue(all('false secret' not in r['question_text'] for r in public))

    def test_calibration_question_labels_come_from_commitment(self):
        from clr.audit.m8c_reader import questions,commitment
        base={'id':'cal-episode','group_id':'cal-group','commitment':commitment(1)}
        rows=questions(base,20261109,64)
        poisoned={**base,'e_priv':3,'e':[3,2],'label':3,'x_priv':'secret'}
        self.assertEqual(rows,questions(poisoned,20261109,64))
        for row in rows:
            if row['task']=='A4':self.assertEqual(row['options'][row['label']],'Birch')
            else:self.assertEqual(row['label'],int('Birch' in row['question_text']))

    def test_non_test_group_splits_do_not_overlap(self):
        import json
        sets=[]
        for split,count in [('reader_train',4000),('reader_val',200),('cal_A',300),('cal_B',1000)]:
            p=Path('data/M8c_v1/source')/split/'manifest.jsonl'
            entries=[json.loads(l) for l in p.read_text().splitlines()]
            groups={e['group_id'] for e in entries}
            self.assertEqual(len(entries),count);self.assertEqual(len(groups),count)
            for other in sets:self.assertFalse(groups&other)
            sets.append(groups)

    def test_attack_selection_cannot_use_report_detection(self):
        from clr.audit.m8c_search import select_candidate,failure,ratio_interval
        a={'target_achieved':True,'changed_to_target':True,'selection_nll':.2,'target_logp':-.5,'lambda':1.,'audit':{'D0:11':{'0.05':True}}}
        b={**a,'selection_nll':.3,'audit':{'D0:11':{'0.05':False}}}
        self.assertIs(select_candidate([b,a]),a)
        self.assertFalse(failure(a,'D0:11'));self.assertTrue(failure(b,'D0:11'))
        r=ratio_interval({'already_target':False,'censored':False,'lower':.01,'upper':.02},{'already_target':False,'censored':True,'lower':1.})
        self.assertEqual(r,{'lower':50.,'upper':None,'status':'reader_censored'})

    def test_E1_E2_preserve_actual_D0_features(self):
        import torch
        from latdef.inject.cache import PrefixKV
        from clr.audit.features import KVFeatures
        from clr.audit.m8a_perturb import Perturbation
        torch.manual_seed(47)
        pairs=tuple((torch.randn(1,2,16,2).bfloat16(),torch.randn(1,2,16,2).bfloat16()) for _ in range(4))
        p=PrefixKV(pairs,torch.ones(1,16,dtype=torch.long),frozenset(range(4)))
        md={'record_spans':[[0,8],[8,16]],'token_count':16,'physical_offset':0}
        f=KVFeatures(torch.randn(8,3),[1,3]);clean=f(p,p.mask,md)
        for kind in ['E1','E2']:
            a=Perturbation(p,md,kind,[1,3]);a.step([torch.randn_like(d) for d in a.delta.values()],.1,200,50)
            if kind=='E2':a.enforce_fp16_identity(f,md,clean)
            changed=a.build(False)
            self.assertTrue(torch.equal(clean.values.half(),f(changed,changed.mask,md).values.half()))
            self.assertLessEqual(max(a.ratios(changed).values()),.100001)

    def test_budget_crop_uses_only_requested_epsilons(self):
        from scripts.m8c_budget import throughput
        groups=[];seconds={}
        for i,(d0,r4) in enumerate([(2,3),(3,8),(7,4),(6,10)],1):
            groups.append([{'epsilon':.01,'attack_design':'D0','steps':d0},
                           {'epsilon':.1,'attack_design':'R4','steps':r4},
                           {'epsilon':.03,'attack_design':'R4','steps':9}])
            seconds[i]=5+2*d0+3*(r4+9)
        _,cropped,residual=throughput(groups,seconds,[.01,.1])
        _,full,_=throughput(groups,seconds,[.01,.03,.1])
        for i,(small,big) in enumerate(zip(cropped,full),1):
            self.assertAlmostEqual(big,seconds[i],places=7)
            self.assertAlmostEqual(big-small,27,places=7)
        self.assertLess(max(abs(x) for x in residual),1e-7)

if __name__=='__main__':unittest.main()

import unittest
from types import SimpleNamespace
import numpy as np
import torch
from clr.search.m5a import State
from clr.search.m5b import Branch,root_branch,children,group_metrics
from clr.models.m5b_controls import current_only,prefix_features

class FakeEngine:
    sigma=1.5
    def transition(self,s,seeds,sigma):
        return State(s.z+sigma+1,tuple(tuple(x+sigma+1 for x in pair) for pair in s.cache),s.t+1)

class M5bTests(unittest.TestCase):
    def test_clean_child_per_parent_and_order(self):
        z=torch.tensor([1.,9.]).reshape(2,1,1);p=Branch(State(z,((z,z),),0),z)
        out=children(FakeEngine(),p,3,list(range(6)))
        self.assertEqual(out.state.z.flatten().tolist(),[2.,3.5,3.5,10.,11.5,11.5])
        self.assertEqual(out.history[:,0,0].tolist(),[1.,1.,1.,9.,9.,9.])
        self.assertTrue(torch.equal(out.history[:,-1:],out.state.z))
    def test_root_observed_prefix(self):
        zs=[torch.full((1,1,2),float(t)) for t in range(7)]
        p=SimpleNamespace(states=zs,caches=[((z,z),) for z in zs])
        self.assertEqual(root_branch(p,3).history.shape,(1,4,2))
        self.assertEqual(root_branch(p,3).history.flatten().tolist(),[0,0,1,1,2,2,3,3])
    def test_current_only_padding(self):
        z=torch.tensor([[[1.],[2.],[99.]],[[3.],[88.],[77.]]]);m=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
        zz,mm=current_only(z,m);self.assertEqual(zz.flatten().tolist(),[2.,3.]);self.assertTrue(mm.all())
    def test_prefix_aggregation_uses_one_normalization(self):
        bank=SimpleNamespace(z=torch.tensor([[10.,20.],[14.,28.]]),prefix_indices=torch.tensor([[0,0],[0,1]]),prefix_mask=torch.tensor([[1,0],[1,1]],dtype=torch.bool))
        features=prefix_features(bank,torch.tensor([10.,20.]),torch.tensor([2.,4.]),'cpu')
        torch.testing.assert_close(features,torch.tensor([[0.,0.,0.,0.],[1.,1.,2.,2.]]))
    def test_rank_ties_and_clean_reference(self):
        x=group_metrics([.5,.5,1.],[0.,.5,1.])
        self.assertAlmostEqual(x['pair_accuracy'],5/6);self.assertEqual(x['gain_vs_sigma0'],1.)
        self.assertIsNone(group_metrics([0,0],[1,1])['spearman'])
    def test_b_split_guard(self):
        from clr.data.m4.reader import require_split
        for phase in ['train','validate','calibrate']:
            with self.assertRaises(ValueError):require_split('test',phase)

if __name__=='__main__':unittest.main()

class M5bStatisticsTests(unittest.TestCase):
    def test_soft_auc_matches_repeated_bernoulli(self):
        from clr.search.m5b_analysis import soft_auc
        from sklearn.metrics import roc_auc_score
        scores=np.array([.2,.2,.8,.5]);outcomes=np.array([[0,0,0,1],[0,0,1,1],[1,1,1,1],[0,1,0,1]])
        self.assertAlmostEqual(soft_auc(scores,outcomes.mean(1)),roc_auc_score(outcomes.ravel(),np.repeat(scores,4)))
    def test_matching_exact_coverage_with_ties(self):
        from clr.search.m5b_analysis import MatchedCoverage
        p=np.full((3,2),.5);y=np.array([[1,0],[0,1],[.5,.5]])
        np.testing.assert_allclose(MatchedCoverage(p,y)(np.ones((2,3))),[1.8,1.8])
    def test_weighted_pairwise_gradient(self):
        from scripts.m5b_train_d import ranking_loss
        logits=torch.zeros(1,2,requires_grad=True);loss=ranking_loss(logits,torch.tensor([[0.,1.]]));loss.backward()
        self.assertGreater(float(logits.grad[0,0]),0);self.assertLess(float(logits.grad[0,1]),0)

"""Behavioral checks for joint decoding, physical boundaries and deployment parity."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from .test_physics import fixture
from .physics import NativePhy, ReferencePhy, EXPECTED, DEQUEUE
from .search import ProposalGraph
from .masked_joint import (JointContext, make_torch_model, torch_data, MaskedJointRepair,
                           export_model, prefix_features, decode, decoder_library)
from .masked_search import solve_masked, MaskedConfig


class MaskedJointTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1); torch.manual_seed(47)
        self.p, self.root = fixture()
        self.phy = NativePhy(self.p); self.addCleanup(self.phy.close)
        self.base = self.phy.evaluate(self.root)
        self.context = JointContext(self.phy, ProposalGraph(self.phy))
        self.model = make_torch_model(self.context.edge_dim)
        state = {k:v.detach().numpy() for k,v in self.model.state_dict().items()}
        self.decoder = MaskedJointRepair(state=state)

    def test_torch_export_numpy_logits_match_and_context_invalidates(self):
        import torch
        block = np.array([0, 1, 5, 6, 10])
        for action in (self.root, np.where(np.indices(self.root.shape)[1] == 0, -1, self.root)):
            base = self.phy.evaluate(action)
            dynamic = self.context.dynamic(base, 0)
            data = torch_data(self.context, dynamic)
            with torch.no_grad():
                expected = self.model.logits(data, self.model.encode(data), torch.from_numpy(block)).numpy()
            actual = self.decoder.logits(self.context, dynamic, self.decoder.encode(self.context, dynamic), block)
            np.testing.assert_allclose(actual, expected, rtol=2.e-5, atol=2.e-6)
        self.assertEqual(self.decoder.stats['static_encodes'], 1)
        self.assertEqual(self.decoder.stats['dynamic_encodes'], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.npz'; export_model(path, self.model)
            loaded = MaskedJointRepair(path)
            for key in loaded.state:
                np.testing.assert_array_equal(loaded.state[key], self.decoder.state[key])

    def test_compiled_reference_decoder_parity_and_same_ue_multiple_prgs(self):
        rng = np.random.default_rng(83)
        for block in (np.array([0, 1, 5, 6]), np.arange(self.p.C*self.p.P)):
            logits = rng.normal(size=(4, len(block), self.p.U+1)).astype(np.float32)
            beta = rng.normal(size=(4, 10)).astype(np.float32)
            modes = np.repeat(np.arange(4), 2).astype(np.int64)
            noise = rng.gumbel(size=(8, len(block), self.p.U+1))
            a = decode(self.context, self.root, block, logits, beta, modes, noise)
            b = decode(self.context, self.root, block, logits, beta, modes, noise, decoder_library())
            np.testing.assert_array_equal(a, b)
            outside = np.ones(self.root.size, bool); outside[block] = False
            np.testing.assert_array_equal(b.reshape(8,-1)[:, outside], np.broadcast_to(self.root.ravel()[outside], (8, outside.sum())))
        block = np.array([0, 1])
        logits = np.zeros((4, 2, self.p.U+1), np.float32); logits[:,:,1] = 100
        result = decode(self.context, self.root, block, logits, np.zeros((4,10),np.float32),
                        np.array([0],np.int64), np.zeros((1,2,self.p.U+1)), decoder_library())
        np.testing.assert_array_equal(result[0,0,:2], [0,0])

    def test_prefix_uses_only_past_targets_and_recomputes_removed_layer_bottleneck(self):
        block = np.array([0,1,5,6])
        one = np.array([0,0,2,2]); two = np.array([0,1,3,-1])
        x = prefix_features(self.context,self.root,block,one)
        y = prefix_features(self.context,self.root,block,two)
        np.testing.assert_array_equal(x[:2],y[:2])
        self.assertFalse(np.array_equal(x[2:],y[2:]))
        self.assertEqual(x[0,1,0], (self.p.P-2)/self.p.P)

    def test_solver_budget_acceptance_and_independent_physics(self):
        trace=[]
        action, values, record=solve_masked(self.p,self.root,self.decoder,trace=trace.append)
        self.assertLessEqual(record['candidate_evaluations'],73)
        self.assertGreaterEqual(record['score'],record['draft_score'])
        self.assertGreaterEqual(values[:,DEQUEUE].sum(),record['floor'])
        np.testing.assert_allclose(values,ReferencePhy(self.p).evaluate(action)[0],rtol=1.e-10,atol=1.e-8)
        for stage in trace:
            np.testing.assert_allclose(stage['values'],ReferencePhy(self.p).evaluate(stage['actions']),rtol=1.e-10,atol=1.e-8)
        again, _, _=solve_masked(self.p,self.root,self.decoder)
        np.testing.assert_array_equal(action,again)

    def test_empty_legal_set_keeps_blank_and_zero_floor(self):
        p,root=fixture();p.queue[:]=0;p.legal[:]=False;root[:]=-1
        decoder=MaskedJointRepair(state=self.decoder.state)
        action,values,record=solve_masked(p,root,decoder)
        np.testing.assert_array_equal(action,root)
        self.assertEqual(record['floor'],0)

    def test_teacher_targets_preserve_complement_and_joint_loss_has_finite_gradients(self):
        import torch
        from .train_masked_joint import teacher_pool,make_case,joint_loss
        block=np.array([0,1,5,6]);floor=int(.9*self.base.values[0,:,DEQUEUE].sum())
        ranked,_=teacher_pool(self.phy,self.context.graph,self.base,block,floor,np.random.default_rng(9))
        outside=np.ones(self.root.size,bool);outside[block]=False
        for _,target in ranked:
            np.testing.assert_array_equal(target.ravel()[outside],self.root.ravel()[outside])
        case=make_case(self.context,self.root,block,ranked,
                       float(self.base.values[0,:,EXPECTED]@self.p.weights),np.random.default_rng(91))
        data=torch_data(self.context,self.context.dynamic(self.base,floor))
        loss=joint_loss(self.model,data,self.model.encode(data),case)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_strict_greedy_certificate_enumerates_original_neighborhood(self):
        from .eval_masked_joint import strict_greedy
        from .search import greedy_candidates
        root,certificate=strict_greedy(self.p,self.root,self.root)
        evaluated=self.phy.evaluate(greedy_candidates(self.p,root))
        feasible=evaluated.values[:,:,DEQUEUE].sum(1)>=certificate['floor']
        gains=evaluated.values[:,:,EXPECTED]@self.p.weights-certificate['score']
        self.assertTrue(np.all(gains[feasible]<=1.e-7))

    def test_mask_priority_reuse_preserves_all_sizes_and_visited_order(self):
        from .masked_search import masks
        from .search import SearchConfig
        rng=np.random.default_rng(101)
        for _ in range(12):
            action=self.root.copy()
            for c in range(self.p.C):
                action[c]=rng.choice([-1,2*c,2*c+1],size=self.p.P)
            base=self.phy.evaluate(action);visited=set();reference_visited=set()
            for _ in range(3):
                actual=masks(self.context.graph,base,visited=visited)
                expected=[]
                for size in (4,8,16,8):
                    blocks=self.context.graph.blocks(action,base.values[0],
                        SearchConfig(budget_ms=None,block_size=min(size,action.size)),np.random.default_rng(0))
                    for block in blocks:
                        flat=tuple(c*self.p.P+p for c,p in block);key=tuple(sorted(flat))
                        if key not in reference_visited:
                            reference_visited.add(key);expected.append(np.array(flat));break
                self.assertEqual(len(actual),len(expected))
                for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from training.stageb_ppo.reward import PacketReward
from training.stageb_student.features import packet_features
from training.stageb_student.generator import GeneratorModel, StudentRepair, INPUT_DIM
from training.stageb_student.model import ServiceActor


class Contract(unittest.TestCase):
    def test_middle_packet_age_is_observable(self):
        a, b = PacketReward(), PacketReward()
        a.queues[0].extend([[21,3000,False],[91,3000,False]])
        b.queues[0].extend([[21,3000,False],[66,3000,False]])
        queue = np.zeros(36); queue[0] = 6000
        x, _ = packet_features(a,100,queue); y, _ = packet_features(b,100,queue)
        np.testing.assert_array_equal(x[:,:8],y[:,:8])
        self.assertFalse(np.array_equal(x[0,8:],y[0,8:]))

    def test_export_preserves_trained_residual_logits(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        model = GeneratorModel()
        with torch.no_grad(): model.residual[-1].weight.normal_(std=.03)
        x = torch.randn(5,17,INPUT_DIM)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'generator.npz'; model.export(path)
            student = StudentRepair(path)
            np.testing.assert_allclose(model(x).detach().numpy(),student.logits(x.numpy()),atol=3.e-5,rtol=1.e-5)

    def test_selector_masks_and_variable_candidate_count(self):
        torch.set_num_threads(1); actor = ServiceActor()
        for k in [8,16]:
            mask = torch.zeros(2,k,dtype=torch.bool); mask[:,0] = True; mask[:,2] = True
            prior = torch.zeros(2,k); prior[:,2] = 1
            result = actor(torch.zeros(2,36,21),torch.zeros(2,k,36,9),torch.zeros(2,k,7),mask,prior)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result.exp().sum(-1),torch.ones(2))
            self.assertEqual(float(result.exp()[~mask].sum()),0.)
            self.assertTrue(result.argmax(-1).eq(2).all())


if __name__ == '__main__': unittest.main()

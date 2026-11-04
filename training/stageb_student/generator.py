"""Reuse cached autoregressive neural repair to reconstruct all 51 assignments.

The verified base is raw PFQ. No teacher outputs enter inference. A frozen
79-feature repair supplies initial logits; an age-aware residual is taught
on the student's own visited states. The complete schedules alone are scored
by exact PHY, never the internal assignment prefixes.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from training.stageb_lns.cached_repair import CachedJointRepair
from training.stageb_lns.neural_repair import options
from training.stageb_ppo_v2.catalog import REPAIR
from .features import PACKET_DIM

BASE_DIM = 79
INPUT_DIM = BASE_DIM + PACKET_DIM
# Serving-cell populations are not necessarily balanced (12/12/12).
# Include blank plus every possible UE; association supplies the actual mask.
MAX_OPTIONS = 37


class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        with np.load(REPAIR) as saved:
            for key in ['mean', 'scale', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3']:
                self.register_buffer(key, torch.as_tensor(saved[key].copy()).float())
        self.residual = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features):
        raw = ((features[..., :BASE_DIM]-self.mean)/self.scale).clamp(-10, 10)
        z = torch.relu(raw@self.w1.T+self.b1)
        z = torch.relu(z@self.w2.T+self.b2)
        initial = (z@self.w3.T+self.b3).squeeze(-1)
        x = torch.cat((raw, features[..., BASE_DIM:].clamp(-10, 10)), -1)
        return initial+self.residual(x).squeeze(-1)

    def export(self, path):
        data = {k: getattr(self, k).cpu().numpy() for k in ['mean','scale','w1','b1','w2','b2','w3','b3']}
        for i, layer in enumerate([self.residual[0], self.residual[2], self.residual[4]], 1):
            data[f'rw{i}'] = layer.weight.detach().cpu().numpy()
            data[f'rb{i}'] = layer.bias.detach().cpu().numpy()
        np.savez(path, feature_version=1, student_version=1, **data)


class StudentRepair(CachedJointRepair):
    def __init__(self, checkpoint):
        super().__init__(checkpoint, width=1)
        with np.load(checkpoint) as data:
            assert int(data['student_version']) == 1
            for key in ['rw1','rb1','rw2','rb2','rw3','rb3']:
                setattr(self, key, data[key].copy())

    def logits(self, x):
        raw = np.clip((x[..., :BASE_DIM]-self.mean)/self.scale, -10, 10)
        z = np.maximum(raw@self.w1.T+self.b1, 0)
        z = np.maximum(z@self.w2.T+self.b2, 0)
        initial = (z@self.w3.T+self.b3)[..., 0]
        z = np.concatenate((raw, np.clip(x[..., BASE_DIM:], -10, 10)), -1)
        z = np.maximum(z@self.rw1.T+self.rb1, 0)
        z = np.maximum(z@self.rw2.T+self.rb2, 0)
        return initial+(z@self.rw3.T+self.rb3)[..., 0]

    def decode(self, graph, base, packet, count_candidates, target=None):
        p = graph.problem
        # Round-robin cells retains previously chosen cross-cell activity.
        block = tuple((c, j) for j in range(p.P) for c in range(p.C))
        cache = self.cache_for(graph, base)
        b = 1 if target is not None else count_candidates
        action = np.full((b, p.C, p.P), -1, np.int64)
        count = np.zeros((b, p.U), np.int64)
        layer = np.full((b, p.U), 4, np.int64)
        active = np.zeros((b, p.C, p.P), bool)
        fingerprint = hashlib.sha256(base.actions.tobytes()+p.queue.tobytes()+p.mcs.tobytes()).digest()
        noise_seed = int.from_bytes(fingerprint[:8], 'little')
        features, masks, labels, option_ids = [], [], [], []
        for depth, (cell, prg) in enumerate(block):
            choices = options(p, cell, prg)
            # Keep the pretrained repair's local cross-cell interference
            # context; a global average hides this PRG's victim cells.
            local_block = tuple((c, prg) for c in range(p.C))
            raw = cache.features(local_block, cell, count, layer, active)
            extra = packet[np.maximum(choices, 0)].copy(); extra[choices < 0] = 0
            x = np.concatenate((raw, np.broadcast_to(extra, (*raw.shape[:2], PACKET_DIM))), -1)
            if target is not None:
                index = np.flatnonzero(choices == target[cell, prg]); assert len(index) == 1
                padded = np.zeros((MAX_OPTIONS, INPUT_DIM), np.float32); padded[:len(choices)] = x[0]
                features.append(padded); masks.append(np.arange(MAX_OPTIONS) < len(choices)); labels.append(int(index[0]))
                ids = np.full(MAX_OPTIONS, -2, np.int64); ids[:len(choices)] = choices; option_ids.append(ids)
                selected = np.array([target[cell, prg]])
            else:
                logits = self.logits(x)
                # Stateless per-prefix noise makes the 4-candidate pool a prefix
                # of the 8-candidate pool on exactly the same observation.
                rng = np.random.default_rng((noise_seed+depth) % (2**64))
                uniform = rng.uniform(1.e-6, 1-1.e-6, (7, MAX_OPTIONS))[:b, :len(choices)]
                perturb = -np.log(-np.log(uniform))*.35; perturb[0] = 0
                selected = choices[(logits+perturb).argmax(-1)]
            action[:, cell, prg] = selected
            active[:, cell, prg] = selected >= 0
            rows = np.flatnonzero(selected >= 0); users = selected[rows]
            count[rows, users] += 1
            layer[rows, users] = np.minimum(layer[rows, users], p.caps[users, prg])
        p.actions(action)
        if target is not None:
            np.testing.assert_array_equal(action[0], target)
            return dict(features=np.asarray(features, np.float16), mask=np.asarray(masks),
                target=np.asarray(labels, np.int64), option_ue=np.asarray(option_ids))
        return action

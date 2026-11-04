"""An autoregressive joint Type-0 policy with static legality masks only.

The 51 tokens run PRG-major, then cell-major. Previous selected identities
condition subsequent decisions. Teacher forcing and sampling use the same
GRU and distributions. No demand cap, service weight, or post-policy repair.
"""
import math
import numpy as np
import torch
from torch import nn

FEATURES = 21
BLANK = 36


def observation(obs, phy_inputs):
    ue = np.asarray(obs['obs_ue_features'], dtype=np.float32)
    base = np.stack((np.log1p(ue[:, 0] / 3000),
        10 * np.log10(np.maximum(ue[:, 2], 1.e-9)) / 30,
        ue[:, 3] / 15, ue[:, 4] / 4, ue[:, 5], ue[:, 6],
        ue[:, 7] / 400, ue[:, 8] / 200, ue[:, 9] / 200), -1)
    sinr = 10 * np.log10(np.maximum(obs['post_eq_sinr'], 1.e-9)) / 30
    mcs = np.asarray([phy_inputs.reconstructed_mcs_for_ue(u) for u in range(36)]) / 27
    caps = np.asarray(phy_inputs.prg_layer_cap) / 4
    slots = np.asarray(obs['teacher_action_ue_select'])
    alloc = np.asarray(obs['teacher_action_prg_alloc']).reshape(17, 3)
    x = np.zeros((51, 37, FEATURES), np.float32)
    legal = np.zeros((51, 37), bool)
    for p in range(17):
        for c in range(3):
            t = p * 3 + c
            x[t, :36, :9] = base
            x[t, :36, 9:13] = sinr[:, p]
            x[t, :36, 13] = mcs
            x[t, :36, 14] = caps[:, p]
            x[t, :, 15] = p / 16
            x[t, :, 16 + c] = 1
            if alloc[p, c] >= 0:
                x[t, int(slots[alloc[p, c]]), 19] = 1
            else:
                x[t, BLANK, 19] = 1
            x[t, BLANK, 20] = 1
            legal[t, :36] = (np.asarray(obs['action_mask_cell_ue'][c], bool)
                & np.asarray(obs['action_mask_ue'], bool) & (ue[:, 0] > 0)
                & bool(obs['action_mask_prg_cell'][c, p]))
            legal[t, BLANK] = True
    x = np.clip(x, -10, 10)
    assert np.isfinite(x).all()
    return torch.from_numpy(x), torch.from_numpy(legal)


def action_tokens(action):
    a = np.asarray(action).T.reshape(51).copy()
    a[a < 0] = BLANK
    return torch.from_numpy(a).long()


def native_action(tokens):
    a = torch.as_tensor(tokens).detach().cpu().numpy().copy().reshape(17, 3).T
    a[a == BLANK] = -1
    return a


class JointActor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.context = nn.Linear(2 * hidden, hidden)
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True)
        self.query = nn.Linear(hidden, hidden)
        self.base = nn.Linear(hidden, 1)

    def encode(self, x, mask):
        e = self.encoder(x)
        mean = (e * mask[..., None]).sum(2) / mask.sum(2, keepdim=True)
        global_mean = mean.mean(1, keepdim=True).expand_as(mean)
        context = torch.tanh(self.context(torch.cat((mean, global_mean), -1)))
        return e, context

    def distributions(self, x, mask, actions):
        e, context = self.encode(x, mask)
        selected = e.gather(2, actions[..., None, None].expand(-1, -1, 1, self.hidden)).squeeze(2)
        previous = torch.cat((torch.zeros_like(selected[:, :1]), selected[:, :-1]), 1)
        state, _ = self.gru(torch.cat((context, previous), -1))
        logits = (e * self.query(state)[:, :, None]).sum(-1) / math.sqrt(self.hidden) + self.base(e).squeeze(-1)
        return logits.masked_fill(~mask, -1.e9).log_softmax(-1)

    def replay(self, x, mask, actions):
        logps = self.distributions(x, mask, actions)
        assert bool(mask.gather(-1, actions[..., None]).all())
        logp = logps.gather(-1, actions[..., None]).squeeze(-1).sum(-1)
        entropy = -(logps.exp() * logps).sum(-1).sum(-1)
        return logp, entropy, logps

    @torch.no_grad()
    def sample(self, x, mask, *, greedy=False, generator=None):
        x, mask = x[None], mask[None]
        e, context = self.encode(x, mask)
        hidden = None
        previous = torch.zeros((1, 1, self.hidden), dtype=x.dtype, device=x.device)
        choices, rows = [], []
        for t in range(51):
            state, hidden = self.gru(torch.cat((context[:, t:t+1], previous), -1), hidden)
            logits = (e[:, t] * self.query(state[:, 0])[:, None]).sum(-1) / math.sqrt(self.hidden) + self.base(e[:, t]).squeeze(-1)
            logps = logits.masked_fill(~mask[:, t], -1.e9).log_softmax(-1)
            choice = logps.argmax(-1) if greedy else torch.multinomial(logps.exp(), 1, generator=generator)[:, 0]
            previous = e[:, t].gather(1, choice[:, None, None].expand(-1, 1, self.hidden))
            choices.append(choice[0]); rows.append(logps[0])
        actions, logps = torch.stack(choices), torch.stack(rows)
        return actions, logps.gather(-1, actions[:, None]).sum(), logps


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES * 2, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x, mask):
        mean = (x * mask[..., None]).sum((1, 2)) / mask.sum((1, 2))[:, None]
        maximum = x.masked_fill(~mask[..., None], -1.e9).amax((1, 2))
        return self.net(torch.cat((mean, maximum), -1)).squeeze(-1)


def gae(rewards, values, bootstrap, gamma, lam):
    advantages = torch.empty_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype)
    next_value = bootstrap
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + gamma * next_value - values[i]
        carry = delta + gamma * lam * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, advantages + values

"""Service PPO over a frozen, search-free neural proposal distribution."""
import torch
from torch import nn
from training.stageb_ppo_v2.model import ServiceActor as OriginalActor
from .features import STATE_DIM


class ServiceActor(OriginalActor):
    def __init__(self):
        super().__init__()
        self.ue[0] = nn.Linear(STATE_DIM, 64)


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(STATE_DIM*2+7), nn.Linear(STATE_DIM*2+7, 128),
            nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, summary, mask):
        opportunity = summary.masked_fill(~mask[..., None], -1.e9).amax(1)
        return self.net(torch.cat((state.mean(1), state.amax(1), opportunity), -1)).squeeze(-1)

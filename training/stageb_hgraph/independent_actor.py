from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .graph_ops import PackedPrgCandidates
from .model import HeteroMessageLayer
from .template_encoders import CellTemplateEncoder, PrgTemplateEncoder, UeTemplateEncoder
from .types import SparseEntityGraph


@dataclass(frozen=True)
class IndependentActorConfig:
    cell_feat_dim: int = 5
    ue_feat_dim: int = 16
    prg_feat_dim: int = 12
    hidden_dim: int = 128
    message_layers: int = 2
    up_edge_attr_dim: int = 5
    attention_heads: int = 4
    max_candidates: int = 12
    dropout: float = 0.0


def _fit_feature_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    target = int(target_dim)
    current = int(features.shape[-1])
    if current == target:
        return features
    if current > target:
        return features[..., :target]
    padding = torch.zeros(
        (*features.shape[:-1], target - current),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat((features, padding), dim=-1)


class PFQIndependentReuseActor(nn.Module):
    actor_family = "pfq_independent_reuse_v1"
    requires_pfq_features = False

    def __init__(self, cfg: IndependentActorConfig | None = None):
        super().__init__()
        self.cfg = cfg or IndependentActorConfig()
        hidden = int(self.cfg.hidden_dim)
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.cfg.attention_heads) != 4:
            raise ValueError("the first independent actor fixes attention_heads=4")
        if hidden % int(self.cfg.attention_heads) != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        self.cell_encoder = CellTemplateEncoder(
            in_dim=int(self.cfg.cell_feat_dim),
            hidden_dim=hidden,
            out_dim=hidden,
        )
        self.ue_encoder = UeTemplateEncoder(
            in_dim=int(self.cfg.ue_feat_dim),
            hidden_dim=max(hidden, 96),
            out_dim=hidden,
        )
        self.prg_encoder = PrgTemplateEncoder(
            in_dim=int(self.cfg.prg_feat_dim),
            hidden_dim=hidden,
            out_dim=hidden,
        )
        self.ue_tail_projection = (
            nn.Linear(max(1, int(self.cfg.ue_feat_dim) - 12), hidden)
            if int(self.cfg.ue_feat_dim) > 12
            else None
        )
        self.ue_tail_norm = nn.LayerNorm(hidden) if self.ue_tail_projection is not None else None

        self.message_layers = nn.ModuleList(
            HeteroMessageLayer(
                hidden_dim=hidden,
                cc_edge_attr_dim=2,
                up_edge_attr_dim=int(self.cfg.up_edge_attr_dim),
                dropout=float(self.cfg.dropout),
            )
            for _ in range(max(1, int(self.cfg.message_layers)))
        )
        self.reuse_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=int(self.cfg.attention_heads),
            dropout=float(self.cfg.dropout),
            batch_first=True,
        )
        self.reuse_norm = nn.LayerNorm(hidden)

        self.candidate_edge_projection = nn.Linear(
            max(1, int(self.cfg.up_edge_attr_dim)),
            hidden,
        )
        self.ue_scorer = nn.Sequential(
            nn.LayerNorm(hidden * 4 + 3),
            nn.Linear(hidden * 4 + 3, hidden),
            nn.GELU(),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(hidden, 1),
        )
        self.blank_scorer = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(float(self.cfg.dropout)),
            nn.Linear(hidden, 1),
        )

    def encode_graph(self, graph: SparseEntityGraph) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        cell_features = _fit_feature_dim(
            graph.snapshot.cell_features.values.to(device=device),
            self.cfg.cell_feat_dim,
        )
        ue_features = _fit_feature_dim(
            graph.snapshot.ue_features.values.to(device=device),
            self.cfg.ue_feat_dim,
        )
        prg_features = _fit_feature_dim(
            graph.snapshot.prg_features.values.to(device=device),
            self.cfg.prg_feat_dim,
        )

        cell_embeddings = self.cell_encoder(cell_features)
        ue_embeddings = self.ue_encoder(ue_features)
        if self.ue_tail_projection is not None and self.ue_tail_norm is not None:
            ue_embeddings = self.ue_tail_norm(
                ue_embeddings + self.ue_tail_projection(ue_features[:, 12:])
            )
        prg_embeddings = self.prg_encoder(prg_features)
        for layer in self.message_layers:
            cell_embeddings, ue_embeddings, prg_embeddings = layer(
                cell_embeddings,
                ue_embeddings,
                prg_embeddings,
                graph,
            )
        reuse_embeddings = self.apply_reuse_attention(
            prg_embeddings,
            n_cell=graph.snapshot.n_cell,
            n_prg=graph.snapshot.n_prg,
        )
        return {
            "cell_embeddings": cell_embeddings,
            "ue_embeddings": ue_embeddings,
            "prg_embeddings": prg_embeddings,
            "reuse_embeddings": reuse_embeddings,
        }

    def apply_reuse_attention(
        self,
        prg_embeddings: torch.Tensor,
        n_cell: int,
        n_prg: int,
    ) -> torch.Tensor:
        hidden = int(self.cfg.hidden_dim)
        expected = (int(n_cell) * int(n_prg), hidden)
        if tuple(prg_embeddings.shape) != expected:
            raise ValueError(
                f"prg_embeddings shape mismatch: got={tuple(prg_embeddings.shape)} expected={expected}"
            )
        by_local_prg = prg_embeddings.reshape(int(n_cell), int(n_prg), hidden).permute(1, 0, 2)
        attended, _ = self.reuse_attention(
            by_local_prg,
            by_local_prg,
            by_local_prg,
            need_weights=False,
        )
        return self.reuse_norm(by_local_prg + attended)

    def score_reuse_group(
        self,
        encoded: dict[str, torch.Tensor],
        graph: SparseEntityGraph,
        packed: PackedPrgCandidates,
        local_prg: int,
        sequential_features: torch.Tensor,
    ) -> torch.Tensor:
        meta = graph.snapshot.static_meta
        n_cell = int(meta.n_cell)
        n_prg = int(meta.n_prg)
        max_candidates = int(packed.max_candidates)
        local = int(local_prg)
        if local < 0 or local >= n_prg:
            raise ValueError(f"local_prg out of range: {local}")
        if max_candidates != int(self.cfg.max_candidates):
            raise ValueError(
                f"candidate width mismatch: got={max_candidates} expected={int(self.cfg.max_candidates)}"
            )
        if tuple(sequential_features.shape) != (n_cell, max_candidates, 3):
            raise ValueError(
                "sequential_features must be [C,K,3], "
                f"got={tuple(sequential_features.shape)}"
            )

        device = encoded["ue_embeddings"].device
        dtype = encoded["ue_embeddings"].dtype
        token_indices = torch.arange(n_cell, device=device, dtype=torch.long) * n_prg + local
        candidate_ids = packed.ue_ids.to(device=device, dtype=torch.long).index_select(0, token_indices)
        safe_ids = candidate_ids.clamp(min=0, max=max(0, int(meta.n_ue) - 1))
        candidate_ue = encoded["ue_embeddings"].index_select(0, safe_ids.reshape(-1)).reshape(
            n_cell,
            max_candidates,
            -1,
        )
        edge_attr = torch.zeros(
            (n_cell, max_candidates, max(1, int(self.cfg.up_edge_attr_dim))),
            dtype=dtype,
            device=device,
        )
        if packed.edge_attr is not None:
            raw_attr = packed.edge_attr.to(device=device, dtype=dtype).index_select(0, token_indices)
            copy_dim = min(int(edge_attr.shape[-1]), int(raw_attr.shape[-1]))
            edge_attr[..., :copy_dim] = raw_attr[..., :copy_dim]
        edge_context = self.candidate_edge_projection(edge_attr)

        cell_context = encoded["cell_embeddings"].unsqueeze(1).expand(-1, max_candidates, -1)
        reuse_context = encoded["reuse_embeddings"][local].unsqueeze(1).expand(
            -1,
            max_candidates,
            -1,
        )
        sequence_context = sequential_features.to(device=device, dtype=dtype)
        ue_logits = self.ue_scorer(
            torch.cat(
                (
                    reuse_context,
                    cell_context,
                    candidate_ue,
                    edge_context,
                    sequence_context,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        blank_logits = self.blank_scorer(
            torch.cat((encoded["reuse_embeddings"][local], encoded["cell_embeddings"]), dim=-1)
        )
        return torch.cat((ue_logits, blank_logits), dim=-1)

    def model_config_dict(self) -> dict[str, int | float | str | bool]:
        config: dict[str, int | float | str | bool] = dict(asdict(self.cfg))
        config.update(
            {
                "actor_family": self.actor_family,
                "requires_pfq_features": self.requires_pfq_features,
            }
        )
        return config


__all__ = ["IndependentActorConfig", "PFQIndependentReuseActor"]

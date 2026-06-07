"""Dual-channel (individual + group) CVR model in PyTorch."""
# update date：2026-06-07

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


def build_mlp(in_dim: int, units: List[int], final_activation: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for i, u in enumerate(units):
        layers.append(nn.Linear(prev, u))
        is_last = i == len(units) - 1
        if (not is_last) or final_activation:
            layers.append(nn.ReLU())
        prev = u
    return nn.Sequential(*layers)


class TargetAttention(nn.Module):

    def __init__(self, emb_dim: int, hidden: List[int]):
        super().__init__()
        self.score_mlp = build_mlp(emb_dim * 4, hidden + [1], final_activation=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                key_ids: torch.Tensor) -> torch.Tensor:
        length = keys.shape[1]
        q = query.unsqueeze(1).expand(-1, length, -1)
        feat = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.score_mlp(feat).squeeze(-1)  # [B, L]

        mask = (key_ids != 0)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg_inf)
        empty = ~mask.any(dim=1, keepdim=True)
        weights = torch.softmax(scores, dim=1)
        weights = weights.masked_fill(empty, 0.0).unsqueeze(-1)
        return (weights * keys).sum(dim=1)


class GroupIdFusion(nn.Module):
    """Hierarchical Group-ID prefix fusion."""

    def __init__(self, num_stages: int, centroids_per_stage: int, emb_dim: int,
                 proj_hidden: List[int]):
        super().__init__()
        self.num_stages = num_stages
        self.codebook = nn.ModuleList(
            [nn.Embedding(centroids_per_stage, emb_dim) for _ in range(num_stages)]
        )
        self.fusion = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.Tanh())
             for _ in range(num_stages - 1)]
        )
        self.proj = build_mlp(num_stages * emb_dim, proj_hidden + [emb_dim],
                              final_activation=False)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        base = [self.codebook[s](codes[:, s]) for s in range(self.num_stages)]
        fused = [base[0]]
        running = base[0]
        for s in range(1, self.num_stages):
            running = self.fusion[s - 1](torch.cat([running, base[s]], dim=-1))
            fused.append(running)
        return self.proj(torch.cat(fused, dim=-1))


class EmbeddingLayer(nn.Module):
    """Maps raw integer-id features into dense embeddings (id 0 = PAD)."""

    def __init__(self, cfg: ModelConfig, num_users: int, num_items: int, num_l1: int,
                 num_leaf: int, num_city: int, num_age: int, num_gender: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, cfg.user_emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items, cfg.item_emb_dim, padding_idx=0)
        self.l1_emb = nn.Embedding(num_l1, cfg.cat_emb_dim, padding_idx=0)
        self.leaf_emb = nn.Embedding(num_leaf, cfg.cat_emb_dim, padding_idx=0)
        self.city_emb = nn.Embedding(num_city, cfg.cat_emb_dim, padding_idx=0)
        self.age_emb = nn.Embedding(num_age, cfg.cat_emb_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(num_gender, cfg.cat_emb_dim, padding_idx=0)


class ChoirRecModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        num_users: int,
        num_items: int,
        num_l1: int,
        num_leaf: int,
        num_city: int,
        num_age: int,
        num_gender: int,
        num_stages: int,
        centroids_per_stage: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_stages = num_stages

        self.emb = EmbeddingLayer(
            cfg, num_users, num_items, num_l1, num_leaf, num_city, num_age, num_gender,
        )
        self.group_id_fusion = GroupIdFusion(
            num_stages, centroids_per_stage, cfg.group_id_emb_dim, cfg.group_id_fusion_mlp,
        )

        self.user_attention = TargetAttention(cfg.cat_emb_dim, cfg.attention_mlp)
        self.group_attention = TargetAttention(cfg.cat_emb_dim, cfg.attention_mlp)

        ind_in = (
            cfg.user_emb_dim + cfg.item_emb_dim + cfg.cat_emb_dim + cfg.cat_emb_dim
            + cfg.cat_emb_dim * 3
            + cfg.cat_emb_dim + cfg.cat_emb_dim
        )
        grp_in = (
            cfg.group_id_emb_dim + cfg.cat_emb_dim * 3
            + cfg.cat_emb_dim + cfg.cat_emb_dim
            + cfg.item_emb_dim + cfg.cat_emb_dim
        )

        # The individual tower is split so the asymmetric injection can enter
        # mid-tower: seg1 is the first layer feeding the injection, seg2 is the
        # middle block whose output the injection is added to, seg3 is the head.
        ind_units = cfg.individual_mlp
        grp_units = cfg.group_mlp
        self.ind_seg1 = build_mlp(ind_in, ind_units[:1])
        self.ind_seg2 = build_mlp(ind_units[0], ind_units[1:-1])
        self.ind_seg3 = build_mlp(ind_units[-2], ind_units[-1:])

        self.grp_seg1 = build_mlp(grp_in, grp_units[:1])
        self.grp_rest = build_mlp(grp_units[0], grp_units[1:])

        first_dim = ind_units[0] + grp_units[0]
        inject_out = ind_units[-2]
        self.inject_tower = build_mlp(
            first_dim, cfg.asymmetric_mlp + [inject_out], final_activation=False
        )

        repr_dim = ind_units[-1]
        grp_repr_dim = grp_units[-1]
        self.individual_head = nn.Linear(repr_dim, 1)
        self.group_head = nn.Linear(grp_repr_dim, 1)

        # One shared trunk feeds two heads: the soft distillation weight and the
        # adaptive-fusion weight, both conditioned on user activity.
        self.reliability_trunk = build_mlp(4, cfg.reliability_mlp)
        rel_out = cfg.reliability_mlp[-1]
        self.distill_head = nn.Linear(rel_out, 1)
        self.fusion_head = nn.Linear(rel_out, 1)

    def _individual_features(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        user = self.emb.user_emb(b["user_id"])
        item = self.emb.item_emb(b["item_id"])
        item_l1 = self.emb.l1_emb(b["item_l1"])
        item_leaf = self.emb.leaf_emb(b["item_leaf"])
        age = self.emb.age_emb(b["age_bucket"])
        gender = self.emb.gender_emb(b["gender_id"])
        city = self.emb.city_emb(b["city_id"])

        l1_keys = self.emb.l1_emb(b["user_l1_seq"])
        leaf_keys = self.emb.leaf_emb(b["user_leaf_seq"])
        l1_pool = self.user_attention(item_l1, l1_keys, b["user_l1_seq"])
        leaf_pool = self.user_attention(item_leaf, leaf_keys, b["user_leaf_seq"])
        return torch.cat(
            [user, item, item_l1, item_leaf, age, gender, city, l1_pool, leaf_pool],
            dim=-1,
        )

    def _group_features(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        group_id_vec = self.group_id_fusion(b["user_codes"])
        age = self.emb.age_emb(b["group_age_bucket"])
        gender = self.emb.gender_emb(b["group_gender_id"])
        city = self.emb.city_emb(b["group_city_id"])

        item = self.emb.item_emb(b["item_id"])
        item_l1 = self.emb.l1_emb(b["item_l1"])
        item_leaf = self.emb.leaf_emb(b["item_leaf"])

        g_l1_keys = self.emb.l1_emb(b["group_l1_seq"])
        g_leaf_keys = self.emb.leaf_emb(b["group_leaf_seq"])
        g_l1 = self.group_attention(item_l1, g_l1_keys, b["group_l1_seq"])
        g_leaf = self.group_attention(item_leaf, g_leaf_keys, b["group_leaf_seq"])
        return torch.cat(
            [group_id_vec, age, gender, city, g_l1, g_leaf, item, item_l1], dim=-1,
        )

    def _reliability(self, b: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        activity = torch.stack(
            [
                torch.log1p(b["purchase_count"]),
                torch.log1p(b["click_count"]),
                torch.log1p(b["active_days"]),
                b["is_low_activity"].float(),
            ],
            dim=-1,
        )
        h = self.reliability_trunk(activity)
        alpha_distill = torch.sigmoid(self.distill_head(h)).squeeze(-1)
        alpha_fusion = torch.sigmoid(self.fusion_head(h)).squeeze(-1)
        return alpha_distill, alpha_fusion

    def forward(self, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ind_feat = self._individual_features(b)
        grp_feat = self._group_features(b)

        ind_h1 = self.ind_seg1(ind_feat)
        grp_h1 = self.grp_seg1(grp_feat)

        ind_h2 = self.ind_seg2(ind_h1)
        grp_repr = self.grp_rest(grp_h1)

        # Asymmetric injection.
        fuse_in = torch.cat([ind_h1, grp_h1.detach()], dim=-1)
        injection = self.inject_tower(fuse_in)
        ind_h2 = ind_h2 + injection

        ind_repr = self.ind_seg3(ind_h2)

        ind_logit = self.individual_head(ind_repr).squeeze(-1)
        grp_logit = self.group_head(grp_repr).squeeze(-1)

        alpha_distill, alpha_fusion = self._reliability(b)
        final_logit = (1.0 - alpha_fusion) * ind_logit + alpha_fusion * grp_logit

        return {
            "logit": final_logit,
            "individual_logit": ind_logit,
            "group_logit": grp_logit,
            "distill_alpha": alpha_distill,
        }

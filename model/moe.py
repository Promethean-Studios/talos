"""Mixture-of-Experts feed-forward with top-k routing & load-balancing loss.

The MoE block is the "wide" counterpart to :class:`DenseFFNBlock`. It holds
``num_experts`` small SwiGLU experts and routes each token to the top-k most
compatible ones. Only ``num_experts_per_tok + num_shared_experts`` experts run
per token, which is how a ~400B model stays at ~30B FLOPs active per token.

Shared/grouped experts
----------------------
``num_shared_experts`` experts are applied to *every* token (like a small dense
head) and always count as active. ``grouped_experts`` optionally partitions the
router into expert groups so that routing is selective within groups (this
mirrors DeepSeek-V3/Wiki-MoE style grouping); for Phase 1 grouping affects the
router head count but the dispatch is still global top-k.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activations import SwiGLU
from model.ffn import FFNInterface

NEG_INF: float = -3.3895e38


def topk_router_logits(
    logits: torch.Tensor,
    top_k: int,
    jitter_noise: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute top-k routing weights/indices and the raw softmax probs.

    Args:
        logits: ``(n_tokens, num_experts)`` router logits.
        top_k: number of experts to route to per token.
        jitter_noise: if > 0, add scaled Gaussian noise to logits (training
            regularisation, switched off at inference).

    Returns:
        ``(topk_weights, topk_indices, probs)`` where weights have been
        normalized to sum to 1 over the selected experts.
    """
    if jitter_noise > 0:
        logits = logits + (torch.randn_like(logits) * jitter_noise * logits.abs())
    probs = F.softmax(logits, dim=-1)

    topk_weights, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    # Normalize selected weights to sum to 1 per token.
    norm = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    topk_weights = topk_weights / norm
    return topk_weights, topk_indices, probs


def load_balancing_loss(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    router_probs: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Auxiliary load-balancing loss (Switch/Mixtral-style).

    Encourages tokens to be spread evenly across experts. The loss is the
    product of the fraction of *routing weight* an expert receives and the
    fraction of *router probability* mass it attracts, summed over experts and
    scaled by the number of experts.

    Args:
        topk_indices: ``(n_tokens, top_k)`` selected expert ids.
        topk_weights: ``(n_tokens, top_k)`` normalized routing weights.
        router_probs: ``(n_tokens, num_experts)`` full softmax router probs.
        num_experts: total number of experts.

    Returns:
        A scalar loss tensor.
    """
    n_tokens = topk_indices.shape[0]
    onehot = F.one_hot(topk_indices, num_classes=num_experts).float()  # (N,k,E)
    weighted = (topk_weights.unsqueeze(-1) * onehot).sum(dim=1)  # (N,E)
    f_i = weighted.sum(dim=0) / max(n_tokens, 1)  # (E,) routed-weight fraction
    p_i = router_probs.mean(dim=0)  # (E,) mean dispatcher probability
    loss = num_experts * (f_i * p_i).sum()
    return loss


class MoEFFNBlock(nn.Module, FFNInterface):
    """Sparse Mixture-of-Experts feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        num_shared_experts: int = 0,
        grouped_experts: int | None = None,
        load_balance_coef: float = 0.01,
        router_jitter_noise: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts <= 0 or num_experts_per_tok <= 0:
            raise ValueError("MoE requires positive num_experts and top-k")
        if num_experts_per_tok > num_experts:
            raise ValueError("top-k cannot exceed the number of experts")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.load_balance_coef = load_balance_coef
        self.router_jitter_noise = router_jitter_noise

        # With grouped experts we give the router a separate logit per group;
        # the outgoing logit is the max over the groups a token belongs to.
        self.grouped_experts = grouped_experts
        self.router = nn.Linear(
            hidden_size, num_experts * (grouped_experts or 1), bias=False
        )
        self.experts = nn.ModuleList(
            [
                SwiGLU(hidden_size, moe_intermediate_size, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [
                    SwiGLU(hidden_size, moe_intermediate_size, dropout=dropout)
                    for _ in range(num_shared_experts)
                ]
            )
        else:
            self.shared_experts = nn.ModuleList()

    # -- routing -----------------------------------------------------------------
    def router_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        if self.grouped_experts is not None and self.grouped_experts > 1:
            # (N, G) -> max-pool over groups to a single per-expert logit.
            logits = logits.view(*logits.shape[:-1], self.grouped_experts, self.num_experts)
            logits = logits.max(dim=-2).values
        return logits

    # -- forward -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, self.hidden_size)
        n_tokens = flat.shape[0]

        logits = self.router_logits(flat)
        topk_w, topk_idx, probs = topk_router_logits(
            logits, self.num_experts_per_tok, self.router_jitter_noise
        )

        # (N, E) per-token routing weight for each expert (0 where not chosen).
        sel = F.one_hot(topk_idx, num_classes=self.num_experts)  # (N, k, E)
        per_expert_weight = (topk_w.unsqueeze(-1) * sel).sum(dim=1)  # (N, E)

        # Sparse, autograd-compatible dispatch: for each expert, gather its
        # assigned tokens, run the expert on them, and scatter-add the weighted
        # result back. `index_add` is differentiable w.r.t. the expert output
        # and the routing weights, so gradients flow to both.
        out: Optional[torch.Tensor] = None
        for expert_id in range(self.num_experts):
            w_e = per_expert_weight[:, expert_id]  # (N,)
            token_ids = torch.nonzero(w_e > 0.0, as_tuple=False).squeeze(-1)
            if token_ids.numel() == 0:
                continue
            w_e_sel = w_e.index_select(0, token_ids).unsqueeze(-1)  # (nsel, 1)
            flat_in = flat.index_select(0, token_ids)
            contrib = w_e_sel * self.experts[expert_id](flat_in)  # (nsel, H)
            if out is None:
                out = torch.zeros(n_tokens, self.hidden_size,
                                  dtype=flat.dtype, device=flat.device)
            out = out.index_add(0, token_ids, contrib)
        if out is None:
            out = torch.zeros(n_tokens, self.hidden_size,
                              dtype=flat.dtype, device=flat.device)

        # Shared experts run on every token.
        for shared in self.shared_experts:
            out = out + shared(flat)

        aux_loss = load_balancing_loss(topk_idx, topk_w, probs, self.num_experts)
        return out.view_as(x), self.load_balance_coef * aux_loss

    def __repr__(self) -> str:
        return (
            f"MoEFFNBlock(expert={self.hidden_size}→{self.experts[0].intermediate_size}, "
            f"experts={self.num_experts}, topk={self.num_experts_per_tok}, "
            f"shared={self.num_shared_experts})"
        )

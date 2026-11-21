"""Merge operators for Multi-Scale KV compression.

Includes both training-free operators (for rapid zero-shot verification and diagnostics)
and the improved trainable GatedCayleyMerger that fixes the flaws of linear Stiefel MERA:
  1. Exact constant-signal preservation: f(u, u) = u for ALL weights (no 2^{l/2} explosion).
  2. Input-dependent convex gating: can route salient tokens without sqrt(2) decay.
  3. Orthogonal detail mixing via Cayley map on (x0 - x1) (4x smaller than 2D x 2D).
  4. RG Fixed-Point weight sharing for scale levels l >= l_crit.
  5. Exact identity initialization: at training step 0, behaves as stable mean pooling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .device import safe_cayley_orthogonal


class BaseMerger(nn.Module):
    """Interface for merging pairs of equal-scale KV blocks.

    Inputs:
        k0, k1, v0, v1: Tensors of shape (B, H_kv, M, D)
        level: Scale level l >= 0 being merged into l + 1
        s0, s1: Optional salience tensors of shape (M,)
        pos0, pos1: Optional barycentric position tensors of shape (M,)

    Returns:
        k_new, v_new: (B, H_kv, M, D)
        pos_new: (M,)
        s_new: (M,)
    """

    def merge(
        self,
        k0: torch.Tensor,
        k1: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        level: int,
        pos0: torch.Tensor,
        pos1: torch.Tensor,
        s0: torch.Tensor,
        s1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MeanMerger(BaseMerger):
    """Training-free exact convex average (stable baseline, no norm explosion)."""

    def merge(self, k0, k1, v0, v1, level, pos0, pos1, s0, s1):
        k_new = 0.5 * (k0 + k1)
        v_new = 0.5 * (v0 + v1)
        pos_new = 0.5 * (pos0 + pos1)
        s_new = torch.maximum(s0, s1)
        return k_new, v_new, pos_new, s_new


class SalienceWeightedMerger(BaseMerger):
    """Training-free input-dependent convex merge weighted by token salience."""

    def __init__(self, eps: float = 1e-4, sharp: float = 1.0):
        super().__init__()
        self.eps = eps
        self.sharp = sharp

    def merge(self, k0, k1, v0, v1, level, pos0, pos1, s0, s1):
        w0 = (s0.float().clamp(min=0.0) + self.eps) ** self.sharp
        w1 = (s1.float().clamp(min=0.0) + self.eps) ** self.sharp
        alpha = (w0 / (w0 + w1)).clamp(0.05, 0.95)  # (M,)
        a_kv = alpha.view(1, 1, -1, 1).to(dtype=k0.dtype)
        k_new = a_kv * k0 + (1.0 - a_kv) * k1
        v_new = a_kv * v0 + (1.0 - a_kv) * v1
        pos_new = alpha * pos0 + (1.0 - alpha) * pos1
        s_new = torch.maximum(s0, s1)
        return k_new, v_new, pos_new, s_new


class NormMatchedMerger(BaseMerger):
    """Convex average followed by RMS norm restoration (prevents orthogonal shrinkage without explosion)."""

    def __init__(self, match_values: bool = False, eps: float = 1e-6):
        super().__init__()
        self.match_values = match_values
        self.eps = eps

    def _match(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        avg = 0.5 * (x0 + x1)
        n0 = x0.float().norm(dim=-1, keepdim=True)
        n1 = x1.float().norm(dim=-1, keepdim=True)
        n_avg = avg.float().norm(dim=-1, keepdim=True).clamp(min=self.eps)
        target = 0.5 * (n0 + n1)
        # Never inflate beyond 1.5x to stay safe if x0 and x1 nearly cancel.
        ratio = (target / n_avg).clamp(max=1.5)
        return (avg.float() * ratio).to(dtype=x0.dtype)

    def merge(self, k0, k1, v0, v1, level, pos0, pos1, s0, s1):
        k_new = self._match(k0, k1)
        v_new = self._match(v0, v1) if self.match_values else 0.5 * (v0 + v1)
        pos_new = 0.5 * (pos0 + pos1)
        s_new = torch.maximum(s0, s1)
        return k_new, v_new, pos_new, s_new


class StiefelUnscaledMerger(BaseMerger):
    """Diagnostic reproducing the original draft's strict isometry W^T = [I, I] / sqrt(2).

    Satisfies W^T W = I_d and W^T [u; u] / sqrt(2) = u, demonstrating how unscaled
    isometric coarsening causes 2^{l/2} norm growth on correlated signals.
    """

    def merge(self, k0, k1, v0, v1, level, pos0, pos1, s0, s1):
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        k_new = (k0 + k1) * inv_sqrt2
        v_new = (v0 + v1) * inv_sqrt2
        pos_new = 0.5 * (pos0 + pos1)
        s_new = torch.maximum(s0, s1)
        return k_new, v_new, pos_new, s_new


class _SubspaceGatedCayley(nn.Module):
    """Gated + Orthogonal Difference-Mode merger for a single subspace (K or V) across levels."""

    def __init__(self, num_kv_heads: int, head_dim: int, l_crit: int = 4, gate_rank: int = 32):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.l_crit = l_crit

        # Skew-symmetric generators P_d -> Q_d = Cayley(P_d) in O(D), per level up to l_crit.
        self.skew_p = nn.Parameter(torch.zeros(l_crit, num_kv_heads, head_dim, head_dim))
        # Bounded residual strength on orthogonal detail (x0 - x1), initialized at 0.
        self.detail_alpha = nn.Parameter(torch.zeros(l_crit, num_kv_heads, 1, head_dim))

        # Bottleneck input-dependent gate: [x0, x1] (2D) -> r -> D.
        # Initialized so W2 = 0, b2 = 0 => gate = sigmoid(0) = 0.5 at step 0.
        r = min(gate_rank, head_dim)
        self.gate_w1 = nn.Parameter(torch.empty(l_crit, num_kv_heads, 2 * head_dim, r))
        nn.init.normal_(self.gate_w1, std=0.02)
        self.gate_b1 = nn.Parameter(torch.zeros(l_crit, num_kv_heads, 1, r))
        self.gate_w2 = nn.Parameter(torch.zeros(l_crit, num_kv_heads, r, head_dim))
        self.gate_b2 = nn.Parameter(torch.zeros(l_crit, num_kv_heads, 1, head_dim))

        # Cached materialized orthogonal matrices for zero-overhead inference.
        self._materialized_q: Optional[torch.Tensor] = None

    def materialize(self) -> None:
        """Precompute Q = Cayley(P) for zero-overhead inference."""
        with torch.no_grad():
            self._materialized_q = safe_cayley_orthogonal(self.skew_p)

    def clear_materialized(self) -> None:
        self._materialized_q = None

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, level: int) -> torch.Tensor:
        # RG Fixed-Point weight sharing for l >= l_crit.
        idx = min(level, self.l_crit - 1)
        dtype = x0.dtype
        x0_f = x0.float()
        x1_f = x1.float()

        # Input-dependent convex gate g in (0, 1)^D.
        cat_x = torch.cat((x0_f, x1_f), dim=-1)  # (B, H_kv, M, 2D)
        h = torch.nn.functional.silu(
            torch.einsum("bhmi,hir->bhmr", cat_x, self.gate_w1[idx].float()) + self.gate_b1[idx].float()
        )
        g_logits = torch.einsum("bhmr,hrd->bhmd", h, self.gate_w2[idx].float()) + self.gate_b2[idx].float()
        g = torch.sigmoid(g_logits)

        # Exact convex combination: preserves constant signals identically (g*u + (1-g)*u = u).
        base = g * x0_f + (1.0 - g) * x1_f

        # Orthogonal detail injection on (x0 - x1): vanishes identically when x0 == x1.
        if self._materialized_q is not None and not self.training:
            q_orth = self._materialized_q[idx].to(device=x0.device, dtype=torch.float32)
        else:
            q_orth = safe_cayley_orthogonal(self.skew_p[idx])

        diff = 0.5 * (x0_f - x1_f)
        detail = torch.einsum("bhmd,hde->bhme", diff, q_orth)
        alpha = 0.5 * torch.tanh(self.detail_alpha[idx].float())
        out = base + alpha * detail
        return out.to(dtype=dtype)


class GatedCayleyMerger(BaseMerger):
    """Separate K and V Gated-Cayley mergers with RG Fixed-Point weight sharing."""

    def __init__(self, num_kv_heads: int, head_dim: int, l_crit: int = 4, gate_rank: int = 32):
        super().__init__()
        self.k_merger = _SubspaceGatedCayley(num_kv_heads, head_dim, l_crit=l_crit, gate_rank=gate_rank)
        self.v_merger = _SubspaceGatedCayley(num_kv_heads, head_dim, l_crit=l_crit, gate_rank=gate_rank)

    def materialize(self) -> None:
        self.k_merger.materialize()
        self.v_merger.materialize()

    def clear_materialized(self) -> None:
        self.k_merger.clear_materialized()
        self.v_merger.clear_materialized()

    def merge(self, k0, k1, v0, v1, level, pos0, pos1, s0, s1):
        k_new = self.k_merger(k0, k1, level)
        v_new = self.v_merger(v0, v1, level)
        pos_new = 0.5 * (pos0 + pos1)
        s_new = torch.maximum(s0, s1)
        return k_new, v_new, pos_new, s_new


class ModelMergers(nn.Module):
    """Container holding one GatedCayleyMerger per transformer layer."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        l_crit: int = 4,
        gate_rank: int = 32,
    ):
        super().__init__()
        self.config_meta = {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "l_crit": l_crit,
            "gate_rank": gate_rank,
        }
        self.layers = nn.ModuleList(
            [
                GatedCayleyMerger(num_kv_heads, head_dim, l_crit=l_crit, gate_rank=gate_rank)
                for _ in range(num_layers)
            ]
        )

    def materialize_all(self) -> None:
        for m in self.layers:
            m.materialize()

    def clear_materialized_all(self) -> None:
        for m in self.layers:
            m.clear_materialized()

    def save(self, path: str) -> None:
        torch.save({"meta": self.config_meta, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "ModelMergers":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        obj = cls(**ckpt["meta"])
        obj.load_state_dict(ckpt["state_dict"])
        return obj


def build_merger(merger_mode: str) -> BaseMerger:
    mode = merger_mode.lower().strip()
    if mode == "mean":
        return MeanMerger()
    if mode == "salience_weighted":
        return SalienceWeightedMerger()
    if mode == "norm_matched":
        return NormMatchedMerger(match_values=False)
    if mode == "norm_matched_kv":
        return NormMatchedMerger(match_values=True)
    if mode == "stiefel_unscaled":
        return StiefelUnscaledMerger()
    raise ValueError(
        f"Unknown training-free merger_mode '{merger_mode}'. "
        "Use one of: mean, salience_weighted, norm_matched, norm_matched_kv, stiefel_unscaled "
        "(or pass trained ModelMergers for learned mode)."
    )

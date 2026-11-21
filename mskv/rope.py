"""Rotary Position Embedding (RoPE) utilities including analytic multi-scale damping.

Key mathematical result implemented here:
For a contiguous block of L tokens at positions [m_0, ..., m_0 + L - 1] with constant
pre-RoPE key k_0, the exact mean of rotated keys equals:

    (1/L) * sum_{m} R_{theta, m} k_0  =  gamma_dir(L) * R_{theta, m_bar} k_0

where:
    m_bar          = m_0 + (L - 1) / 2   (barycentric coordinate)
    gamma_dir_j(L) = sin(L * theta_j / 2) / (L * sin(theta_j / 2))

Naive barycentric RoPE sets gamma_j = 1 for all frequencies, injecting high-frequency
phase noise into coarse blocks (where L * theta_j >> 2*pi).
Smooth Gaussian damping:
    gamma_gauss_j(L) = exp(-(L^2 - 1) * theta_j^2 / 24)
matches gamma_dir_j(L) to second order O(theta^2), stays strictly positive, and avoids
sign-flipping artifacts while attenuating unresolved high-frequency RoPE channels.
"""

from __future__ import annotations

from typing import Tuple

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard HuggingFace Llama / Qwen half-split rotation."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def extract_rope_params(model, head_dim: int) -> Tuple[torch.Tensor, float]:
    """Extract (inv_freq, attention_scaling) from a HuggingFace CausalLM."""
    base_model = getattr(model, "model", model)

    rotary_emb = getattr(base_model, "rotary_emb", None)
    if rotary_emb is None:
        layers = getattr(base_model, "layers", None)
        if layers is not None and len(layers) > 0:
            rotary_emb = getattr(layers[0].self_attn, "rotary_emb", None)

    if rotary_emb is not None and hasattr(rotary_emb, "inv_freq"):
        inv_freq = rotary_emb.inv_freq.detach().float().clone()
        attn_scaling = float(getattr(rotary_emb, "attention_scaling", 1.0))
        return inv_freq, attn_scaling

    config = getattr(model, "config", None)
    rope_theta = float(getattr(config, "rope_theta", 10000.0)) if config is not None else 10000.0
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    return inv_freq, 1.0


def compute_rope_damping(
    counts: torch.Tensor,
    inv_freq: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Compute per-dimension RoPE damping factor gamma for blocks of size ``counts``.

    Args:
        counts: Tensor of shape (S,) or (B, S) containing block token counts L >= 1.
        inv_freq: Tensor of shape (D/2,).
        mode: One of ``pre_rope_barycentric``, ``pre_rope_damped_gauss``, ``pre_rope_damped_dirichlet``.

    Returns:
        Tensor of shape (B_or_1, 1, S, D) in float32.
    """
    if counts.ndim == 1:
        counts = counts.unsqueeze(0)  # (1, S)
    theta = inv_freq.to(device=counts.device, dtype=torch.float32)  # (D/2,)
    L = counts.float().unsqueeze(-1)  # (B, S, 1)

    if mode == "pre_rope_barycentric":
        gamma_half = torch.ones((L.shape[0], L.shape[1], theta.shape[0]), device=L.device, dtype=torch.float32)
    elif mode == "pre_rope_damped_gauss":
        # Exact second-order match to the Dirichlet kernel using Var(Uniform(0..L-1)) = (L^2 - 1) / 12.
        gamma_half = torch.exp(-((L * L - 1.0) * (theta * theta)) / 24.0)
    elif mode == "pre_rope_damped_dirichlet":
        half_theta = 0.5 * theta
        denom = L * torch.sin(half_theta)
        num = torch.sin(L * half_theta)
        gamma_half = torch.where(denom.abs() < 1e-7, torch.ones_like(num), num / denom)
    else:
        raise ValueError(f"Unknown pre-RoPE mode: {mode}")

    gamma = torch.cat((gamma_half, gamma_half), dim=-1)  # (B, S, D)
    return gamma.unsqueeze(1)  # (B, 1, S, D)


def apply_rope_1d(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float = 1.0,
) -> torch.Tensor:
    """Apply standard RoPE at arbitrary (possibly fractional) positions.

    Args:
        x: Tensor of shape (B, H, S, D).
        positions: Tensor of shape (S,) or (B, S).
        inv_freq: Tensor of shape (D/2,).
        attention_scaling: RoPE magnitude factor (1.0 for standard RoPE, >1 for some YaRN configs).
    """
    if positions.ndim == 1:
        pos = positions.unsqueeze(0)
    else:
        pos = positions
    theta = inv_freq.to(device=x.device, dtype=torch.float32)
    freqs = pos.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * theta  # (B_or_1, S, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B_or_1, S, D)
    cos = (emb.cos() * attention_scaling).unsqueeze(1).to(dtype=x.dtype)
    sin = (emb.sin() * attention_scaling).unsqueeze(1).to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def apply_multiscale_rope(
    k: torch.Tensor,
    positions: torch.Tensor,
    counts: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float = 1.0,
    mode: str = "pre_rope_damped_gauss",
) -> torch.Tensor:
    """Apply barycentric RoPE with optional multi-scale high-frequency damping."""
    if mode == "post_rope":
        return k
    k_rot = apply_rope_1d(k, positions, inv_freq, attention_scaling=attention_scaling)
    if mode == "pre_rope_barycentric":
        return k_rot
    gamma = compute_rope_damping(counts, inv_freq, mode=mode).to(dtype=k_rot.dtype)
    return k_rot * gamma

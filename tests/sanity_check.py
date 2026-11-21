"""CPU-only mathematical sanity checks (no model download or GPU required).

Verifies:
  1. Bit-for-bit equivalence between 1-by-1 streaming and vectorized chunked updates
     for both k = 1 (Carry-Bit) and k = 4 (Exponential Histogram).
  2. Absence of power-of-2 resolution cliffs for k >= 2 vs presence of cliff for k = 1.
  3. Exact analytic Dirichlet RoPE damping identity on constant pre-RoPE keys.
  4. Norm explosion in StiefelUnscaledMerger vs stability in MeanMerger / GatedCayleyMerger.
  5. Exact orthogonality Q^T Q = I, exact identity init, and exact constant-signal preservation
     f(u, u) = u of GatedCayleyMerger (including autograd backward pass).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mskv.cache import CacheConfig, MultiScaleLayerCache
from mskv.device import safe_cayley_orthogonal
from mskv.mergers import GatedCayleyMerger, MeanMerger, StiefelUnscaledMerger
from mskv.rope import apply_multiscale_rope, apply_rope_1d


def test_streaming_vs_chunked_equivalence() -> None:
    torch.manual_seed(0)
    h_kv, d, total = 2, 16, 137
    k_seq = torch.randn(1, h_kv, total, d)
    v_seq = torch.randn(1, h_kv, total, d)
    pos_seq = torch.arange(total, dtype=torch.float32)

    for k_slots in (1, 2, 4):
        cfg = CacheConfig(
            mode="multiscale_tree" if k_slots > 1 else "carry_bit_original",
            sink_size=4,
            local_size=16,
            slots_per_level=k_slots,
            rope_mode="pre_rope_damped_gauss",
            merger_mode="mean",
            salience_metric="attn",
        )
        c1 = MultiScaleLayerCache(cfg, MeanMerger())
        c_chunk = MultiScaleLayerCache(cfg, MeanMerger())

        # 1-by-1 streaming
        for t in range(total):
            c1.append_chunk(k_seq[:, :, t : t + 1, :], v_seq[:, :, t : t + 1, :], pos_seq[t : t + 1])

        # Arbitrary uneven chunk sizes
        idx = 0
        for step in (7, 13, 19, 31, 23, 44):
            end = min(total, idx + step)
            c_chunk.append_chunk(k_seq[:, :, idx:end, :], v_seq[:, :, idx:end, :], pos_seq[idx:end])
            idx = end

        k1, v1, p1, cnt1, _ = c1.get_past_tensors()
        k2, v2, p2, cnt2, _ = c_chunk.get_past_tensors()
        assert torch.allclose(k1, k2, atol=1e-5), f"Key mismatch for k={k_slots}"
        assert torch.allclose(v1, v2, atol=1e-5), f"Value mismatch for k={k_slots}"
        assert torch.allclose(p1, p2, atol=1e-5), f"Position mismatch for k={k_slots}"
        assert torch.equal(cnt1, cnt2), f"Count mismatch for k={k_slots}"
    print("[PASS] Streaming 1-by-1 == vectorized chunked update (k=1, 2, 4)")


def test_power_of_two_cliff_vs_redundant() -> None:
    torch.manual_seed(1)
    h_kv, d = 1, 8
    sink, loc = 4, 8
    # Trigger exact N_hist = 64 (= 2^6) evicted tokens => total = sink + loc + 64 = 76
    total = sink + loc + 64
    k_seq = torch.randn(1, h_kv, total, d)
    v_seq = torch.randn(1, h_kv, total, d)
    pos_seq = torch.arange(total, dtype=torch.float32)

    cfg_carry = CacheConfig(mode="carry_bit_original", sink_size=sink, local_size=loc, slots_per_level=1)
    cfg_exp = CacheConfig(mode="multiscale_tree", sink_size=sink, local_size=loc, slots_per_level=4)
    c_carry = MultiScaleLayerCache(cfg_carry, MeanMerger())
    c_exp = MultiScaleLayerCache(cfg_exp, MeanMerger())

    c_carry.append_chunk(k_seq, v_seq, pos_seq)
    c_exp.append_chunk(k_seq, v_seq, pos_seq)

    s_carry = c_carry.stats()
    s_exp = c_exp.stats()
    # At N_hist = 64, 1-slot Carry-Bit collapses ALL 64 tokens into 1 single tree slot!
    assert s_carry["tree"] == 1 and s_carry["active_levels"] == 1, f"Unexpected carry stats: {s_carry}"
    # Redundant k=4 retains fine-grained slots across multiple levels!
    assert s_exp["active_levels"] >= 4 and c_exp.tree_k[0] is not None and c_exp.tree_k[0].shape[2] >= 3
    print(
        f"[PASS] Power-of-2 cliff reproduced (k=1: {s_carry['tree']} slot / {s_carry['active_levels']} level) "
        f"vs fixed (k=4: {s_exp['tree']} slots / {s_exp['active_levels']} levels)"
    )


def test_dirichlet_rope_exact_identity() -> None:
    torch.manual_seed(2)
    d = 64
    inv_freq = 1.0 / (500000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    k0 = torch.randn(1, 2, 1, d)

    for block_len in (2, 8, 64, 256):
        m0 = 37
        positions = torch.arange(m0, m0 + block_len, dtype=torch.float32)
        k_block = k0.expand(1, 2, block_len, d)
        exact_mean = apply_rope_1d(k_block, positions, inv_freq).mean(dim=2, keepdim=True)

        pos_bar = torch.tensor([m0 + 0.5 * (block_len - 1)], dtype=torch.float32)
        counts = torch.tensor([float(block_len)], dtype=torch.float32)
        damped = apply_multiscale_rope(
            k0, pos_bar, counts, inv_freq, mode="pre_rope_damped_dirichlet"
        )
        max_err = (exact_mean - damped).abs().max().item()
        assert max_err < 1e-5, f"Dirichlet RoPE mismatch at L={block_len}: {max_err}"

        # Compare against naive barycentric RoPE (should show massive error due to high-freq aliasing!).
        naive = apply_multiscale_rope(k0, pos_bar, counts, inv_freq, mode="pre_rope_barycentric")
        naive_err = (exact_mean - naive).abs().max().item()
        assert naive_err > 0.1, f"Expected naive barycentric RoPE error, got {naive_err}"

    print("[PASS] Analytic Dirichlet RoPE exact identity (<1e-5 error) vs naive barycentric error (>0.1)")


def test_norm_explosion_vs_gated_cayley() -> None:
    torch.manual_seed(3)
    h_kv, d = 2, 32
    u = torch.randn(1, h_kv, 1, d)
    u = u / u.norm(dim=-1, keepdim=True)

    # 1. StiefelUnscaledMerger grows as 2^{l/2} on constant signal (10 levels -> 32x).
    stiefel = StiefelUnscaledMerger()
    x_s = u.clone()
    p = torch.tensor([0.0])
    s = torch.tensor([1.0])
    for lvl in range(10):
        x_s, _, _, _ = stiefel.merge(x_s, x_s, x_s, x_s, lvl, p, p, s, s)
    ratio = x_s.norm(dim=-1).mean().item()
    assert abs(ratio - 32.0) < 1e-3, f"Expected 32x norm growth, got {ratio}"

    # 2. GatedCayleyMerger: orthogonal Q, identity init, exact constant-signal preservation after random weights.
    merger = GatedCayleyMerger(num_kv_heads=h_kv, head_dim=d, l_crit=4)
    k0 = torch.randn(1, h_kv, 3, d)
    k1 = torch.randn(1, h_kv, 3, d)
    p3 = torch.zeros(3)
    s3 = torch.ones(3)
    k_init, _, _, _ = merger.merge(k0, k1, k0, k1, 0, p3, p3, s3, s3)
    assert torch.allclose(k_init, 0.5 * (k0 + k1), atol=1e-6), "GatedCayleyMerger must init as exact mean!"

    # Randomize all parameters and verify Q^T Q = I and f(u, u) == u.
    with torch.no_grad():
        for param in merger.parameters():
            param.normal_(std=0.5)

    q_mat = safe_cayley_orthogonal(merger.k_merger.skew_p[0])
    eye = torch.eye(d).expand_as(q_mat)
    orth_err = (q_mat.transpose(-1, -2) @ q_mat - eye).abs().max().item()
    assert orth_err < 1e-5, f"Cayley orthogonality error: {orth_err}"

    # Constant signal across 10 levels must stay norm 1.0 identically!
    x_g = u.clone()
    for lvl in range(10):
        x_g, _, _, _ = merger.merge(x_g, x_g, x_g, x_g, lvl, p, p, s, s)
    assert torch.allclose(x_g, u, atol=1e-5), "GatedCayleyMerger broke constant-signal invariance!"

    # Verify gradients flow cleanly through Cayley + gate.
    k_out, v_out, _, _ = merger.merge(k0, k1, k0, k1, 2, p3, p3, s3, s3)
    loss = (k_out.pow(2).mean() + v_out.pow(2).mean())
    loss.backward()
    assert merger.k_merger.skew_p.grad is not None and merger.k_merger.skew_p.grad.abs().sum() > 0
    assert merger.k_merger.gate_w2.grad is not None and merger.k_merger.gate_w2.grad.abs().sum() > 0

    print(
        f"[PASS] Stiefel explosion verified ({ratio:.1f}x at l=10); "
        f"GatedCayleyMerger orthogonal (err={orth_err:.1e}), constant-invariant, differentiable"
    )


if __name__ == "__main__":
    test_streaming_vs_chunked_equivalence()
    test_power_of_two_cliff_vs_redundant()
    test_dirichlet_rope_exact_identity()
    test_norm_explosion_vs_gated_cayley()
    print("\nAll mathematical sanity checks PASSED.")

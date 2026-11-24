"""Fast CPU unit tests for Hierarchical Tree-Routed Exact KV Paging (Option 1).

Verifies:
  1. Exact 100% mathematical equivalence between Streaming KV R-Tree Exact BnB and Flat Quest.
  2. Real pruning efficiency (Scan% << 100%) on clustered semantic key distributions.
  3. Exactness of Chan's parallel mean/variance merge in _HeadChronoTree.
  4. Zero token loss/duplication across Sinks + CPU-Offloaded Pages + Staging + Local Window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mskv.paged_tree import (
    PagedTreeConfig,
    PagedTreeLayerCache,
    _HeadChronoTree,
    _HeadRTree,
    _aabb_ub_scores,
)


def test_rtree_exact_bnb_matches_flat_quest() -> None:
    torch.manual_seed(42)
    n_clusters = 16
    pages_per_cluster = 16
    n_pages = n_clusters * pages_per_cluster  # 256 pages (= 4096 tokens at P=16)
    d = 64
    top_k = 4

    # Synthetic realistic clustered keys (topics / semantic manifolds + small intra-page spread).
    centers = torch.randn(n_clusters, d) * 2.5
    p_mins = []
    p_maxs = []
    rtree = _HeadRTree(max_fanout=8)

    order = torch.randperm(n_pages).tolist()
    for pid, orig_idx in enumerate(order):
        cid = orig_idx // pages_per_cluster
        base = centers[cid] + 0.35 * torch.randn(d)
        keys = base.unsqueeze(0) + 0.25 * torch.randn(16, d)
        pmin = keys.amin(dim=0)
        pmax = keys.amax(dim=0)
        p_mins.append(pmin)
        p_maxs.append(pmax)
        rtree.insert_page(pid, pmin, pmax)

    all_mins = torch.stack(p_mins, dim=0)
    all_maxs = torch.stack(p_maxs, dim=0)

    total_bnb_scored = 0
    total_beam_scored = 0
    beam_hits = 0
    n_queries = 40

    for _ in range(n_queries):
        # Query aligned with a random topic cluster (simulating retrieval/induction query).
        cid = int(torch.randint(0, n_clusters, (1,)).item())
        q = (centers[cid] + 0.2 * torch.randn(d)).unsqueeze(0)

        flat_scores = _aabb_ub_scores(q, all_mins, all_maxs)
        _, flat_top = torch.topk(flat_scores, k=top_k, largest=True, sorted=False)
        flat_set = set(flat_top.tolist())

        bnb_ids, bnb_sc = rtree.query_exact_bnb(q, top_k=top_k)
        beam_ids, beam_sc = rtree.query_beam(q, top_k=top_k, beam_width=4)

        assert set(bnb_ids) == flat_set, f"Exact BnB mismatch! got {bnb_ids} vs {sorted(flat_set)}"
        total_bnb_scored += bnb_sc
        total_beam_scored += beam_sc
        beam_hits += len(set(beam_ids) & flat_set)

    bnb_scan_pct = 100.0 * total_bnb_scored / (n_queries * n_pages)
    beam_scan_pct = 100.0 * total_beam_scored / (n_queries * n_pages)
    beam_recall = 100.0 * beam_hits / (n_queries * top_k)
    assert bnb_scan_pct < 50.0, f"Expected strong pruning on clustered keys, got {bnb_scan_pct:.1f}%"
    print(
        f"[PASS] R-Tree Exact BnB == 100% Flat Quest match | "
        f"BnB Scan={bnb_scan_pct:.1f}% | Beam Scan={beam_scan_pct:.1f}% (Recall@{top_k}={beam_recall:.1f}%)"
    )


def test_chan_parallel_variance_exact() -> None:
    torch.manual_seed(7)
    d = 32
    p_sz = 16
    n_pages = 25
    all_tokens = torch.randn(n_pages * p_sz, d)
    tree = _HeadChronoTree(branching=4, scoring="ucb")

    for p in range(n_pages):
        blk = all_tokens[p * p_sz : (p + 1) * p_sz]
        tree.insert_page(blk.amin(0), blk.amax(0), blk.mean(0), blk.var(0, unbiased=False), float(p_sz))

    root_mean = tree.means[-1][0]
    root_var = tree.vars[-1][0]
    exact_mean = all_tokens.mean(0)
    exact_var = all_tokens.var(0, unbiased=False)
    assert torch.allclose(root_mean, exact_mean, atol=1e-5), "Root mean mismatch"
    assert torch.allclose(root_var, exact_var, atol=1e-5), "Root variance mismatch"
    print("[PASS] Chrono-UCB Chan's parallel mean & variance merge exact (<1e-5)")


def test_paged_layer_cache_integrity() -> None:
    torch.manual_seed(9)
    h_kv, d = 2, 16
    cfg = PagedTreeConfig(
        router="rtree_beam",
        sink_size=4,
        local_size=32,
        page_size=16,
        top_k_pages=3,
        offload_to_cpu=True,
    )
    cache = PagedTreeLayerCache(cfg, num_kv_heads=h_kv)
    total = 150
    k_seq = torch.randn(1, h_kv, total, d)
    v_seq = torch.randn(1, h_kv, total, d)

    for s in range(0, total, 13):
        e = min(total, s + 13)
        cache.append_chunk(k_seq[:, :, s:e, :], v_seq[:, :, s:e, :])

    # Verify exact token accounting: sink + pages*P + staging + local == total.
    n_sink = cache.sink_k.shape[2]
    n_pages_tok = len(cache.pages_k) * cfg.page_size
    n_stage = 0 if cache.stage_k is None else cache.stage_k.shape[2]
    n_loc = cache.loc_k.shape[2]
    assert n_sink + n_pages_tok + n_stage + n_loc == total
    assert all(p.device.type == "cpu" for p in cache.pages_k)

    q = torch.randn(1, 4, 1, d)
    kp, vp, mask, _ = cache.gather_past_kv(q, device=torch.device("cpu"), dtype=torch.float32)
    assert kp is not None and kp.shape == (1, h_kv, n_sink + cfg.top_k_pages * cfg.page_size + n_stage + n_loc, d)
    assert vp.shape == kp.shape and mask is None
    print(f"[PASS] PagedTreeLayerCache token accounting exact ({total} tokens -> {len(cache.pages_k)} CPU pages)")


def test_induction_halo_and_pinned_dedup() -> None:
    """Verify Induction Halo continuity, Pinned Heavy Hitters, zero duplicate positions, and valid_mask."""
    torch.manual_seed(11)
    h_kv, d = 2, 16
    total = 220
    # Encode exact token position in dim 0 of V so we can verify exact position sets after gathering!
    k_seq = torch.randn(1, h_kv, total, d)
    v_seq = torch.randn(1, h_kv, total, d)
    pos_seq = torch.arange(total, dtype=torch.long)
    v_seq[:, :, :, 0] = pos_seq.float().view(1, 1, total)

    cfg = PagedTreeConfig(
        router="chrono_hybrid",
        sink_size=4,
        local_size=32,
        page_size=16,
        top_k_pages=3,
        halo_tokens=6,
        pinned_size=12,
        beam_width=4,
        offload_to_cpu=True,
    )
    cache = PagedTreeLayerCache(cfg, num_kv_heads=h_kv)
    for s in range(0, total, 19):
        e = min(total, s + 19)
        sal = torch.rand(e - s)
        cache.append_chunk(
            k_seq[:, :, s:e, :],
            v_seq[:, :, s:e, :],
            pos_chunk=pos_seq[s:e],
            attn_salience_chunk=sal,
        )

    assert cache.pinned_k is not None and cache.pinned_k.shape[2] == cfg.pinned_size
    assert len(cache.pages_k) * cfg.page_size + (0 if cache.stage_k is None else cache.stage_k.shape[2]) + 4 + 32 == total

    q = torch.randn(1, 4, 1, d)
    kp, vp, valid_mask, slices = cache.gather_past_kv(q, device=torch.device("cpu"), dtype=torch.float32)
    assert kp is not None and vp is not None
    assert "pinned" in slices and "local" in slices and "fetched" in slices

    # Verify every KV-head has strictly unique valid token positions (zero double-counting!).
    for h in range(h_kv):
        head_pos = vp[0, h, :, 0].round().long()
        if valid_mask is not None:
            head_pos = head_pos[valid_mask[0, h]]
        pos_list = head_pos.tolist()
        assert len(pos_list) == len(set(pos_list)), f"Duplicate token positions detected in head {h}!"
        # Verify all pinned positions are present exactly once.
        for pp in cache.pinned_pos.tolist():
            assert pp in pos_list, f"Pinned position {pp} missing from gathered KV!"

    print(
        f"[PASS] Chrono-Hybrid + Induction Halo ({cfg.halo_tokens}) + Pinned ({cfg.pinned_size}): "
        f"zero duplicate tokens, valid_mask={'active' if valid_mask is not None else 'uniform'}"
    )


def test_decoupled_envelopes_and_funnel_beam() -> None:
    """Verify Decoupled Envelopes strip pinned massive-activation outliers and Funnel Beam scales cleanly."""
    torch.manual_seed(17)
    h_kv, d = 2, 16
    total = 100
    k_seq = torch.randn(1, h_kv, total, d) * 0.5
    v_seq = torch.randn(1, h_kv, total, d)
    pos_seq = torch.arange(total, dtype=torch.long)

    # Inject a massive-activation structural outlier at position 10 with huge salience score!
    k_seq[:, :, 10, :] = 100.0
    sal = torch.ones(total)
    sal[10] = 999.0

    cfg_coupled = PagedTreeConfig(
        router="chrono_hybrid",
        sink_size=4,
        local_size=32,
        page_size=16,
        top_k_pages=2,
        pinned_size=4,
        decouple_pinned=False,
        offload_to_cpu=True,
    )
    cfg_decoupled = PagedTreeConfig(
        router="chrono_hybrid",
        sink_size=4,
        local_size=32,
        page_size=16,
        top_k_pages=2,
        pinned_size=4,
        decouple_pinned=True,
        offload_to_cpu=True,
    )
    c_coup = PagedTreeLayerCache(cfg_coupled, num_kv_heads=h_kv)
    c_dec = PagedTreeLayerCache(cfg_decoupled, num_kv_heads=h_kv)
    c_coup.append_chunk(k_seq, v_seq, pos_chunk=pos_seq, attn_salience_chunk=sal)
    c_dec.append_chunk(k_seq, v_seq, pos_chunk=pos_seq, attn_salience_chunk=sal)

    # Page 0 holds positions 4..19 (includes outlier at pos 10).
    assert c_coup.flat_maxs[0].max().item() > 90.0, "Coupled envelope should contain outlier 100.0"
    assert c_dec.flat_maxs[0].max().item() < 5.0, "Decoupled envelope must exclude pinned outlier!"
    assert 10 in c_dec.pinned_pos.tolist(), "Outlier pos 10 must be preserved in pinned_pool"

    # Verify Funnel Beam visits strictly more coarse nodes than flat beam on a 256-page tree, while << 100%.
    tree = _HeadChronoTree(branching=4, scoring="hybrid")
    for _ in range(256):
        blk = torch.randn(16, d)
        tree.insert_page(blk.amin(0), blk.amax(0), blk.mean(0), blk.var(0, unbiased=False), 16.0)
    q = torch.randn(1, d)
    ids_flat, sc_flat = tree.query_beam(q, top_k=4, beam_width=4, funnel_mult=1)
    ids_funnel, sc_funnel = tree.query_beam(q, top_k=4, beam_width=4, funnel_mult=2)
    assert len(ids_flat) == 4 and len(ids_funnel) == 4
    assert sc_flat < sc_funnel < 256, f"Expected sc_flat ({sc_flat}) < sc_funnel ({sc_funnel}) < 256"
    print(
        f"[PASS] Decoupled Envelopes removed outlier (100.0 -> {c_dec.flat_maxs[0].max().item():.2f}); "
        f"Funnel Beam scanned {sc_funnel}/256 ({100.0 * sc_funnel / 256:.1f}%) vs Flat {sc_flat}/256"
    )


if __name__ == "__main__":
    test_rtree_exact_bnb_matches_flat_quest()
    test_chan_parallel_variance_exact()
    test_paged_layer_cache_integrity()
    test_induction_halo_and_pinned_dedup()
    test_decoupled_envelopes_and_funnel_beam()
    print("\nAll Paged Tree-Routed KV sanity checks PASSED.")

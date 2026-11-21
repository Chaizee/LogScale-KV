"""Multi-Scale Streaming KV-Cache with Redundant Exponential-Histogram Tree & Salient Pool.

Implements all architectures needed for apples-to-apples ablation:
  - ``full``:               Exact full KV-cache (upper bound).
  - ``streaming_llm``:      Attention Sinks + Local Window only (pure eviction).
  - ``h2o_lite``:           Sinks + Salient Heavy-Hitter Pool + Local Window (no tree).
  - ``carry_bit_original``: Sinks + 1-slot-per-level Carry-Bit Tree + Local Window.
  - ``multiscale_tree``:    Sinks + k-slot Redundant Exponential-Histogram Tree + Local Window.
  - ``hybrid_ms``:          Sinks + Salient Pool + k-slot Multi-Scale Tree + Local Window.

Core algorithmic fix (Redundant Dyadic Buffer):
  Instead of 1 slot per level (which collapses ALL fine-grained levels on every power of 2),
  each scale level l holds up to k blocks of size 2^l. Whenever level l has n > k blocks,
  the oldest 2*m blocks (where m = (n - k + 1) // 2) are merged pairwise into m blocks at
  level l + 1, leaving r in {k - 1, k} blocks at level l.
  - For k >= 2: ZERO power-of-2 resolution cliffs; strict chronological order; O(1) amortized.
  - For k = 1:  Mathematically identical to binary Carry-Bit.
  - Vectorized chunk update is bit-for-bit identical to 1-by-1 streaming AND differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .mergers import BaseMerger, build_merger


@dataclass
class CacheConfig:
    mode: str = "hybrid_ms"
    sink_size: int = 4
    local_size: int = 128
    salient_size: int = 64
    slots_per_level: int = 4  # k >= 2 eliminates power-of-2 cliffs; k = 1 is original Carry-Bit
    max_levels: int = 20
    rope_mode: str = "pre_rope_damped_gauss"  # post_rope | pre_rope_barycentric | pre_rope_damped_gauss | pre_rope_damped_dirichlet
    merger_mode: str = "mean"  # mean | salience_weighted | norm_matched | stiefel_unscaled | learned
    use_log_bias: bool = True
    salience_metric: str = "hybrid"  # attn | key_outlier | hybrid
    outlier_weight: float = 0.25  # weight of query-free key outlier score in hybrid salience

    def resolved_slots_per_level(self) -> int:
        if self.mode == "carry_bit_original":
            return 1
        return max(1, int(self.slots_per_level))

    def uses_tree(self) -> bool:
        return self.mode in ("carry_bit_original", "multiscale_tree", "hybrid_ms")

    def uses_salient_pool(self) -> bool:
        return self.mode in ("h2o_lite", "hybrid_ms") and self.salient_size > 0


class MultiScaleLayerCache:
    """Per-layer streaming KV cache state (supports batch_size = 1)."""

    def __init__(self, config: CacheConfig, merger: BaseMerger):
        self.cfg = config
        self.merger = merger
        self.reset()

    def reset(self) -> None:
        self.total_tokens_seen: int = 0
        self.total_merges: int = 0

        # Attention Sinks: (1, H_kv, S_sink, D)
        self.sink_k: Optional[torch.Tensor] = None
        self.sink_v: Optional[torch.Tensor] = None
        self.sink_pos: Optional[torch.Tensor] = None

        # Salient / Heavy-Hitter Pool: (1, H_kv, S_sal, D)
        self.sal_k: Optional[torch.Tensor] = None
        self.sal_v: Optional[torch.Tensor] = None
        self.sal_pos: Optional[torch.Tensor] = None
        self.sal_score: Optional[torch.Tensor] = None

        # Local Sliding Window (or full history in mode == 'full'): (1, H_kv, S_loc, D)
        self.loc_k: Optional[torch.Tensor] = None
        self.loc_v: Optional[torch.Tensor] = None
        self.loc_pos: Optional[torch.Tensor] = None
        self.loc_score: Optional[torch.Tensor] = None

        # Multi-Scale Exponential-Histogram Tree levels:
        # level l stores blocks of exact size 2^l tokens.
        self.tree_k: List[Optional[torch.Tensor]] = [None] * self.cfg.max_levels
        self.tree_v: List[Optional[torch.Tensor]] = [None] * self.cfg.max_levels
        self.tree_pos: List[Optional[torch.Tensor]] = [None] * self.cfg.max_levels
        self.tree_score: List[Optional[torch.Tensor]] = [None] * self.cfg.max_levels

        # Running mean key for query-free outlier salience detection (1, H_kv, 1, D).
        self._running_k_mean: Optional[torch.Tensor] = None
        self._running_k_count: int = 0

    def _compute_key_outlier_score(self, k_new: torch.Tensor) -> torch.Tensor:
        """Query-free anomaly score measuring L2 deviation from running mean key."""
        with torch.no_grad():
            k_f = k_new.detach().float()  # (1, H_kv, T, D)
            t = k_f.shape[2]
            if self._running_k_mean is None:
                mean_k = k_f.mean(dim=2, keepdim=True)
                self._running_k_mean = mean_k
                self._running_k_count = t
            else:
                mean_k = self._running_k_mean
                # Update running mean safely.
                new_total = self._running_k_count + t
                self._running_k_mean = (
                    mean_k * (self._running_k_count / new_total)
                    + k_f.mean(dim=2, keepdim=True) * (t / new_total)
                )
                self._running_k_count = new_total

            diff = (k_f - mean_k).norm(dim=-1)  # (1, H_kv, T)
            base_norm = mean_k.norm(dim=-1).clamp(min=1e-4)  # (1, H_kv, 1)
            rel_dev = (diff / base_norm).mean(dim=1)[0]  # (T,)
            return rel_dev

    def _push_to_tree(
        self,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        pos_in: torch.Tensor,
        score_in: torch.Tensor,
    ) -> None:
        """Insert chronologically ordered size-1 tokens into the k-slot Exponential-Histogram.

        Vectorized update rule at level l:
          If n > k: m = (n - k + 1) // 2 oldest pairs merge to level l + 1;
                    remaining r = n - 2*m in {k - 1, k} stay at level l.
        Bit-for-bit identical to 1-by-1 streaming for any chunk size and any k >= 1.
        """
        k_cap = self.cfg.resolved_slots_per_level()
        cur_k, cur_v, cur_pos, cur_score = k_in, v_in, pos_in, score_in

        for lvl in range(self.cfg.max_levels):
            if self.tree_k[lvl] is not None:
                cur_k = torch.cat((self.tree_k[lvl], cur_k), dim=2)
                cur_v = torch.cat((self.tree_v[lvl], cur_v), dim=2)
                cur_pos = torch.cat((self.tree_pos[lvl], cur_pos), dim=0)
                cur_score = torch.cat((self.tree_score[lvl], cur_score), dim=0)

            n = cur_k.shape[2]
            if n <= k_cap or lvl == self.cfg.max_levels - 1:
                self.tree_k[lvl] = cur_k
                self.tree_v[lvl] = cur_v
                self.tree_pos[lvl] = cur_pos
                self.tree_score[lvl] = cur_score
                break

            m = (n - k_cap + 1) // 2
            split = 2 * m
            self.total_merges += m

            # Remaining newest r blocks stay at level lvl.
            if split < n:
                self.tree_k[lvl] = cur_k[:, :, split:, :]
                self.tree_v[lvl] = cur_v[:, :, split:, :]
                self.tree_pos[lvl] = cur_pos[split:]
                self.tree_score[lvl] = cur_score[split:]
            else:
                self.tree_k[lvl] = None
                self.tree_v[lvl] = None
                self.tree_pos[lvl] = None
                self.tree_score[lvl] = None

            # Pairwise merge oldest 2*m blocks -> m blocks for level lvl + 1.
            k0 = cur_k[:, :, 0:split:2, :]
            k1 = cur_k[:, :, 1:split:2, :]
            v0 = cur_v[:, :, 0:split:2, :]
            v1 = cur_v[:, :, 1:split:2, :]
            p0 = cur_pos[0:split:2]
            p1 = cur_pos[1:split:2]
            s0 = cur_score[0:split:2]
            s1 = cur_score[1:split:2]

            cur_k, cur_v, cur_pos, cur_score = self.merger.merge(
                k0, k1, v0, v1, lvl, p0, p1, s0, s1
            )

    def _route_evicted(
        self,
        k_ev: torch.Tensor,
        v_ev: torch.Tensor,
        pos_ev: torch.Tensor,
        score_ev: torch.Tensor,
    ) -> None:
        """Route tokens evicted from the local window according to cache mode."""
        if self.cfg.mode == "streaming_llm":
            return

        if not self.cfg.uses_salient_pool():
            if self.cfg.uses_tree():
                self._push_to_tree(k_ev, v_ev, pos_ev, score_ev)
            return

        # Combine existing salient pool with newly evicted candidates and keep top-P.
        # Tokens from k_ev that do NOT enter salient_pool are pushed chronologically to the tree.
        cap = self.cfg.salient_size
        n_ev = k_ev.shape[2]
        n_old = 0 if self.sal_k is None else self.sal_k.shape[2]

        if n_old == 0:
            comb_k, comb_v, comb_pos, comb_score = k_ev, v_ev, pos_ev, score_ev
        else:
            comb_k = torch.cat((self.sal_k, k_ev), dim=2)
            comb_v = torch.cat((self.sal_v, v_ev), dim=2)
            comb_pos = torch.cat((self.sal_pos, pos_ev), dim=0)
            comb_score = torch.cat((self.sal_score, score_ev), dim=0)

        total = n_old + n_ev
        if total <= cap:
            self.sal_k, self.sal_v, self.sal_pos, self.sal_score = (
                comb_k,
                comb_v,
                comb_pos,
                comb_score,
            )
            return

        # Select top-cap indices by salience score; sort selected indices chronologically.
        _, top_idx = torch.topk(comb_score, k=cap, largest=True, sorted=False)
        selected_mask = torch.zeros(total, dtype=torch.bool, device=comb_score.device)
        selected_mask[top_idx] = True

        # Keep selected tokens in chronological order inside salient_pool.
        sel_idx = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        self.sal_k = comb_k[:, :, sel_idx, :]
        self.sal_v = comb_v[:, :, sel_idx, :]
        self.sal_pos = comb_pos[sel_idx]
        self.sal_score = comb_score[sel_idx]

        # Newly evicted tokens NOT chosen for salient_pool flow chronologically into the tree.
        if self.cfg.uses_tree():
            ev_not_selected = ~selected_mask[n_old:]
            if ev_not_selected.any():
                tree_idx = torch.nonzero(ev_not_selected, as_tuple=False).squeeze(-1)
                self._push_to_tree(
                    k_ev[:, :, tree_idx, :],
                    v_ev[:, :, tree_idx, :],
                    pos_ev[tree_idx],
                    score_ev[tree_idx],
                )

    def get_past_tensors(
        self,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, Tuple[int, int]],
    ]:
        """Assemble cached [Sinks | Salient | Tree (coarse->fine) | Local] tensors.

        Returns:
            k_past:      (1, H_kv, M_past, D) or None
            v_past:      (1, H_kv, M_past, D) or None
            pos_past:    (M_past,) float32 or None
            count_past:  (M_past,) float32 (exact token count per slot, 1 for uncompressed, 2^l for tree)
            slices:      dict mapping region name -> (start_idx, end_idx) for salience feedback
        """
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        ps: List[torch.Tensor] = []
        cs: List[torch.Tensor] = []
        slices: Dict[str, Tuple[int, int]] = {}
        offset = 0

        if self.sink_k is not None:
            n = self.sink_k.shape[2]
            ks.append(self.sink_k)
            vs.append(self.sink_v)
            ps.append(self.sink_pos)
            cs.append(torch.ones(n, device=self.sink_k.device, dtype=torch.float32))
            slices["sink"] = (offset, offset + n)
            offset += n

        if self.sal_k is not None:
            n = self.sal_k.shape[2]
            ks.append(self.sal_k)
            vs.append(self.sal_v)
            ps.append(self.sal_pos)
            cs.append(torch.ones(n, device=self.sal_k.device, dtype=torch.float32))
            slices["salient"] = (offset, offset + n)
            offset += n

        # Tree levels from oldest/coarsest (high l) to newest/finest (l = 0) to stay chronological.
        tree_start = offset
        for lvl in range(self.cfg.max_levels - 1, -1, -1):
            tk = self.tree_k[lvl]
            if tk is not None:
                n = tk.shape[2]
                ks.append(tk)
                vs.append(self.tree_v[lvl])
                ps.append(self.tree_pos[lvl])
                cs.append(
                    torch.full((n,), float(1 << lvl), device=tk.device, dtype=torch.float32)
                )
                offset += n
        if offset > tree_start:
            slices["tree"] = (tree_start, offset)

        if self.loc_k is not None:
            n = self.loc_k.shape[2]
            ks.append(self.loc_k)
            vs.append(self.loc_v)
            ps.append(self.loc_pos)
            cs.append(torch.ones(n, device=self.loc_k.device, dtype=torch.float32))
            slices["local"] = (offset, offset + n)
            offset += n

        if not ks:
            return None, None, None, None, slices

        return (
            torch.cat(ks, dim=2),
            torch.cat(vs, dim=2),
            torch.cat(ps, dim=0),
            torch.cat(cs, dim=0),
            slices,
        )

    def apply_salience_feedback(
        self,
        salience_past: Optional[torch.Tensor],
        slices: Dict[str, Tuple[int, int]],
    ) -> None:
        """Accumulate observed attention scores onto salient_pool and local_window entries."""
        if salience_past is None or self.cfg.salience_metric == "key_outlier":
            return
        with torch.no_grad():
            if "salient" in slices and self.sal_score is not None:
                s0, s1 = slices["salient"]
                self.sal_score = self.sal_score + salience_past[s0:s1]
            if "local" in slices and self.loc_score is not None:
                s0, s1 = slices["local"]
                self.loc_score = self.loc_score + salience_past[s0:s1]

    def append_chunk(
        self,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        pos_chunk: torch.Tensor,
        attn_salience_chunk: Optional[torch.Tensor] = None,
    ) -> None:
        """Append a newly processed chunk of tokens into sinks -> local -> evicted routing."""
        if k_chunk.shape[0] != 1:
            raise ValueError(f"Streaming prototype expects batch_size=1, got {k_chunk.shape[0]}.")

        t = k_chunk.shape[2]
        device = k_chunk.device
        pos_f = pos_chunk.to(device=device, dtype=torch.float32)

        # Build initial salience score for incoming chunk tokens.
        if self.cfg.salience_metric == "attn":
            score = (
                attn_salience_chunk.detach().float()
                if attn_salience_chunk is not None
                else torch.zeros(t, device=device, dtype=torch.float32)
            )
        elif self.cfg.salience_metric == "key_outlier":
            score = self._compute_key_outlier_score(k_chunk)
        else:  # hybrid
            attn_s = (
                attn_salience_chunk.detach().float()
                if attn_salience_chunk is not None
                else torch.zeros(t, device=device, dtype=torch.float32)
            )
            out_s = self._compute_key_outlier_score(k_chunk)
            score = attn_s + self.cfg.outlier_weight * out_s

        self.total_tokens_seen += t

        # Full KV-cache mode: never evict or compress.
        if self.cfg.mode == "full":
            if self.loc_k is None:
                self.loc_k, self.loc_v, self.loc_pos, self.loc_score = k_chunk, v_chunk, pos_f, score
            else:
                self.loc_k = torch.cat((self.loc_k, k_chunk), dim=2)
                self.loc_v = torch.cat((self.loc_v, v_chunk), dim=2)
                self.loc_pos = torch.cat((self.loc_pos, pos_f), dim=0)
                self.loc_score = torch.cat((self.loc_score, score), dim=0)
            return

        # 1. Fill Attention Sinks first.
        cur_k, cur_v, cur_pos, cur_score = k_chunk, v_chunk, pos_f, score
        sink_cur = 0 if self.sink_k is None else self.sink_k.shape[2]
        if sink_cur < self.cfg.sink_size:
            need = min(self.cfg.sink_size - sink_cur, cur_k.shape[2])
            if self.sink_k is None:
                self.sink_k = cur_k[:, :, :need, :]
                self.sink_v = cur_v[:, :, :need, :]
                self.sink_pos = cur_pos[:need]
            else:
                self.sink_k = torch.cat((self.sink_k, cur_k[:, :, :need, :]), dim=2)
                self.sink_v = torch.cat((self.sink_v, cur_v[:, :, :need, :]), dim=2)
                self.sink_pos = torch.cat((self.sink_pos, cur_pos[:need]), dim=0)
            if need == cur_k.shape[2]:
                return
            cur_k = cur_k[:, :, need:, :]
            cur_v = cur_v[:, :, need:, :]
            cur_pos = cur_pos[need:]
            cur_score = cur_score[need:]

        # 2. Append remainder to Local Sliding Window.
        if self.loc_k is None:
            self.loc_k, self.loc_v, self.loc_pos, self.loc_score = cur_k, cur_v, cur_pos, cur_score
        else:
            self.loc_k = torch.cat((self.loc_k, cur_k), dim=2)
            self.loc_v = torch.cat((self.loc_v, cur_v), dim=2)
            self.loc_pos = torch.cat((self.loc_pos, cur_pos), dim=0)
            self.loc_score = torch.cat((self.loc_score, cur_score), dim=0)

        # 3. Evict overflow from Local Sliding Window into Salient Pool & Multi-Scale Tree.
        overflow = self.loc_k.shape[2] - self.cfg.local_size
        if overflow > 0:
            ev_k = self.loc_k[:, :, :overflow, :]
            ev_v = self.loc_v[:, :, :overflow, :]
            ev_pos = self.loc_pos[:overflow]
            ev_score = self.loc_score[:overflow]

            self.loc_k = self.loc_k[:, :, overflow:, :]
            self.loc_v = self.loc_v[:, :, overflow:, :]
            self.loc_pos = self.loc_pos[overflow:]
            self.loc_score = self.loc_score[overflow:]

            self._route_evicted(ev_k, ev_v, ev_pos, ev_score)

    def stats(self) -> Dict[str, int]:
        n_sink = 0 if self.sink_k is None else self.sink_k.shape[2]
        n_sal = 0 if self.sal_k is None else self.sal_k.shape[2]
        n_loc = 0 if self.loc_k is None else self.loc_k.shape[2]
        n_tree = sum(0 if tk is None else tk.shape[2] for tk in self.tree_k)
        active_levels = sum(1 for tk in self.tree_k if tk is not None)
        return {
            "tokens_seen": self.total_tokens_seen,
            "cached_entries": n_sink + n_sal + n_tree + n_loc,
            "sink": n_sink,
            "salient": n_sal,
            "tree": n_tree,
            "local": n_loc,
            "active_levels": active_levels,
            "merges": self.total_merges,
        }


class MultiScaleKVCache:
    """Multi-layer cache controller attached to a patched transformer model."""

    def __init__(
        self,
        num_layers: int,
        config: CacheConfig,
        layer_mergers: Optional[List[BaseMerger]] = None,
    ):
        self.num_layers = num_layers
        self.cfg = config
        if layer_mergers is not None:
            if len(layer_mergers) != num_layers:
                raise ValueError(f"Expected {num_layers} layer mergers, got {len(layer_mergers)}.")
            mergers = layer_mergers
        else:
            shared = build_merger(config.merger_mode)
            mergers = [shared for _ in range(num_layers)]
        self.layers = [MultiScaleLayerCache(config, mergers[i]) for i in range(num_layers)]

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def summary(self) -> Dict[str, float]:
        s0 = self.layers[0].stats()
        seen = max(1, s0["tokens_seen"])
        entries = max(1, s0["cached_entries"])
        return {
            **s0,
            "compression_ratio": seen / entries,
        }

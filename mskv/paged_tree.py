"""Hierarchical Tree-Routed Exact KV Paging (Option 1: CPU-Offloaded Pages + O(log N) Tree Index).

Instead of replacing exact KV tokens with lossy merged vectors, exact KV pages of size P
are stored in CPU RAM (pinned/host memory), while a lightweight hierarchical envelope tree
routes each query to the top-K relevant pages and fetches their exact uncompressed (K, V).

Implements four routing indexes for direct scientific comparison:
  1. ``flat_quest``:  SOTA flat scan of all N/P page AABBs (Quest, ICML 2024 baseline).
  2. ``chrono_bvh``:  Chronological b-ary AABB tree with top-down beam search.
  3. ``chrono_ucb``:  Chronological b-ary Moment-UCB tree (Chan's exact parallel mean/var
                      + Extreme Value Theory bound mu^T q + beta * sqrt(2 ln L) * std).
  4. ``rtree_beam``:  Online Streaming KV R-Tree (minimum AABB enlargement insertion +
                      spatial split) with fixed-budget O(log N) top-down beam search.
  5. ``rtree_bnb``:   Exact admissible Best-First Branch-and-Bound over the Streaming KV
                      R-Tree (mathematically guaranteed 100% identical top-K to flat_quest).
"""

from __future__ import annotations

import contextlib
import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from .patching import _detect_attn_return_arity, repeat_kv
from .rope import apply_rope_1d, extract_rope_params


@dataclass
class PagedTreeConfig:
    router: str = "rtree_beam"  # full | streaming_llm | flat_quest | chrono_bvh | chrono_ucb | rtree_beam | rtree_bnb
    sink_size: int = 4
    local_size: int = 128
    page_size: int = 16
    top_k_pages: int = 4  # fetched exact pages per KV-head (e.g., 4*16=64 -> total ~196 slots = H2O-Lite!)
    beam_width: int = 4  # top-down beam width for tree beam search
    branching: int = 4  # branching factor for chrono trees
    rtree_max_fanout: int = 8  # max children per Streaming KV R-Tree node
    ucb_beta: float = 1.0  # exploration factor for chrono_ucb EVT bound
    offload_to_cpu: bool = True  # keep exact KV page pool in host RAM (ideal for 8GB VRAM + 16GB RAM)


def _aabb_ub_scores(
    q_flat: torch.Tensor,
    box_mins: torch.Tensor,
    box_maxs: torch.Tensor,
) -> torch.Tensor:
    """Exact admissible upper bound on max_{q in q_flat, k in [min, max]} q^T k.

    Args:
        q_flat:   (Q, D) float32 on CPU
        box_mins: (M, D) float32 on CPU
        box_maxs: (M, D) float32 on CPU
    Returns:
        (M,) float32 upper bound per candidate box.
    """
    center = 0.5 * (box_maxs + box_mins)
    radius = 0.5 * (box_maxs - box_mins)
    ub = torch.matmul(q_flat, center.t()) + torch.matmul(q_flat.abs(), radius.t())
    return ub.amax(dim=0)


def _ucb_scores(
    q_flat: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    counts: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Extreme-Value-Theory Moment-UCB score for subtrees of size ``counts``.

    UCB(q, B) = q^T mu_B + beta * sqrt(2 * ln(L_B)) * sqrt((q^2)^T var_B + eps)
    """
    mean_term = torch.matmul(q_flat, means.t())  # (Q, M)
    var_term = torch.matmul(q_flat * q_flat, variances.clamp(min=0.0).t())  # (Q, M)
    evt_scale = torch.sqrt(2.0 * torch.log(counts.clamp(min=2.0))).unsqueeze(0)  # (1, M)
    ub = mean_term + beta * evt_scale * torch.sqrt(var_term + 1e-8)
    return ub.amax(dim=0)


class _HeadChronoTree:
    """Per-KV-head chronological b-ary tree supporting AABB (BVH) and Moment-UCB routing."""

    def __init__(self, branching: int = 4, scoring: str = "aabb", ucb_beta: float = 1.0):
        self.b = max(2, branching)
        self.scoring = scoring
        self.ucb_beta = ucb_beta
        # Each level l stores lists of tensors (kept compact in CPU float32).
        self.mins: List[List[torch.Tensor]] = [[]]
        self.maxs: List[List[torch.Tensor]] = [[]]
        self.means: List[List[torch.Tensor]] = [[]]
        self.vars: List[List[torch.Tensor]] = [[]]
        self.counts: List[List[float]] = [[]]

    def num_pages(self) -> int:
        return len(self.mins[0])

    def insert_page(
        self,
        p_min: torch.Tensor,
        p_max: torch.Tensor,
        p_mean: torch.Tensor,
        p_var: torch.Tensor,
        p_count: float,
    ) -> None:
        self.mins[0].append(p_min)
        self.maxs[0].append(p_max)
        self.means[0].append(p_mean)
        self.vars[0].append(p_var)
        self.counts[0].append(p_count)

        idx = len(self.mins[0]) - 1
        lvl = 0
        while len(self.mins[lvl]) > 1:
            p_idx = idx // self.b
            next_lvl = lvl + 1
            if next_lvl == len(self.mins):
                self.mins.append([])
                self.maxs.append([])
                self.means.append([])
                self.vars.append([])
                self.counts.append([])

            start = p_idx * self.b
            end = min(len(self.mins[lvl]), start + self.b)

            c_mins = torch.stack(self.mins[lvl][start:end], dim=0)
            c_maxs = torch.stack(self.maxs[lvl][start:end], dim=0)
            new_min = c_mins.amin(dim=0)
            new_max = c_maxs.amax(dim=0)

            # Exact Chan's parallel variance merge across children.
            c_means = torch.stack(self.means[lvl][start:end], dim=0)
            c_vars = torch.stack(self.vars[lvl][start:end], dim=0)
            c_cnts = torch.tensor(self.counts[lvl][start:end], dtype=torch.float32)
            tot_cnt = float(c_cnts.sum().item())
            w = (c_cnts / max(tot_cnt, 1.0)).unsqueeze(-1)
            new_mean = (w * c_means).sum(dim=0)
            new_var = (w * (c_vars + (c_means - new_mean).pow(2))).sum(dim=0)

            if p_idx < len(self.mins[next_lvl]):
                self.mins[next_lvl][p_idx] = new_min
                self.maxs[next_lvl][p_idx] = new_max
                self.means[next_lvl][p_idx] = new_mean
                self.vars[next_lvl][p_idx] = new_var
                self.counts[next_lvl][p_idx] = tot_cnt
            else:
                self.mins[next_lvl].append(new_min)
                self.maxs[next_lvl].append(new_max)
                self.means[next_lvl].append(new_mean)
                self.vars[next_lvl].append(new_var)
                self.counts[next_lvl].append(tot_cnt)

            idx = p_idx
            lvl = next_lvl

    def _score_nodes(self, q_flat: torch.Tensor, lvl: int, indices: List[int]) -> torch.Tensor:
        if self.scoring == "aabb":
            b_min = torch.stack([self.mins[lvl][i] for i in indices], dim=0)
            b_max = torch.stack([self.maxs[lvl][i] for i in indices], dim=0)
            return _aabb_ub_scores(q_flat, b_min, b_max)
        means = torch.stack([self.means[lvl][i] for i in indices], dim=0)
        variances = torch.stack([self.vars[lvl][i] for i in indices], dim=0)
        cnts = torch.tensor([self.counts[lvl][i] for i in indices], dtype=torch.float32)
        return _ucb_scores(q_flat, means, variances, cnts, beta=self.ucb_beta)

    def query_beam(self, q_flat: torch.Tensor, top_k: int, beam_width: int) -> Tuple[List[int], int]:
        n_pages = self.num_pages()
        if n_pages <= top_k:
            return list(range(n_pages)), n_pages

        # At most one rightmost subtree at level 1 is partial (>= 1 page); all others have b pages.
        min_safe_beam = math.ceil(max(0, top_k - 1) / self.b) + 1
        eff_beam = max(beam_width, min_safe_beam, 1)

        top_lvl = len(self.mins) - 1
        active = list(range(len(self.mins[top_lvl])))
        nodes_scored = 0

        for lvl in range(top_lvl, 0, -1):
            child_lvl = lvl - 1
            n_children_total = len(self.mins[child_lvl])
            candidates: List[int] = []
            for p_idx in active:
                start = p_idx * self.b
                end = min(n_children_total, start + self.b)
                candidates.extend(range(start, end))

            if not candidates:
                break

            limit = top_k if child_lvl == 0 else eff_beam
            if len(candidates) <= limit:
                active = candidates
            else:
                scores = self._score_nodes(q_flat, child_lvl, candidates)
                nodes_scored += len(candidates)
                _, top_local = torch.topk(scores, k=limit, largest=True, sorted=False)
                active = [candidates[int(i)] for i in top_local.tolist()]

        active.sort()
        return active[:top_k], nodes_scored


class _RTreeNode:
    """Node in the Online Streaming KV R-Tree."""

    __slots__ = ("is_leaf", "box_min", "box_max", "page_ids", "children", "child_mins", "child_maxs")

    def __init__(self, is_leaf: bool):
        self.is_leaf: bool = is_leaf
        self.box_min: Optional[torch.Tensor] = None
        self.box_max: Optional[torch.Tensor] = None
        # For leaf: page_ids holds int IDs; for internal: children holds _RTreeNode objects.
        self.page_ids: List[int] = []
        self.children: List["_RTreeNode"] = []
        self.child_mins: List[torch.Tensor] = []
        self.child_maxs: List[torch.Tensor] = []

    def count(self) -> int:
        return len(self.page_ids) if self.is_leaf else len(self.children)

    def recompute_box(self) -> None:
        if not self.child_mins:
            self.box_min = None
            self.box_max = None
            return
        self.box_min = torch.stack(self.child_mins, dim=0).amin(dim=0)
        self.box_max = torch.stack(self.child_maxs, dim=0).amax(dim=0)


class _HeadRTree:
    """Per-KV-head Online Streaming KV R-Tree (minimum AABB enlargement + spatial splitting).

    Maintains tight, mathematically admissible post-RoPE bounding boxes across semantic clusters
    of pages so both fixed-budget beam search (O(log N)) and exact Best-First Branch-and-Bound
    (100% lossless vs Flat Quest) prune irrelevant subtrees cleanly.
    """

    def __init__(self, max_fanout: int = 8):
        self.max_fanout = max(4, max_fanout)
        self.min_fill = max(2, self.max_fanout // 3)
        self.root = _RTreeNode(is_leaf=True)
        self.total_pages = 0

    @staticmethod
    def _enlargement(
        c_min: torch.Tensor,
        c_max: torch.Tensor,
        p_min: torch.Tensor,
        p_max: torch.Tensor,
    ) -> Tuple[float, float]:
        old_span = float((c_max - c_min).sum().item())
        new_span = float((torch.maximum(c_max, p_max) - torch.minimum(c_min, p_min)).sum().item())
        return new_span - old_span, new_span

    def _split_node(self, node: _RTreeNode) -> Tuple[_RTreeNode, _RTreeNode]:
        n = node.count()
        mins_t = torch.stack(node.child_mins, dim=0)  # (n, D)
        maxs_t = torch.stack(node.child_maxs, dim=0)  # (n, D)
        centers = 0.5 * (mins_t + maxs_t)  # (n, D)

        # Pick two most distant seeds by L1 distance between box centers (O(F^2), F <= 9).
        best_dist = -1.0
        seed_a, seed_b = 0, 1
        for i in range(n):
            dists = (centers[i + 1 :] - centers[i]).abs().sum(dim=-1)
            if dists.numel() > 0:
                val, rel_j = torch.max(dists, dim=0)
                if float(val.item()) > best_dist:
                    best_dist = float(val.item())
                    seed_a = i
                    seed_b = i + 1 + int(rel_j.item())

        left = _RTreeNode(is_leaf=node.is_leaf)
        right = _RTreeNode(is_leaf=node.is_leaf)

        def _add(target: _RTreeNode, idx: int) -> None:
            if node.is_leaf:
                target.page_ids.append(node.page_ids[idx])
            else:
                target.children.append(node.children[idx])
            cmin, cmax = node.child_mins[idx], node.child_maxs[idx]
            target.child_mins.append(cmin)
            target.child_maxs.append(cmax)
            if target.box_min is None:
                target.box_min = cmin.clone()
                target.box_max = cmax.clone()
            else:
                target.box_min = torch.minimum(target.box_min, cmin)
                target.box_max = torch.maximum(target.box_max, cmax)

        _add(left, seed_a)
        _add(right, seed_b)

        remaining = [i for i in range(n) if i != seed_a and i != seed_b]
        for k_rem, idx in enumerate(remaining):
            rem_after = len(remaining) - k_rem
            if left.count() + rem_after == self.min_fill:
                _add(left, idx)
                continue
            if right.count() + rem_after == self.min_fill:
                _add(right, idx)
                continue

            enl_l, span_l = self._enlargement(left.box_min, left.box_max, node.child_mins[idx], node.child_maxs[idx])
            enl_r, span_r = self._enlargement(right.box_min, right.box_max, node.child_mins[idx], node.child_maxs[idx])
            if (enl_l, span_l, left.count()) <= (enl_r, span_r, right.count()):
                _add(left, idx)
            else:
                _add(right, idx)

        return left, right

    def _insert_recursive(
        self, node: _RTreeNode, page_id: int, p_min: torch.Tensor, p_max: torch.Tensor
    ) -> Optional[Tuple[_RTreeNode, _RTreeNode]]:
        if node.is_leaf:
            node.page_ids.append(page_id)
            node.child_mins.append(p_min)
            node.child_maxs.append(p_max)
            if node.box_min is None:
                node.box_min = p_min.clone()
                node.box_max = p_max.clone()
            else:
                node.box_min = torch.minimum(node.box_min, p_min)
                node.box_max = torch.maximum(node.box_max, p_max)
            if node.count() > self.max_fanout:
                return self._split_node(node)
            return None

        # Choose child causing minimum AABB L1 enlargement (vectorized across fanout <= 8).
        c_mins = torch.stack(node.child_mins, dim=0)  # (F, D)
        c_maxs = torch.stack(node.child_maxs, dim=0)  # (F, D)
        old_spans = (c_maxs - c_mins).sum(dim=-1)
        new_spans = (torch.maximum(c_maxs, p_max) - torch.minimum(c_mins, p_min)).sum(dim=-1)
        enlargements = new_spans - old_spans
        best_idx = int(torch.argmin(enlargements + 1e-6 * new_spans).item())

        split_res = self._insert_recursive(node.children[best_idx], page_id, p_min, p_max)
        if split_res is None:
            child = node.children[best_idx]
            node.child_mins[best_idx] = child.box_min
            node.child_maxs[best_idx] = child.box_max
            node.box_min = torch.minimum(node.box_min, p_min)
            node.box_max = torch.maximum(node.box_max, p_max)
            return None

        left, right = split_res
        node.children[best_idx] = left
        node.child_mins[best_idx] = left.box_min
        node.child_maxs[best_idx] = left.box_max
        node.children.append(right)
        node.child_mins.append(right.box_min)
        node.child_maxs.append(right.box_max)
        node.recompute_box()

        if node.count() > self.max_fanout:
            return self._split_node(node)
        return None

    def insert_page(self, page_id: int, p_min: torch.Tensor, p_max: torch.Tensor) -> None:
        self.total_pages += 1
        split_res = self._insert_recursive(self.root, page_id, p_min, p_max)
        if split_res is not None:
            left, right = split_res
            new_root = _RTreeNode(is_leaf=False)
            new_root.children = [left, right]
            new_root.child_mins = [left.box_min, right.box_min]
            new_root.child_maxs = [left.box_max, right.box_max]
            new_root.recompute_box()
            self.root = new_root

    def query_beam(self, q_flat: torch.Tensor, top_k: int, beam_width: int) -> Tuple[List[int], int]:
        if self.total_pages <= top_k:
            return list(range(self.total_pages)), self.total_pages

        # Every leaf cluster holds >= min_fill pages, so keeping >= ceil(top_k / min_fill)
        # leaf clusters guarantees at least top_k candidate pages for every KV-head.
        eff_beam = max(beam_width, math.ceil(top_k / self.min_fill), 1)
        active: List[_RTreeNode] = [self.root]
        nodes_scored = 0

        while active and not active[0].is_leaf:
            cand_nodes: List[_RTreeNode] = []
            cand_mins: List[torch.Tensor] = []
            cand_maxs: List[torch.Tensor] = []
            for nd in active:
                cand_nodes.extend(nd.children)
                cand_mins.extend(nd.child_mins)
                cand_maxs.extend(nd.child_maxs)

            if len(cand_nodes) <= eff_beam:
                active = cand_nodes
            else:
                scores = _aabb_ub_scores(q_flat, torch.stack(cand_mins, dim=0), torch.stack(cand_maxs, dim=0))
                nodes_scored += len(cand_nodes)
                _, top_idx = torch.topk(scores, k=eff_beam, largest=True, sorted=False)
                active = [cand_nodes[int(i)] for i in top_idx.tolist()]

        # Score leaf pages inside surviving leaf clusters.
        cand_pages: List[int] = []
        cand_mins = []
        cand_maxs = []
        for leaf in active:
            cand_pages.extend(leaf.page_ids)
            cand_mins.extend(leaf.child_mins)
            cand_maxs.extend(leaf.child_maxs)

        if len(cand_pages) <= top_k:
            selected = sorted(cand_pages)
            return selected, nodes_scored

        scores = _aabb_ub_scores(q_flat, torch.stack(cand_mins, dim=0), torch.stack(cand_maxs, dim=0))
        nodes_scored += len(cand_pages)
        _, top_idx = torch.topk(scores, k=top_k, largest=True, sorted=False)
        selected = sorted(cand_pages[int(i)] for i in top_idx.tolist())
        return selected, nodes_scored

    def query_exact_bnb(self, q_flat: torch.Tensor, top_k: int) -> Tuple[List[int], int]:
        """Exact admissible Best-First Branch-and-Bound (100% identical top-K to Flat Quest)."""
        if self.total_pages <= top_k:
            return list(range(self.total_pages)), self.total_pages

        # Max-heap via negated UB score: (-ub, tie_breaker, is_page, payload)
        heap: List[Tuple[float, int, bool, object]] = []
        counter = 0
        nodes_scored = 0

        def _expand(node: _RTreeNode) -> None:
            nonlocal counter, nodes_scored
            if not node.child_mins:
                return
            scores = _aabb_ub_scores(
                q_flat, torch.stack(node.child_mins, dim=0), torch.stack(node.child_maxs, dim=0)
            )
            nodes_scored += len(node.child_mins)
            score_list = scores.tolist()
            if node.is_leaf:
                for pid, sc in zip(node.page_ids, score_list):
                    counter += 1
                    heapq.heappush(heap, (-sc, counter, True, pid))
            else:
                for child, sc in zip(node.children, score_list):
                    counter += 1
                    heapq.heappush(heap, (-sc, counter, False, child))

        _expand(self.root)
        selected: List[int] = []
        while heap and len(selected) < top_k:
            _, _, is_page, payload = heapq.heappop(heap)
            if is_page:
                selected.append(int(payload))  # type: ignore[arg-type]
            else:
                _expand(payload)  # type: ignore[arg-type]

        selected.sort()
        return selected, nodes_scored


class PagedTreeLayerCache:
    """Per-layer exact CPU-offloaded KV page store + VRAM Sinks/Local + Hierarchical Index."""

    def __init__(self, config: PagedTreeConfig, num_kv_heads: int):
        self.cfg = config
        self.num_kv_heads = num_kv_heads
        self.reset()

    def reset(self) -> None:
        self.total_tokens_seen: int = 0
        self.total_queries: int = 0
        self.total_nodes_scored: int = 0
        self.total_pages_available_sum: int = 0
        self.last_active_slots: int = 0

        # Always GPU-resident exact regions:
        self.sink_k: Optional[torch.Tensor] = None
        self.sink_v: Optional[torch.Tensor] = None
        self.stage_k: Optional[torch.Tensor] = None  # < page_size tokens awaiting seal
        self.stage_v: Optional[torch.Tensor] = None
        self.loc_k: Optional[torch.Tensor] = None
        self.loc_v: Optional[torch.Tensor] = None

        # Exact KV Page Pool (stored on CPU RAM when offload_to_cpu=True):
        # Each entry has shape (H_kv, P, D)
        self.pages_k: List[torch.Tensor] = []
        self.pages_v: List[torch.Tensor] = []

        # Flat AABB descriptors per KV-head on CPU float32: list of (H_kv, D)
        self.flat_mins: List[torch.Tensor] = []
        self.flat_maxs: List[torch.Tensor] = []

        # Per-KV-head tree indexes:
        self.chrono_trees: Optional[List[_HeadChronoTree]] = None
        self.rtrees: Optional[List[_HeadRTree]] = None
        if self.cfg.router in ("chrono_bvh", "chrono_ucb"):
            scoring = "aabb" if self.cfg.router == "chrono_bvh" else "ucb"
            self.chrono_trees = [
                _HeadChronoTree(branching=self.cfg.branching, scoring=scoring, ucb_beta=self.cfg.ucb_beta)
                for _ in range(self.num_kv_heads)
            ]
        elif self.cfg.router in ("rtree_beam", "rtree_bnb"):
            self.rtrees = [
                _HeadRTree(max_fanout=self.cfg.rtree_max_fanout)
                for _ in range(self.num_kv_heads)
            ]

    def _seal_pages(self, k_ev: torch.Tensor, v_ev: torch.Tensor) -> None:
        """Buffer evicted tokens and seal complete exact pages of length page_size."""
        if self.cfg.router == "streaming_llm":
            return

        if self.stage_k is None:
            self.stage_k, self.stage_v = k_ev, v_ev
        else:
            self.stage_k = torch.cat((self.stage_k, k_ev), dim=2)
            self.stage_v = torch.cat((self.stage_v, v_ev), dim=2)

        p_sz = self.cfg.page_size
        n_stage = self.stage_k.shape[2]
        n_new_pages = n_stage // p_sz
        if n_new_pages == 0:
            return

        seal_len = n_new_pages * p_sz
        k_seal = self.stage_k[:, :, :seal_len, :]
        v_seal = self.stage_v[:, :, :seal_len, :]

        if seal_len < n_stage:
            self.stage_k = self.stage_k[:, :, seal_len:, :]
            self.stage_v = self.stage_v[:, :, seal_len:, :]
        else:
            self.stage_k = None
            self.stage_v = None

        # Compute page descriptors on CPU float32 and store exact pages in host RAM (if offloading).
        for p in range(n_new_pages):
            kp = k_seal[0, :, p * p_sz : (p + 1) * p_sz, :]  # (H_kv, P, D)
            vp = v_seal[0, :, p * p_sz : (p + 1) * p_sz, :]  # (H_kv, P, D)

            kp_f_cpu = kp.detach().to(device="cpu", dtype=torch.float32)
            p_min = kp_f_cpu.amin(dim=1)  # (H_kv, D)
            p_max = kp_f_cpu.amax(dim=1)  # (H_kv, D)
            p_mean = kp_f_cpu.mean(dim=1)  # (H_kv, D)
            p_var = kp_f_cpu.var(dim=1, unbiased=False)  # (H_kv, D)

            page_id = len(self.pages_k)
            if self.cfg.offload_to_cpu:
                self.pages_k.append(kp.detach().cpu())
                self.pages_v.append(vp.detach().cpu())
            else:
                self.pages_k.append(kp.detach())
                self.pages_v.append(vp.detach())

            self.flat_mins.append(p_min)
            self.flat_maxs.append(p_max)

            if self.chrono_trees is not None:
                for h in range(self.num_kv_heads):
                    self.chrono_trees[h].insert_page(
                        p_min[h], p_max[h], p_mean[h], p_var[h], float(p_sz)
                    )
            elif self.rtrees is not None:
                for h in range(self.num_kv_heads):
                    self.rtrees[h].insert_page(page_id, p_min[h], p_max[h])

    def _select_pages_per_head(self, q_rot: torch.Tensor) -> Optional[List[List[int]]]:
        """Route queries through the configured index and return selected page IDs per KV-head."""
        n_pages = len(self.pages_k)
        if n_pages == 0 or self.cfg.router in ("full", "streaming_llm"):
            return None

        top_k = min(self.cfg.top_k_pages, n_pages)
        if n_pages <= top_k:
            all_ids = list(range(n_pages))
            self.total_queries += 1
            self.total_pages_available_sum += n_pages * self.num_kv_heads
            self.total_nodes_scored += n_pages * self.num_kv_heads
            return [all_ids for _ in range(self.num_kv_heads)]

        # Move tiny query chunk to CPU float32: (1, H_q, T, D) -> (H_kv, G*T, D).
        bsz, num_heads, q_len, head_dim = q_rot.shape
        g = num_heads // self.num_kv_heads
        q_cpu = (
            q_rot[0]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(self.num_kv_heads, g * q_len, head_dim)
        )

        selected_per_head: List[List[int]] = []
        scored_this_step = 0

        if self.cfg.router == "flat_quest":
            all_mins = torch.stack(self.flat_mins, dim=0)  # (N_p, H_kv, D)
            all_maxs = torch.stack(self.flat_maxs, dim=0)  # (N_p, H_kv, D)
            for h in range(self.num_kv_heads):
                scores = _aabb_ub_scores(q_cpu[h], all_mins[:, h, :], all_maxs[:, h, :])
                _, top_idx = torch.topk(scores, k=top_k, largest=True, sorted=False)
                selected_per_head.append(sorted(top_idx.tolist()))
                scored_this_step += n_pages
        elif self.chrono_trees is not None:
            for h in range(self.num_kv_heads):
                ids, sc = self.chrono_trees[h].query_beam(q_cpu[h], top_k=top_k, beam_width=self.cfg.beam_width)
                selected_per_head.append(ids)
                scored_this_step += sc
        elif self.rtrees is not None:
            for h in range(self.num_kv_heads):
                if self.cfg.router == "rtree_bnb":
                    ids, sc = self.rtrees[h].query_exact_bnb(q_cpu[h], top_k=top_k)
                else:
                    ids, sc = self.rtrees[h].query_beam(q_cpu[h], top_k=top_k, beam_width=self.cfg.beam_width)
                selected_per_head.append(ids)
                scored_this_step += sc

        self.total_queries += 1
        self.total_nodes_scored += scored_this_step
        self.total_pages_available_sum += n_pages * self.num_kv_heads
        return selected_per_head

    def gather_past_kv(
        self, q_rot: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Assemble active exact KV context on GPU: [Sinks | Fetched Pages | Staging | Local]."""
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []

        if self.sink_k is not None:
            ks.append(self.sink_k)
            vs.append(self.sink_v)

        selected = self._select_pages_per_head(q_rot)
        if selected is not None:
            # Gather exact pages independently per KV-head and transfer to GPU in one tensor.
            head_ks = []
            head_vs = []
            for h, page_ids in enumerate(selected):
                hk = torch.cat([self.pages_k[pid][h] for pid in page_ids], dim=0)  # (K*P, D)
                hv = torch.cat([self.pages_v[pid][h] for pid in page_ids], dim=0)  # (K*P, D)
                head_ks.append(hk)
                head_vs.append(hv)
            fetched_k = torch.stack(head_ks, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
            fetched_v = torch.stack(head_vs, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
            ks.append(fetched_k)
            vs.append(fetched_v)

        if self.stage_k is not None:
            ks.append(self.stage_k)
            vs.append(self.stage_v)

        if self.loc_k is not None:
            ks.append(self.loc_k)
            vs.append(self.loc_v)

        if not ks:
            self.last_active_slots = 0
            return None, None

        k_past = torch.cat(ks, dim=2)
        v_past = torch.cat(vs, dim=2)
        self.last_active_slots = int(k_past.shape[2])
        return k_past, v_past

    def append_chunk(self, k_rot: torch.Tensor, v: torch.Tensor) -> None:
        self.total_tokens_seen += int(k_rot.shape[2])

        if self.cfg.router == "full":
            if self.loc_k is None:
                self.loc_k, self.loc_v = k_rot.detach(), v.detach()
            else:
                self.loc_k = torch.cat((self.loc_k, k_rot.detach()), dim=2)
                self.loc_v = torch.cat((self.loc_v, v.detach()), dim=2)
            return

        cur_k, cur_v = k_rot.detach(), v.detach()
        sink_cur = 0 if self.sink_k is None else self.sink_k.shape[2]
        if sink_cur < self.cfg.sink_size:
            need = min(self.cfg.sink_size - sink_cur, cur_k.shape[2])
            if self.sink_k is None:
                self.sink_k = cur_k[:, :, :need, :]
                self.sink_v = cur_v[:, :, :need, :]
            else:
                self.sink_k = torch.cat((self.sink_k, cur_k[:, :, :need, :]), dim=2)
                self.sink_v = torch.cat((self.sink_v, cur_v[:, :, :need, :]), dim=2)
            if need == cur_k.shape[2]:
                return
            cur_k = cur_k[:, :, need:, :]
            cur_v = cur_v[:, :, need:, :]

        if self.loc_k is None:
            self.loc_k, self.loc_v = cur_k, cur_v
        else:
            self.loc_k = torch.cat((self.loc_k, cur_k), dim=2)
            self.loc_v = torch.cat((self.loc_v, cur_v), dim=2)

        overflow = self.loc_k.shape[2] - self.cfg.local_size
        if overflow > 0:
            ev_k = self.loc_k[:, :, :overflow, :]
            ev_v = self.loc_v[:, :, :overflow, :]
            self.loc_k = self.loc_k[:, :, overflow:, :]
            self.loc_v = self.loc_v[:, :, overflow:, :]
            self._seal_pages(ev_k, ev_v)

    def stats(self) -> Dict[str, float]:
        n_sink = 0 if self.sink_k is None else self.sink_k.shape[2]
        n_stage = 0 if self.stage_k is None else self.stage_k.shape[2]
        n_loc = 0 if self.loc_k is None else self.loc_k.shape[2]
        n_pages = len(self.pages_k)
        fetched_cap = min(self.cfg.top_k_pages, n_pages) * self.cfg.page_size if self.cfg.router not in ("full", "streaming_llm") else 0
        active_vram_slots = n_sink + fetched_cap + n_stage + n_loc
        scan_pct = (
            100.0 * self.total_nodes_scored / max(1, self.total_pages_available_sum)
            if self.total_pages_available_sum > 0
            else 0.0
        )
        return {
            "tokens_seen": float(self.total_tokens_seen),
            "active_slots": float(active_vram_slots),
            "offloaded_pages": float(n_pages),
            "scan_pct": scan_pct,
            "compression_ratio": max(1.0, self.total_tokens_seen) / max(1.0, active_vram_slots),
        }


class PagedTreeKVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, config: PagedTreeConfig):
        self.cfg = config
        self.layers = [PagedTreeLayerCache(config, num_kv_heads=num_kv_heads) for _ in range(num_layers)]

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def summary(self) -> Dict[str, float]:
        # Average scan_pct across all layers; slot counts are uniform across layers.
        s0 = self.layers[0].stats()
        avg_scan = sum(layer.stats()["scan_pct"] for layer in self.layers) / max(1, len(self.layers))
        s0["scan_pct"] = avg_scan
        return s0


def _make_paged_forward(
    attn_module,
    layer_idx: int,
    cache: PagedTreeKVCache,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    return_arity: int,
):
    num_kv_groups = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    layer_cache = cache.layers[layer_idx]

    def paged_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = attn_module.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k = attn_module.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = attn_module.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        if hasattr(attn_module, "q_norm") and attn_module.q_norm is not None:
            q = attn_module.q_norm(q)
        if hasattr(attn_module, "k_norm") and attn_module.k_norm is not None:
            k = attn_module.k_norm(k)

        if position_ids is not None:
            chunk_pos = position_ids[0].to(device=device, dtype=torch.float32)
        elif cache_position is not None:
            chunk_pos = cache_position.to(device=device, dtype=torch.float32)
        else:
            start = layer_cache.total_tokens_seen
            chunk_pos = torch.arange(start, start + q_len, device=device, dtype=torch.float32)

        q_rot = apply_rope_1d(q, chunk_pos, inv_freq, attention_scaling=attention_scaling)
        k_rot = apply_rope_1d(k, chunk_pos, inv_freq, attention_scaling=attention_scaling)

        # Fetch exact routed past pages + GPU-resident sinks/staging/local.
        k_past, v_past = layer_cache.gather_past_kv(q_rot, device=device, dtype=dtype)
        if k_past is not None:
            m_past = k_past.shape[2]
            k_all = torch.cat((k_past, k_rot), dim=2)
            v_all = torch.cat((v_past, v), dim=2)
        else:
            m_past = 0
            k_all = k_rot
            v_all = v

        k_rep = repeat_kv(k_all, num_kv_groups)
        v_rep = repeat_kv(v_all, num_kv_groups)

        attn_logits = torch.matmul(q_rot.float(), k_rep.float().transpose(-2, -1)) * scale
        if q_len > 1:
            causal = torch.triu(torch.ones((q_len, q_len), device=device, dtype=torch.bool), diagonal=1)
            attn_logits[..., m_past:] = attn_logits[..., m_past:].masked_fill(causal, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights.to(dtype=v_rep.dtype), v_rep)

        layer_cache.append_chunk(k_rot, v)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, num_heads * head_dim)
        attn_output = attn_module.o_proj(attn_output)

        returned_weights = attn_weights if output_attentions else None
        if return_arity == 3:
            return attn_output, returned_weights, None
        return attn_output, returned_weights

    return paged_forward


@contextlib.contextmanager
def patched_paged_tree_attention(model, config: PagedTreeConfig) -> Iterator[PagedTreeKVCache]:
    base_model = getattr(model, "model", model)
    layers = base_model.layers
    cfg_m = model.config

    num_heads = int(cfg_m.num_attention_heads)
    num_kv_heads = int(getattr(cfg_m, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg_m, "head_dim", cfg_m.hidden_size // num_heads))
    inv_freq, attention_scaling = extract_rope_params(model, head_dim)
    return_arity = _detect_attn_return_arity(layers[0])

    cache = PagedTreeKVCache(num_layers=len(layers), num_kv_heads=num_kv_heads, config=config)
    orig_forwards = []
    try:
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            orig_forwards.append((attn, attn.forward))
            attn.forward = _make_paged_forward(
                attn_module=attn,
                layer_idx=i,
                cache=cache,
                inv_freq=inv_freq,
                attention_scaling=attention_scaling,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                return_arity=return_arity,
            )
        yield cache
    finally:
        for attn, orig_fwd in orig_forwards:
            attn.forward = orig_fwd

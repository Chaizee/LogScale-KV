"""Head-to-head evaluation for Hierarchical Tree-Routed Exact KV Paging (Option 1).

Compares CPU-offloaded exact KV page routing strategies on PPL, NIAH, Multi-Key (mini-RULER),
active GPU attention slots, tree scan percentage (vs 100% Flat Quest), and throughput:
  1. Full KV (Upper Bound)
  2. StreamingLLM (Pure Eviction)
  3. Flat Quest (SOTA 100% AABB page scan)
  4. Chrono-BVH Tree (Chronological 4-ary AABB tree, O(log N) beam)
  5. Chrono-UCB Tree (Chronological 4-ary Chan's Mean+Var EVT tree, O(log N) beam)
  6. KV R-Tree Beam (Online Semantic AABB R-Tree, O(log N) beam)
  7. KV R-Tree Exact BnB (Online Semantic AABB R-Tree, 100% lossless Branch-and-Bound)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_eval import _build_haystack_prompt, load_eval_text
from mskv.device import get_best_device, get_best_dtype, get_peak_vram_mb, reset_peak_vram
from mskv.paged_tree import PagedTreeConfig, patched_paged_tree_attention


@torch.no_grad()
def evaluate_paged_ppl(
    model,
    token_ids: torch.Tensor,
    cfg: PagedTreeConfig,
    device: torch.device,
    backend: str,
    chunk_size: int = 32,
) -> Dict[str, float]:
    reset_peak_vram(backend)
    seq_len = token_ids.shape[1]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

    total_nll = 0.0
    total_tokens = 0
    prev_last_logit: Optional[torch.Tensor] = None
    t0 = time.perf_counter()

    with patched_paged_tree_attention(model, cfg) as cache:
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            chunk = token_ids[:, start:end].to(device)
            pos = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)

            out = model(input_ids=chunk, position_ids=pos, use_cache=False)
            logits = out.logits.float()

            if prev_last_logit is not None:
                total_nll += loss_fct(prev_last_logit, chunk[:, 0]).item()
                total_tokens += 1

            if logits.shape[1] > 1:
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = chunk[:, 1:].reshape(-1)
                total_nll += loss_fct(shift_logits, shift_labels).item()
                total_tokens += shift_labels.numel()

            prev_last_logit = logits[:, -1, :].detach()

        summary = cache.summary()

    elapsed = time.perf_counter() - t0
    avg_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(min(avg_nll, 20.0))
    return {
        "ppl": ppl,
        "nll": avg_nll,
        "active_slots": int(summary["active_slots"]),
        "compression_x": summary["compression_ratio"],
        "scan_pct": summary["scan_pct"],
        "tok_per_sec": seq_len / max(elapsed, 1e-6),
        "peak_vram_mb": get_peak_vram_mb(backend),
    }


@torch.no_grad()
def greedy_generate_paged(
    model,
    prompt_ids: torch.Tensor,
    cfg: PagedTreeConfig,
    device: torch.device,
    max_new_tokens: int = 20,
    chunk_size: int = 32,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[int], Dict[str, float]]:
    prompt_len = prompt_ids.shape[1]
    generated: List[int] = []

    with patched_paged_tree_attention(model, cfg) as cache:
        # Prefill prompt[:-1] in chunks, then run the final prompt token ("Answer:") at T=1
        # so the first generated digit gets dedicated single-token page routing (matching real decode).
        prefill_end = max(0, prompt_len - 1)
        for start in range(0, prefill_end, chunk_size):
            end = min(prefill_end, start + chunk_size)
            chunk = prompt_ids[:, start:end].to(device)
            pos = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)
            model(input_ids=chunk, position_ids=pos, use_cache=False)

        last_tok = prompt_ids[:, prompt_len - 1 : prompt_len].to(device)
        last_pos = torch.tensor([[prompt_len - 1]], device=device, dtype=torch.long)
        last_logits = model(input_ids=last_tok, position_ids=last_pos, use_cache=False).logits[:, -1, :]

        cur_pos = prompt_len
        next_tok = last_logits.argmax(dim=-1, keepdim=True)
        for _ in range(max_new_tokens):
            tok_id = int(next_tok.item())
            generated.append(tok_id)
            if eos_token_id is not None and tok_id == eos_token_id:
                break
            pos = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
            out = model(input_ids=next_tok, position_ids=pos, use_cache=False)
            cur_pos += 1
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        summary = cache.summary()
    return generated, summary


def evaluate_paged_niah(
    model,
    tokenizer,
    filler_text: str,
    cfg: PagedTreeConfig,
    device: torch.device,
    context_len: int = 2048,
    depths: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    chunk_size: int = 32,
) -> Dict[str, object]:
    rng = random.Random(123)
    hits = 0
    per_depth: Dict[str, bool] = {}
    scan_pcts: List[float] = []

    for depth in depths:
        passkey = str(rng.randint(100000, 999999))
        needle = f"IMPORTANT SECRET RECORD: The magic vault passkey is {passkey}. Remember {passkey}."
        question = "What is the magic vault passkey mentioned in the document? Reply with the 6-digit number only."
        prompt_ids = _build_haystack_prompt(
            tokenizer, filler_text, target_tokens=context_len, needles=[(depth, needle)], question=question
        )
        gen_ids, summary = greedy_generate_paged(
            model,
            prompt_ids,
            cfg,
            device,
            max_new_tokens=20,
            chunk_size=chunk_size,
            eos_token_id=tokenizer.eos_token_id,
        )
        ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ok = passkey in ans
        hits += int(ok)
        per_depth[f"{int(depth * 100)}%"] = ok
        scan_pcts.append(summary["scan_pct"])

    return {
        "niah_acc": 100.0 * hits / max(1, len(depths)),
        "niah_per_depth": per_depth,
        "niah_scan_pct": sum(scan_pcts) / max(1, len(scan_pcts)),
    }


def evaluate_paged_multikey(
    model,
    tokenizer,
    filler_text: str,
    cfg: PagedTreeConfig,
    device: torch.device,
    context_len: int = 2048,
    chunk_size: int = 32,
) -> Dict[str, object]:
    pairs = [
        ("ALPHA-ORBIT", "482917"),
        ("BETA-QUARTZ", "739104"),
        ("GAMMA-PULSE", "295648"),
        ("DELTA-PRISM", "614823"),
    ]
    depths = [0.12, 0.24, 0.36, 0.48]
    needles = [
        (d, f"REGISTRY ENTRY: The access code for {name} is {code}.")
        for d, (name, code) in zip(depths, pairs)
    ]

    hits = 0
    details: Dict[str, bool] = {}
    for name, code in pairs:
        question = f"What is the exact 6-digit access code for {name}? Reply with the number only."
        prompt_ids = _build_haystack_prompt(
            tokenizer, filler_text, target_tokens=context_len, needles=needles, question=question
        )
        gen_ids, _ = greedy_generate_paged(
            model,
            prompt_ids,
            cfg,
            device,
            max_new_tokens=20,
            chunk_size=chunk_size,
            eos_token_id=tokenizer.eos_token_id,
        )
        ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ok = code in ans
        hits += int(ok)
        details[name] = ok

    return {
        "multikey_acc": 100.0 * hits / len(pairs),
        "multikey_details": details,
    }


def build_paged_suite(args) -> List[Tuple[str, PagedTreeConfig]]:
    base = dict(
        sink_size=args.sink_size,
        local_size=args.local_size,
        page_size=args.page_size,
        beam_width=args.beam_width,
        branching=args.branching,
        rtree_max_fanout=args.rtree_fanout,
        ucb_beta=args.ucb_beta,
        offload_to_cpu=not args.no_offload,
    )
    k = args.top_k_pages
    h = args.halo_tokens
    pin = args.pinned_size

    if args.suite == "v2":
        return [
            ("1. Full KV (Upper Bound)", PagedTreeConfig(router="full", top_k_pages=k, **base)),
            ("2. StreamingLLM (Eviction)", PagedTreeConfig(router="streaming_llm", top_k_pages=k, **base)),
            (f"3. Flat Quest (AABB, 100% scan, K={k})", PagedTreeConfig(router="flat_quest", top_k_pages=k, **base)),
            (f"4. Chrono-BVH Tree (AABB, beam, K={k})", PagedTreeConfig(router="chrono_bvh", top_k_pages=k, **base)),
            (f"5. Chrono-UCB Tree (Mean+Var, beam, K={k})", PagedTreeConfig(router="chrono_ucb", top_k_pages=k, **base)),
            (f"6. KV R-Tree Beam (Semantic AABB, K={k})", PagedTreeConfig(router="rtree_beam", top_k_pages=k, **base)),
            (f"7. KV R-Tree Exact BnB (Lossless, K={k})", PagedTreeConfig(router="rtree_bnb", top_k_pages=k, **base)),
            (f"8. KV R-Tree Beam (Semantic AABB, K={k * 2})", PagedTreeConfig(router="rtree_beam", top_k_pages=k * 2, **base)),
        ]

    if args.suite == "v3":
        base_wide = {**base, "beam_width": max(args.beam_width, 6)}
        return [
            ("1. Full KV (Upper Bound)", PagedTreeConfig(router="full", top_k_pages=k, **base)),
            ("2. StreamingLLM (Eviction)", PagedTreeConfig(router="streaming_llm", top_k_pages=k, **base)),
            (f"3. Flat Quest (K={k}, no halo)", PagedTreeConfig(router="flat_quest", top_k_pages=k, halo_tokens=0, pinned_size=0, **base)),
            (f"4. Flat Quest + Halo={h} (K={k})", PagedTreeConfig(router="flat_quest", top_k_pages=k, halo_tokens=h, pinned_size=0, **base)),
            (f"5. Chrono-BVH (K={k}, no halo)", PagedTreeConfig(router="chrono_bvh", top_k_pages=k, halo_tokens=0, pinned_size=0, **base)),
            (f"6. Chrono-BVH + Halo={h} (K={k})", PagedTreeConfig(router="chrono_bvh", top_k_pages=k, halo_tokens=h, pinned_size=0, **base)),
            (f"7. Chrono-Hybrid + Halo={h} (K={k})", PagedTreeConfig(router="chrono_hybrid", top_k_pages=k, halo_tokens=h, pinned_size=0, **base)),
            (f"8. Chrono-BVH + Halo={h} + Pin={pin} (K={k})", PagedTreeConfig(router="chrono_bvh", top_k_pages=k, halo_tokens=h, pinned_size=pin, **base)),
            (f"9. Chrono-Hybrid + Halo={h} + Pin={pin} (K={k})", PagedTreeConfig(router="chrono_hybrid", top_k_pages=k, halo_tokens=h, pinned_size=pin, **base)),
            (f"10. Chrono-BVH + Halo={h} + Pin={pin} (K={k + 2}, beam=6)", PagedTreeConfig(router="chrono_bvh", top_k_pages=k + 2, halo_tokens=h, pinned_size=pin, **base_wide)),
        ]

    if args.suite == "v35":
        fm = args.funnel_mult
        base_wide = {**base, "beam_width": max(args.beam_width, 6)}
        return [
            ("1. Full KV (Upper Bound)", PagedTreeConfig(router="full", top_k_pages=k, **base)),
            (f"2. Flat Quest + Halo={h} (K={k})", PagedTreeConfig(router="flat_quest", top_k_pages=k, halo_tokens=h, pinned_size=0, **base)),
            (f"3. Chrono-Hybrid + Halo={h} + Pin={pin} (K={k}, flat)", PagedTreeConfig(router="chrono_hybrid", top_k_pages=k, halo_tokens=h, pinned_size=pin, decouple_pinned=False, funnel_mult=1, **base)),
            (f"4. Chrono-Hybrid + Halo={h} + Pin={pin} + Decoupled (K={k})", PagedTreeConfig(router="chrono_hybrid", top_k_pages=k, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=1, **base)),
            (f"5. Chrono-Hybrid + Decoupled + Funnel x{fm} (K={k})", PagedTreeConfig(router="chrono_hybrid", top_k_pages=k, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base)),
            (f"6. Chrono-BVH + Decoupled + Funnel x{fm} (K={k})", PagedTreeConfig(router="chrono_bvh", top_k_pages=k, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base)),
            (f"7. Chrono-Hybrid + Decoupled + Funnel x{fm} (K={k + 2})", PagedTreeConfig(router="chrono_hybrid", top_k_pages=k + 2, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base)),
            (f"8. Chrono-Hybrid + Decoupled + Funnel x{fm} (K={k + 2}, beam=6)", PagedTreeConfig(router="chrono_hybrid", top_k_pages=k + 2, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base_wide)),
        ]

    # Default v36 suite: tests Chrono-Hybrid-L2 (AABB on lvl<=1, UCB on lvl>=2) + apples-to-apples K=6 vs Flat Quest.
    fm = args.funnel_mult
    base_wide = {**base, "beam_width": max(args.beam_width, 6)}
    suite: List[Tuple[str, PagedTreeConfig]] = []
    if args.include_full:
        suite.append(("0. Full KV (Upper Bound)", PagedTreeConfig(router="full", top_k_pages=k, **base)))
    suite.extend(
        [
            (f"1. Flat Quest + Halo={h} (K={k})", PagedTreeConfig(router="flat_quest", top_k_pages=k, halo_tokens=h, pinned_size=0, **base)),
            (f"2. Flat Quest + Halo={h} (K={k + 2})", PagedTreeConfig(router="flat_quest", top_k_pages=k + 2, halo_tokens=h, pinned_size=0, **base)),
            (f"3. Chrono-Hybrid-L2 + Decoupled + Funnel x{fm} (K={k}, beam={args.beam_width})", PagedTreeConfig(router="chrono_hybrid_l2", top_k_pages=k, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base)),
            (f"4. Chrono-Hybrid-L2 + Decoupled + Funnel x{fm} (K={k + 2}, beam={args.beam_width})", PagedTreeConfig(router="chrono_hybrid_l2", top_k_pages=k + 2, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base)),
            (f"5. Chrono-BVH + Decoupled + Funnel x{fm} (K={k + 2}, beam=6)", PagedTreeConfig(router="chrono_bvh", top_k_pages=k + 2, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base_wide)),
            (f"6. Chrono-Hybrid-L2 + Decoupled + Funnel x{fm} (K={k + 2}, beam=6)", PagedTreeConfig(router="chrono_hybrid_l2", top_k_pages=k + 2, halo_tokens=h, pinned_size=pin, decouple_pinned=True, funnel_mult=fm, **base_wide)),
        ]
    )
    return suite


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = ["Config", "PPL", "NIAH%", "MultiKey%", "ActiveSlots", "Compr", "Scan%", "Tok/s"]
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        print(
            f"| {r['name']} "
            f"| {r['ppl']:.2f} "
            f"| {r['niah_acc']:.0f}% "
            f"| {r['multikey_acc']:.0f}% "
            f"| {r['active_slots']} "
            f"| {r['compression_x']:.1f}x "
            f"| {r['scan_pct']:.1f}% "
            f"| {r['tok_per_sec']:.1f} |"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Hierarchical Tree-Routed Exact KV Paging.")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--text-file", type=str, required=True, help="Path to evaluation text (e.g. eval.txt)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--sink-size", type=int, default=4)
    p.add_argument("--local-size", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--suite", type=str, default="v36", choices=["v36", "v35", "v3", "v2"], help="Benchmark suite (v36=Hybrid-L2+K6, v35=Decoupled+Funnel, v3=Halo+Pin, v2=R-Tree)")
    p.add_argument("--include-full", action="store_true", help="Prepend Full KV upper bound in v36 suite (useful for 8192 or 1.5B)")
    p.add_argument("--top-k-pages", type=int, default=4, help="Default K=4 pages * 16 = 64 tokens")
    p.add_argument("--halo-tokens", type=int, default=8, help="Induction Halo tokens appended from next page (0..page_size)")
    p.add_argument("--pinned-size", type=int, default=32, help="Static GPU-pinned Heavy-Hitter anchor budget")
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--funnel-mult", type=int, default=2, help="Funnel Beam multiplier on coarse upper levels (lvl >= 2)")
    p.add_argument("--branching", type=int, default=4)
    p.add_argument("--rtree-fanout", type=int, default=8)
    p.add_argument("--ucb-beta", type=float, default=1.0)
    p.add_argument("--no-offload", action="store_true", help="Keep KV pages on GPU instead of CPU RAM")
    p.add_argument(
        "--routers",
        type=str,
        default=None,
        help="Optional comma-separated filter of router names (e.g. flat_quest,rtree_beam,rtree_bnb)",
    )
    p.add_argument("--out-json", type=str, default="results/paged_tree_2k.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, backend = get_best_device(args.device)
    dtype = get_best_dtype(device, backend, args.dtype)
    print(f"[PagedEval] model={args.model} | device={device} ({backend}) | dtype={dtype} | offload_cpu={not args.no_offload}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    eval_text = load_eval_text(args.text_file, min_chars=max(120_000, args.seq_len * 10))
    ids = tokenizer.encode(eval_text, add_special_tokens=False)
    while len(ids) < args.seq_len + 64:
        ids = ids + ids
    token_ids = torch.tensor([ids[: args.seq_len]], dtype=torch.long)

    suite = build_paged_suite(args)
    if args.routers:
        allowed = {r.strip() for r in args.routers.split(",") if r.strip()}
        suite = [(name, cfg) for name, cfg in suite if cfg.router in allowed]

    rows: List[Dict[str, object]] = []
    for name, cfg in suite:
        print(f"\n>>> Running: {name}")
        ppl_res = evaluate_paged_ppl(model, token_ids, cfg, device, backend, chunk_size=args.chunk_size)
        niah_res = evaluate_paged_niah(
            model, tokenizer, eval_text, cfg, device, context_len=args.seq_len, chunk_size=args.chunk_size
        )
        mk_res = evaluate_paged_multikey(
            model, tokenizer, eval_text, cfg, device, context_len=args.seq_len, chunk_size=args.chunk_size
        )
        row = {"name": name, "config": asdict(cfg), **ppl_res, **niah_res, **mk_res}
        rows.append(row)
        print(
            f"    PPL={row['ppl']:.2f} | NIAH={row['niah_acc']:.0f}% {row['niah_per_depth']} "
            f"| MultiKey={row['multikey_acc']:.0f}% | Active={row['active_slots']} ({row['compression_x']:.1f}x) "
            f"| Scan={row['scan_pct']:.1f}% | {row['tok_per_sec']:.1f} tok/s"
        )
        if backend in ("rocm", "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(rows)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"model": args.model, "seq_len": args.seq_len, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

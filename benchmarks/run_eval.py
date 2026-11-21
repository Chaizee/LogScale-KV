"""Long-context evaluation & ablation suite for Multi-Scale Streaming KV-Cache.

Runs cleanly on AMD GPUs (ROCm / DirectML), CUDA, or CPU without CUDA-only extensions.
Evaluates:
  1. PPL:      Long-context streaming perplexity (WikiText / custom file / built-in fallback).
  2. NIAH:     Needle-In-A-Haystack passkey retrieval across depths 10%..90%.
  3. MultiKey: Mini-RULER multi-key retrieval (4 distinct keys in distant history).
  4. Cliff:    Per-token NLL diagnostic across power-of-2 history boundaries (Carry-Bit vs k-slot).
  5. Ablation: Full comparison matrix across modes, RoPE damping, mergers, and log-bias.
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

from mskv import (
    CacheConfig,
    ModelMergers,
    get_best_device,
    get_best_dtype,
    patched_multiscale_attention,
)
from mskv.device import get_peak_vram_mb, reset_peak_vram


_FALLBACK_PARAGRAPHS = [
    (
        "In distributed systems and modern numerical linear algebra, hierarchical low-rank "
        "approximations enable sub-quadratic interaction between distant spatial or temporal regions. "
        "When fine-grained local interactions dominate nearby steps while coarse summaries capture "
        "long-range context, logarithmic multi-resolution representations balance memory against fidelity."
    ),
    (
        "During autoregressive decoding in transformer language models, the key-value cache grows "
        "linearly with sequence length and quickly becomes the primary memory bandwidth bottleneck. "
        "Eviction heuristics discard older states irreversibly, whereas structured compression retains "
        "macroscopic summaries of distant passages while protecting critical anchor tokens."
    ),
    (
        "Rotary positional embeddings encode relative distance through block-diagonal rotations across "
        "a geometric spectrum of angular frequencies. High-frequency components rotate rapidly every few "
        "tokens, while low-frequency components vary smoothly across thousands of positions and carry "
        "macro-structural ordering information across chapters and sections."
    ),
    (
        "Historical archives of scientific expeditions document how instrumentation evolved from heavy "
        "mechanical chronometers to compact optical interferometers. Observers recorded barometric "
        "pressure, geomagnetic declination, and cloud formations at regular intervals, producing dense "
        "longitudinal records that reward careful statistical aggregation over broad windows."
    ),
]


def load_eval_text(text_file: Optional[str], min_chars: int = 120_000) -> str:
    """Load long evaluation text from a local file, HuggingFace WikiText, or built-in corpus."""
    if text_file:
        text = Path(text_file).read_text(encoding="utf-8")
        if len(text) >= min_chars:
            return text
        reps = (min_chars // max(1, len(text))) + 1
        return ("\n\n".join([text] * reps))[: min_chars + 4096]

    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        joined = "\n".join(row["text"] for row in ds if row["text"].strip())
        if len(joined) >= min_chars:
            return joined
        reps = (min_chars // len(joined)) + 1
        return ("\n\n".join([joined] * reps))[: min_chars + 4096]
    except Exception as exc:
        print(f"[Info] Could not load wikitext from HF ({exc}); using built-in synthetic prose.")
        chunks = []
        rng = random.Random(42)
        idx = 0
        while sum(len(c) for c in chunks) < min_chars:
            p = _FALLBACK_PARAGRAPHS[idx % len(_FALLBACK_PARAGRAPHS)]
            chunks.append(f"Section {idx + 1}. {p} Reference observation {rng.randint(100, 999)}.")
            idx += 1
        return "\n\n".join(chunks)


@torch.no_grad()
def evaluate_ppl(
    model,
    tokenizer,
    token_ids: torch.Tensor,
    cfg: CacheConfig,
    device: torch.device,
    backend: str,
    chunk_size: int = 64,
    model_mergers: Optional[ModelMergers] = None,
) -> Dict[str, float]:
    """Stream through token_ids chunk-by-chunk and measure perplexity, speed, and VRAM."""
    reset_peak_vram(backend)
    seq_len = token_ids.shape[1]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

    total_nll = 0.0
    total_tokens = 0
    prev_last_logit: Optional[torch.Tensor] = None
    t0 = time.perf_counter()

    with patched_multiscale_attention(model, cfg, model_mergers=model_mergers) as cache:
        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            chunk = token_ids[:, start:end].to(device)
            pos = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)

            out = model(input_ids=chunk, position_ids=pos, use_cache=False)
            logits = out.logits.float()  # (1, T, V)

            # Cross-boundary prediction from previous chunk's final logit to current chunk's first token.
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
        "tokens": total_tokens,
        "cached_entries": summary["cached_entries"],
        "compression_x": summary["compression_ratio"],
        "tok_per_sec": seq_len / max(elapsed, 1e-6),
        "peak_vram_mb": get_peak_vram_mb(backend),
    }


@torch.no_grad()
def greedy_generate_streaming(
    model,
    prompt_ids: torch.Tensor,
    cfg: CacheConfig,
    device: torch.device,
    max_new_tokens: int = 24,
    chunk_size: int = 64,
    eos_token_id: Optional[int] = None,
    model_mergers: Optional[ModelMergers] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """Prefill prompt in chunks and greedily decode max_new_tokens using MultiScaleKVCache."""
    prompt_len = prompt_ids.shape[1]
    generated: List[int] = []

    with patched_multiscale_attention(model, cfg, model_mergers=model_mergers) as cache:
        last_logits = None
        for start in range(0, prompt_len, chunk_size):
            end = min(prompt_len, start + chunk_size)
            chunk = prompt_ids[:, start:end].to(device)
            pos = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)
            out = model(input_ids=chunk, position_ids=pos, use_cache=False)
            last_logits = out.logits[:, -1, :]

        cur_pos = prompt_len
        next_tok = last_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

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


def _build_haystack_prompt(
    tokenizer,
    filler_text: str,
    target_tokens: int,
    needles: List[Tuple[float, str]],
    question: str,
) -> torch.Tensor:
    """Insert one or more needles at given fractional depths into filler text."""
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    while len(filler_ids) < target_tokens + 256:
        filler_ids = filler_ids + filler_ids
    filler_ids = filler_ids[:target_tokens]

    # Insert needles from deepest to shallowest so earlier indices don't shift.
    inserted = list(filler_ids)
    for depth, needle_str in sorted(needles, key=lambda x: x[0], reverse=True):
        n_ids = tokenizer.encode(f"\n{needle_str}\n", add_special_tokens=False)
        pos = max(8, min(len(inserted) - 8, int(depth * len(inserted))))
        inserted[pos:pos] = n_ids

    prefix_ids = tokenizer.encode("Read the following document carefully and answer the final question.\n\n", add_special_tokens=True)
    q_ids = tokenizer.encode(f"\n\nQuestion: {question}\nAnswer:", add_special_tokens=False)
    full_ids = prefix_ids + inserted + q_ids
    return torch.tensor([full_ids], dtype=torch.long)


def evaluate_niah(
    model,
    tokenizer,
    filler_text: str,
    cfg: CacheConfig,
    device: torch.device,
    context_len: int = 2048,
    depths: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    chunk_size: int = 64,
    model_mergers: Optional[ModelMergers] = None,
) -> Dict[str, object]:
    """Single-needle passkey retrieval across depths."""
    rng = random.Random(123)
    hits = 0
    per_depth: Dict[str, bool] = {}

    for depth in depths:
        passkey = str(rng.randint(100000, 999999))
        needle = f"IMPORTANT SECRET RECORD: The magic vault passkey is {passkey}. Remember {passkey}."
        question = "What is the magic vault passkey mentioned in the document? Reply with the 6-digit number only."
        prompt_ids = _build_haystack_prompt(
            tokenizer, filler_text, target_tokens=context_len, needles=[(depth, needle)], question=question
        )
        gen_ids, _ = greedy_generate_streaming(
            model,
            prompt_ids,
            cfg,
            device,
            max_new_tokens=20,
            chunk_size=chunk_size,
            eos_token_id=tokenizer.eos_token_id,
            model_mergers=model_mergers,
        )
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ok = passkey in answer
        hits += int(ok)
        per_depth[f"{int(depth * 100)}%"] = ok

    return {
        "niah_acc": 100.0 * hits / max(1, len(depths)),
        "per_depth": per_depth,
    }


def evaluate_multikey(
    model,
    tokenizer,
    filler_text: str,
    cfg: CacheConfig,
    device: torch.device,
    context_len: int = 2048,
    chunk_size: int = 64,
    model_mergers: Optional[ModelMergers] = None,
) -> Dict[str, object]:
    """Mini-RULER: 4 distinct codename->code pairs hidden in the distant half of the context."""
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
        gen_ids, _ = greedy_generate_streaming(
            model,
            prompt_ids,
            cfg,
            device,
            max_new_tokens=20,
            chunk_size=chunk_size,
            eos_token_id=tokenizer.eos_token_id,
            model_mergers=model_mergers,
        )
        ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        ok = code in ans
        hits += int(ok)
        details[name] = ok

    return {
        "multikey_acc": 100.0 * hits / len(pairs),
        "details": details,
    }


@torch.no_grad()
def evaluate_resolution_cliff(
    model,
    token_ids: torch.Tensor,
    base_cfg: CacheConfig,
    device: torch.device,
    power_of_two: int = 512,
    window: int = 16,
) -> Dict[str, Dict[str, float]]:
    """Measure exact per-token NLL immediately before vs after N_hist = power_of_two.

    Compares 1-slot Carry-Bit (which suffers a total tree collapse at N_hist = 2^k)
    against k-slot Redundant Exponential Histogram (which maintains smooth resolution).
    """
    target_t = base_cfg.sink_size + base_cfg.local_size + power_of_two
    needed = target_t + window + 2
    if token_ids.shape[1] < needed:
        raise ValueError(f"Need at least {needed} tokens for cliff diagnostic at 2^k={power_of_two}.")

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    results: Dict[str, Dict[str, float]] = {}

    for label, mode, k_slots in [
        ("carry_bit_k1", "carry_bit_original", 1),
        ("redundant_k4", "multiscale_tree", 4),
    ]:
        cfg = CacheConfig(
            mode=mode,
            sink_size=base_cfg.sink_size,
            local_size=base_cfg.local_size,
            salient_size=0,
            slots_per_level=k_slots,
            rope_mode=base_cfg.rope_mode,
            merger_mode=base_cfg.merger_mode,
            use_log_bias=base_cfg.use_log_bias,
        )
        nll_before: List[float] = []
        nll_after: List[float] = []
        tree_before = 0
        tree_after = 0

        with patched_multiscale_attention(model, cfg) as cache:
            # Fast-forward up to target_t - window in chunks, then step 1-by-1 across the boundary.
            fast_end = max(0, target_t - window - 1)
            prev_logit: Optional[torch.Tensor] = None
            for s in range(0, fast_end, 64):
                e = min(fast_end, s + 64)
                chunk = token_ids[:, s:e].to(device)
                pos = torch.arange(s, e, device=device, dtype=torch.long).unsqueeze(0)
                out = model(input_ids=chunk, position_ids=pos, use_cache=False)
                prev_logit = out.logits[:, -1, :].float()

            for t in range(fast_end, target_t + window + 1):
                tok = token_ids[:, t : t + 1].to(device)
                if prev_logit is not None:
                    # prev_logit was produced at step t-1:
                    # - for t <= target_t, step t-1 saw pre-collapse N_hist <= power_of_two - 1;
                    # - for t >= target_t + 1, step t-1 saw post-collapse N_hist >= power_of_two.
                    nll = float(loss_fct(prev_logit, tok[:, 0]).item())
                    if target_t - window + 1 <= t <= target_t:
                        nll_before.append(nll)
                    elif target_t + 1 <= t <= target_t + window:
                        nll_after.append(nll)

                pos = torch.tensor([[t]], device=device, dtype=torch.long)
                out = model(input_ids=tok, position_ids=pos, use_cache=False)
                prev_logit = out.logits[:, -1, :].float()

                # At t = target_t - 2, total_tokens_seen = S + W_loc + (2^k - 1) => N_hist = 2^k - 1.
                # At t = target_t - 1, total_tokens_seen = S + W_loc + 2^k       => N_hist = 2^k (collapse!).
                if t == target_t - 2:
                    tree_before = cache.layers[0].stats()["tree"]
                elif t == target_t - 1:
                    tree_after = cache.layers[0].stats()["tree"]

        avg_b = sum(nll_before) / max(1, len(nll_before))
        avg_a = sum(nll_after) / max(1, len(nll_after))
        results[label] = {
            "tree_slots_before": tree_before,
            "tree_slots_after": tree_after,
            "nll_before": avg_b,
            "nll_after": avg_a,
            "delta_nll": avg_a - avg_b,
        }

    return results


def build_ablation_suite(args) -> List[Tuple[str, CacheConfig]]:
    """Core scientific ablation matrix directly testing every claim and fix."""
    common = dict(
        sink_size=args.sink_size,
        local_size=args.local_size,
        salient_size=args.salient_size,
        slots_per_level=args.slots_per_level,
        salience_metric=args.salience_metric,
    )
    return [
        ("1. Full KV (Upper Bound)", CacheConfig(mode="full", **common)),
        ("2. StreamingLLM (Eviction)", CacheConfig(mode="streaming_llm", **common)),
        ("3. H2O-Lite (Salient Only)", CacheConfig(mode="h2o_lite", **common)),
        (
            "4. Orig Carry-Bit (k=1, BaryRoPE, Stiefel)",
            CacheConfig(
                mode="carry_bit_original",
                rope_mode="pre_rope_barycentric",
                merger_mode="stiefel_unscaled",
                use_log_bias=True,
                **common,
            ),
        ),
        (
            "5. Orig Carry-Bit (k=1, BaryRoPE, Mean)",
            CacheConfig(
                mode="carry_bit_original",
                rope_mode="pre_rope_barycentric",
                merger_mode="mean",
                use_log_bias=True,
                **common,
            ),
        ),
        (
            "6. MS-Tree (k-slot, GaussDampedRoPE, Mean)",
            CacheConfig(
                mode="multiscale_tree",
                rope_mode="pre_rope_damped_gauss",
                merger_mode="mean",
                use_log_bias=True,
                **common,
            ),
        ),
        (
            "7. MS-Tree (k-slot, PostRoPE, Mean)",
            CacheConfig(
                mode="multiscale_tree",
                rope_mode="post_rope",
                merger_mode="mean",
                use_log_bias=True,
                **common,
            ),
        ),
        (
            "8. MS-Tree (k-slot, GaussDamped, No LogBias)",
            CacheConfig(
                mode="multiscale_tree",
                rope_mode="pre_rope_damped_gauss",
                merger_mode="mean",
                use_log_bias=False,
                **common,
            ),
        ),
        (
            "9. MS-Tree (k-slot, Salience-Weighted Merge)",
            CacheConfig(
                mode="multiscale_tree",
                rope_mode="pre_rope_damped_gauss",
                merger_mode="salience_weighted",
                use_log_bias=True,
                **common,
            ),
        ),
        (
            "10. Hybrid MS-KV (Salient + k-slot Tree + GaussDamped)",
            CacheConfig(
                mode="hybrid_ms",
                rope_mode="pre_rope_damped_gauss",
                merger_mode="salience_weighted",
                use_log_bias=True,
                **common,
            ),
        ),
    ]


def print_markdown_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "Config",
        "PPL",
        "NIAH%",
        "MultiKey%",
        "CacheSlots",
        "Compr",
        "Tok/s",
        "VRAM(MB)",
    ]
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        print(
            f"| {r['name']} "
            f"| {r.get('ppl', float('nan')):.2f} "
            f"| {r.get('niah_acc', float('nan')):.0f}% "
            f"| {r.get('multikey_acc', float('nan')):.0f}% "
            f"| {r.get('cached_entries', 0)} "
            f"| {r.get('compression_x', 1.0):.1f}x "
            f"| {r.get('tok_per_sec', 0.0):.1f} "
            f"| {r.get('peak_vram_mb', 0.0):.0f} |"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Multi-Scale Streaming KV-Cache prototype.")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | directml | cpu")
    p.add_argument("--dtype", type=str, default="auto", help="auto | bf16 | fp16 | fp32")
    p.add_argument("--task", type=str, default="ablation", choices=["ppl", "niah", "multikey", "cliff", "ablation", "all"])
    p.add_argument("--seq-len", type=int, default=2048, help="Sequence length for PPL and retrieval tasks")
    p.add_argument("--chunk-size", type=int, default=64, help="Chunk size for streaming prefill")
    p.add_argument("--text-file", type=str, default=None, help="Optional local UTF-8 text file for PPL/haystack")
    # Single-config parameters (used when --task is not ablation)
    p.add_argument("--mode", type=str, default="hybrid_ms")
    p.add_argument("--sink-size", type=int, default=4)
    p.add_argument("--local-size", type=int, default=128)
    p.add_argument("--salient-size", type=int, default=64)
    p.add_argument("--slots-per-level", type=int, default=4)
    p.add_argument("--rope-mode", type=str, default="pre_rope_damped_gauss")
    p.add_argument("--merger-mode", type=str, default="salience_weighted")
    p.add_argument("--salience-metric", type=str, default="hybrid")
    p.add_argument("--no-log-bias", action="store_true")
    p.add_argument("--merger-ckpt", type=str, default=None, help="Path to trained ModelMergers checkpoint")
    p.add_argument("--cliff-pow2", type=int, default=512, help="Power-of-2 boundary for cliff diagnostic")
    p.add_argument("--out-json", type=str, default="results/eval_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, backend = get_best_device(args.device)
    dtype = get_best_dtype(device, backend, args.dtype)
    print(f"[Setup] model={args.model} | device={device} ({backend}) | dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    loaded_mergers: Optional[ModelMergers] = None
    if args.merger_ckpt:
        loaded_mergers = ModelMergers.load(args.merger_ckpt, map_location=device)
        loaded_mergers.to(device)
        loaded_mergers.eval()
        loaded_mergers.materialize_all()
        print(f"[Setup] Loaded trained GatedCayleyMerger from {args.merger_ckpt}")

    eval_text = load_eval_text(args.text_file, min_chars=max(120_000, args.seq_len * 10))
    ids = tokenizer.encode(eval_text, add_special_tokens=False)
    cliff_needed = args.sink_size + args.local_size + args.cliff_pow2 + 32
    min_ids = max(args.seq_len + 64, cliff_needed)
    while len(ids) < min_ids:
        ids = ids + ids
    token_ids = torch.tensor([ids[: args.seq_len]], dtype=torch.long)
    cliff_token_ids = torch.tensor([ids[: max(args.seq_len, cliff_needed)]], dtype=torch.long)

    all_output: Dict[str, object] = {
        "model": args.model,
        "backend": backend,
        "dtype": str(dtype),
        "seq_len": args.seq_len,
    }

    if args.task in ("ablation", "all"):
        suite = build_ablation_suite(args)
        if loaded_mergers is not None:
            suite.append(
                (
                    "11. Hybrid MS-KV (Trained GatedCayley)",
                    CacheConfig(
                        mode="hybrid_ms",
                        sink_size=args.sink_size,
                        local_size=args.local_size,
                        salient_size=args.salient_size,
                        slots_per_level=args.slots_per_level,
                        rope_mode="pre_rope_damped_gauss",
                        merger_mode="learned",
                        use_log_bias=True,
                        salience_metric=args.salience_metric,
                    ),
                )
            )

        rows: List[Dict[str, object]] = []
        for name, cfg in suite:
            print(f"\n>>> Running: {name}")
            mergers = loaded_mergers if cfg.merger_mode == "learned" else None
            ppl_res = evaluate_ppl(
                model, tokenizer, token_ids, cfg, device, backend, chunk_size=args.chunk_size, model_mergers=mergers
            )
            niah_res = evaluate_niah(
                model,
                tokenizer,
                eval_text,
                cfg,
                device,
                context_len=args.seq_len,
                chunk_size=args.chunk_size,
                model_mergers=mergers,
            )
            mk_res = evaluate_multikey(
                model,
                tokenizer,
                eval_text,
                cfg,
                device,
                context_len=args.seq_len,
                chunk_size=args.chunk_size,
                model_mergers=mergers,
            )
            row = {"name": name, "config": asdict(cfg), **ppl_res, **niah_res, **mk_res}
            rows.append(row)
            print(
                f"    PPL={row['ppl']:.2f} | NIAH={row['niah_acc']:.0f}% | "
                f"MultiKey={row['multikey_acc']:.0f}% | Slots={row['cached_entries']} ({row['compression_x']:.1f}x)"
            )
            if backend in ("rocm", "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print_markdown_table(rows)
        all_output["ablation"] = rows

    else:
        cfg = CacheConfig(
            mode=args.mode,
            sink_size=args.sink_size,
            local_size=args.local_size,
            salient_size=args.salient_size,
            slots_per_level=args.slots_per_level,
            rope_mode=args.rope_mode,
            merger_mode="learned" if loaded_mergers is not None else args.merger_mode,
            use_log_bias=not args.no_log_bias,
            salience_metric=args.salience_metric,
        )
        if args.task == "ppl":
            res = evaluate_ppl(
                model, tokenizer, token_ids, cfg, device, backend, chunk_size=args.chunk_size, model_mergers=loaded_mergers
            )
            print("\n[PPL Result]", json.dumps(res, indent=2))
            all_output["ppl"] = res
        elif args.task == "niah":
            res = evaluate_niah(
                model, tokenizer, eval_text, cfg, device, context_len=args.seq_len, chunk_size=args.chunk_size, model_mergers=loaded_mergers
            )
            print("\n[NIAH Result]", json.dumps(res, indent=2))
            all_output["niah"] = res
        elif args.task == "multikey":
            res = evaluate_multikey(
                model, tokenizer, eval_text, cfg, device, context_len=args.seq_len, chunk_size=args.chunk_size, model_mergers=loaded_mergers
            )
            print("\n[MultiKey Result]", json.dumps(res, indent=2))
            all_output["multikey"] = res

    if args.task in ("cliff", "all"):
        cliff_cfg = CacheConfig(
            sink_size=args.sink_size,
            local_size=args.local_size,
            rope_mode=args.rope_mode,
            merger_mode="mean",
            use_log_bias=not args.no_log_bias,
        )
        cliff_res = evaluate_resolution_cliff(
            model, cliff_token_ids, cliff_cfg, device, power_of_two=args.cliff_pow2
        )
        print("\n[Power-of-2 Resolution Cliff Diagnostic]")
        print(json.dumps(cliff_res, indent=2))
        all_output["cliff"] = cliff_res

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_output, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

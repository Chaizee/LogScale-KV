"""Lightweight suffix-chunk distillation prototype for GatedCayleyMerger.

Solves the parallel-training bottleneck described in the paper critique:
  1. Split each training document into a prefix (length L_pre) and a suffix (length L_suf).
  2. Run the frozen teacher LLM once with torch.no_grad() to obtain exact teacher
     attention outputs (and/or logits) on the suffix.
  3. Compress the prefix differentiably via the vectorized O(log N) k-slot Exponential
     Histogram Tree using GatedCayleyMerger (initialized at exact mean pooling).
  4. Optimize ONLY the GatedCayleyMerger weights (LLM backbone 100% frozen).

Supports two training modes:
  - ``layerwise`` (default): aligns each layer's attention output on the suffix (MSE + cosine).
    Very low VRAM (<6-8 GB on AMD GPUs) and fast.
  - ``e2e_kl``: aligns end-to-end suffix logits via KL(P_teacher || P_student) through frozen LLM.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_eval import load_eval_text
from mskv import (
    CacheConfig,
    ModelMergers,
    MultiScaleLayerCache,
    get_best_device,
    get_best_dtype,
    patched_multiscale_attention,
)
from mskv.patching import repeat_kv
from mskv.rope import apply_multiscale_rope, apply_rope_1d, extract_rope_params


def build_training_windows(
    tokenizer, text: str, total_len: int, num_samples: int, seed: int = 42
) -> List[torch.Tensor]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < total_len * 4:
        ids = ids + ids
    rng = random.Random(seed)
    max_start = len(ids) - total_len - 1
    windows = []
    for _ in range(num_samples):
        s = rng.randint(0, max_start)
        windows.append(torch.tensor([ids[s : s + total_len]], dtype=torch.long))
    return windows


@torch.no_grad()
def capture_teacher_layer_io(model, input_ids: torch.Tensor, prefix_len: int) -> List[Dict[str, torch.Tensor]]:
    """Capture per-layer hidden inputs & exact teacher attention outputs for layerwise distillation."""
    base_model = getattr(model, "model", model)
    records: List[Dict[str, torch.Tensor]] = [{} for _ in range(len(base_model.layers))]
    hooks = []

    for idx, layer in enumerate(base_model.layers):
        attn = layer.self_attn

        def _hook(module, args, kwargs, output, layer_idx=idx):
            hidden = args[0] if len(args) > 0 else kwargs["hidden_states"]
            attn_out = output[0] if isinstance(output, tuple) else output
            # Store compactly in CPU fp16/bf16 to keep AMD VRAM usage tiny.
            records[layer_idx]["hidden_prefix"] = hidden[:, :prefix_len, :].detach().cpu()
            records[layer_idx]["hidden_suffix"] = hidden[:, prefix_len:, :].detach().cpu()
            records[layer_idx]["teacher_attn_suffix"] = attn_out[:, prefix_len:, :].detach().cpu()

        hooks.append(attn.register_forward_hook(_hook, with_kwargs=True))

    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return records


def layerwise_distill_step(
    model,
    model_mergers: ModelMergers,
    records: List[Dict[str, torch.Tensor]],
    cfg: CacheConfig,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    device: torch.device,
    dtype: torch.dtype,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Compute layer-by-layer attention alignment loss (MSE + 1 - cosine) on suffix."""
    base_model = getattr(model, "model", model)
    cfg_m = model.config
    num_heads = int(cfg_m.num_attention_heads)
    num_kv_heads = int(getattr(cfg_m, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg_m, "head_dim", cfg_m.hidden_size // num_heads))
    num_kv_groups = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    optimizer.zero_grad(set_to_none=True)
    total_loss_val = 0.0
    n_layers = len(base_model.layers)

    for layer_idx, layer in enumerate(base_model.layers):
        attn = layer.self_attn
        rec = records[layer_idx]
        h_pre = rec["hidden_prefix"].to(device=device, dtype=dtype)
        h_suf = rec["hidden_suffix"].to(device=device, dtype=dtype)
        t_out = rec["teacher_attn_suffix"].to(device=device, dtype=torch.float32)

        pre_len = h_pre.shape[1]
        suf_len = h_suf.shape[1]

        with torch.no_grad():
            k_pre = attn.k_proj(h_pre).view(1, pre_len, num_kv_heads, head_dim).transpose(1, 2)
            v_pre = attn.v_proj(h_pre).view(1, pre_len, num_kv_heads, head_dim).transpose(1, 2)
            q_suf = attn.q_proj(h_suf).view(1, suf_len, num_heads, head_dim).transpose(1, 2)
            k_suf = attn.k_proj(h_suf).view(1, suf_len, num_kv_heads, head_dim).transpose(1, 2)
            v_suf = attn.v_proj(h_suf).view(1, suf_len, num_kv_heads, head_dim).transpose(1, 2)
            if hasattr(attn, "q_norm") and attn.q_norm is not None:
                q_suf = attn.q_norm(q_suf)
            if hasattr(attn, "k_norm") and attn.k_norm is not None:
                k_pre = attn.k_norm(k_pre)
                k_suf = attn.k_norm(k_suf)

        pos_pre = torch.arange(0, pre_len, device=device, dtype=torch.float32)
        pos_suf = torch.arange(pre_len, pre_len + suf_len, device=device, dtype=torch.float32)

        # Differentiably compress prefix into the exact k-slot Exponential Histogram Tree.
        layer_cache = MultiScaleLayerCache(cfg, model_mergers.layers[layer_idx])
        k_store = (
            apply_rope_1d(k_pre, pos_pre, inv_freq, attention_scaling=attention_scaling)
            if cfg.rope_mode == "post_rope"
            else k_pre
        )
        layer_cache.append_chunk(k_store, v_pre, pos_pre)

        k_past, v_past, pos_past, count_past, _ = layer_cache.get_past_tensors()
        assert k_past is not None and v_past is not None
        m_past = k_past.shape[2]

        k_past_rot = apply_multiscale_rope(
            k_past, pos_past, count_past, inv_freq, attention_scaling=attention_scaling, mode=cfg.rope_mode
        )
        q_suf_rot = apply_rope_1d(q_suf, pos_suf, inv_freq, attention_scaling=attention_scaling)
        k_suf_rot = apply_rope_1d(k_suf, pos_suf, inv_freq, attention_scaling=attention_scaling)

        k_all = repeat_kv(torch.cat((k_past_rot, k_suf_rot), dim=2), num_kv_groups)
        v_all = repeat_kv(torch.cat((v_past, v_suf), dim=2), num_kv_groups)

        logits = torch.matmul(q_suf_rot.float(), k_all.float().transpose(-2, -1)) * scale
        if cfg.use_log_bias:
            logits[..., :m_past] = logits[..., :m_past] + torch.log(count_past.clamp(min=1.0)).view(1, 1, 1, m_past)
        if suf_len > 1:
            causal = torch.triu(torch.ones((suf_len, suf_len), device=device, dtype=torch.bool), diagonal=1)
            logits[..., m_past:] = logits[..., m_past:].masked_fill(causal, float("-inf"))

        weights = torch.softmax(logits, dim=-1)
        out_heads = torch.matmul(weights.to(dtype=v_all.dtype), v_all)
        out_heads = out_heads.transpose(1, 2).contiguous().reshape(1, suf_len, num_heads * head_dim)
        s_out = attn.o_proj(out_heads).float()

        mse = F.mse_loss(s_out, t_out)
        cos = 1.0 - F.cosine_similarity(s_out, t_out, dim=-1).mean()
        layer_loss = (mse + 0.5 * cos) / n_layers
        layer_loss.backward()
        total_loss_val += float(layer_loss.item())

    torch.nn.utils.clip_grad_norm_(model_mergers.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss_val


def e2e_kl_distill_step(
    model,
    model_mergers: ModelMergers,
    input_ids: torch.Tensor,
    prefix_len: int,
    cfg: CacheConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    temperature: float = 1.0,
) -> float:
    """End-to-end suffix KL(P_teacher || P_student) distillation with frozen LLM backbone."""
    seq_len = input_ids.shape[1]
    with torch.no_grad():
        teacher_logits = model(input_ids=input_ids, use_cache=False).logits[:, prefix_len:, :].float()
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    optimizer.zero_grad(set_to_none=True)
    with patched_multiscale_attention(model, cfg, model_mergers=model_mergers):
        # Differentiable prefix compression in one vectorized pass (suffix attends to compressed prefix).
        pos_pre = torch.arange(0, prefix_len, device=device, dtype=torch.long).unsqueeze(0)
        model(input_ids=input_ids[:, :prefix_len], position_ids=pos_pre, use_cache=False)

        pos_suf = torch.arange(prefix_len, seq_len, device=device, dtype=torch.long).unsqueeze(0)
        student_logits = model(input_ids=input_ids[:, prefix_len:], position_ids=pos_suf, use_cache=False).logits.float()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(
        student_log_probs.reshape(-1, student_log_probs.shape[-1]),
        teacher_probs.reshape(-1, teacher_probs.shape[-1]),
        reduction="batchmean",
    ) * (temperature ** 2)
    kl.backward()
    torch.nn.utils.clip_grad_norm_(model_mergers.parameters(), max_norm=1.0)
    optimizer.step()
    return float(kl.item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill GatedCayleyMerger for Multi-Scale KV-Cache.")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--train-mode", type=str, default="layerwise", choices=["layerwise", "e2e_kl"])
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--prefix-len", type=int, default=1536)
    p.add_argument("--suffix-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sink-size", type=int, default=4)
    p.add_argument("--local-size", type=int, default=128)
    p.add_argument("--slots-per-level", type=int, default=4)
    p.add_argument("--l-crit", type=int, default=4, help="RG Fixed-Point weight sharing start level")
    p.add_argument("--gate-rank", type=int, default=32)
    p.add_argument("--rope-mode", type=str, default="pre_rope_damped_gauss")
    p.add_argument("--save-path", type=str, default="results/gated_cayley_merger.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, backend = get_best_device(args.device)
    dtype = get_best_dtype(device, backend, args.dtype)
    print(f"[Train] model={args.model} | mode={args.train_mode} | device={device} ({backend}) | dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg_m = model.config
    num_layers = int(cfg_m.num_hidden_layers)
    num_heads = int(cfg_m.num_attention_heads)
    num_kv_heads = int(getattr(cfg_m, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg_m, "head_dim", cfg_m.hidden_size // num_heads))
    inv_freq, attention_scaling = extract_rope_params(model, head_dim)

    model_mergers = ModelMergers(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        l_crit=args.l_crit,
        gate_rank=args.gate_rank,
    ).to(device)
    model_mergers.train()

    total_params = sum(p.numel() for p in model_mergers.parameters())
    print(f"[Train] Trainable merger params: {total_params:,} (LLM frozen)")

    cache_cfg = CacheConfig(
        mode="multiscale_tree",
        sink_size=args.sink_size,
        local_size=args.local_size,
        salient_size=0,  # train pure tree mergers first; can combine with salient pool at inference!
        slots_per_level=args.slots_per_level,
        rope_mode=args.rope_mode,
        merger_mode="learned",
        use_log_bias=True,
    )

    text = load_eval_text(args.text_file, min_chars=250_000)
    total_len = args.prefix_len + args.suffix_len
    windows = build_training_windows(tokenizer, text, total_len=total_len, num_samples=args.steps)

    optimizer = torch.optim.AdamW(model_mergers.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps))

    for step, seq in enumerate(windows, start=1):
        seq = seq.to(device)
        if args.train_mode == "layerwise":
            records = capture_teacher_layer_io(model, seq, prefix_len=args.prefix_len)
            loss = layerwise_distill_step(
                model, model_mergers, records, cache_cfg, inv_freq, attention_scaling, device, dtype, optimizer
            )
        else:
            loss = e2e_kl_distill_step(
                model, model_mergers, seq, args.prefix_len, cache_cfg, device, optimizer
            )
        scheduler.step()
        if step == 1 or step % 5 == 0 or step == args.steps:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step:3d}/{args.steps} | loss={loss:.5f} | lr={lr_now:.2e}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_mergers.eval()
    model_mergers.materialize_all()
    model_mergers.save(str(save_path))
    print(f"\n[Train] Saved trained mergers to {save_path}")


if __name__ == "__main__":
    main()

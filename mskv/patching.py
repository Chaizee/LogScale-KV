"""Version-resilient HuggingFace attention patcher for Qwen2 / Qwen2.5 / Qwen3 / Llama / Mistral.

Bypasses HF DynamicCache API instability across transformers versions by managing
MultiScaleKVCache directly inside each attention layer when use_cache=False.
"""

from __future__ import annotations

import contextlib
import inspect
import math
from typing import Iterator, Optional

import torch

from .cache import CacheConfig, MultiScaleKVCache
from .mergers import ModelMergers
from .rope import apply_multiscale_rope, apply_rope_1d, extract_rope_params


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h_kv, s, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, h_kv, n_rep, s, d)
        .reshape(b, h_kv * n_rep, s, d)
    )


def _detect_attn_return_arity(decoder_layer) -> int:
    """Detect whether DecoderLayer.forward expects 2 or 3 outputs from self.self_attn."""
    try:
        src = inspect.getsource(decoder_layer.forward)
        for line in src.splitlines():
            if "self.self_attn(" in line and "=" in line:
                lhs = line.split("=", 1)[0]
                n_vars = len([v for v in lhs.split(",") if v.strip()])
                if n_vars in (2, 3):
                    return n_vars
    except Exception:
        pass

    import transformers

    parts = transformers.__version__.split(".")
    try:
        ver = (int(parts[0]), int(parts[1]))
        return 2 if ver >= (4, 48) else 3
    except Exception:
        return 2


def _make_patched_forward(
    attn_module,
    layer_idx: int,
    cache: MultiScaleKVCache,
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
    cfg = cache.cfg

    def patched_forward(
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

        # 1. Project Q, K (pre-RoPE), V using the pretrained model's exact linear layers.
        q = attn_module.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        k_pre = attn_module.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        v = attn_module.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Optional QK-Norm support (e.g. Qwen3 / Gemma2 variants).
        if hasattr(attn_module, "q_norm") and attn_module.q_norm is not None:
            q = attn_module.q_norm(q)
        if hasattr(attn_module, "k_norm") and attn_module.k_norm is not None:
            k_pre = attn_module.k_norm(k_pre)

        # Determine exact token positions for this chunk.
        if position_ids is not None:
            chunk_pos = position_ids[0].to(device=device, dtype=torch.float32)
        elif cache_position is not None:
            chunk_pos = cache_position.to(device=device, dtype=torch.float32)
        else:
            start = layer_cache.total_tokens_seen
            chunk_pos = torch.arange(start, start + q_len, device=device, dtype=torch.float32)

        # 2. Apply RoPE to queries and current-chunk keys.
        q_rot = apply_rope_1d(q, chunk_pos, inv_freq, attention_scaling=attention_scaling)
        k_cur_rot = apply_rope_1d(k_pre, chunk_pos, inv_freq, attention_scaling=attention_scaling)

        # 3. Fetch past compressed cache and assemble full K, V for attention.
        k_past, v_past, pos_past, count_past, slices = layer_cache.get_past_tensors()
        if k_past is not None:
            m_past = k_past.shape[2]
            k_past_rot = apply_multiscale_rope(
                k_past,
                pos_past,
                count_past,
                inv_freq,
                attention_scaling=attention_scaling,
                mode=cfg.rope_mode,
            )
            k_all = torch.cat((k_past_rot, k_cur_rot), dim=2)
            v_all = torch.cat((v_past, v), dim=2)
        else:
            m_past = 0
            k_all = k_cur_rot
            v_all = v

        k_rep = repeat_kv(k_all, num_kv_groups)
        v_rep = repeat_kv(v_all, num_kv_groups)

        # 4. Scaled Dot-Product Attention in float32 for numerical stability on ROCm / DirectML.
        attn_logits = torch.matmul(q_rot.float(), k_rep.float().transpose(-2, -1)) * scale

        # Proportional Attention bias: +log(count_i) = l * ln(2) on compressed blocks.
        if m_past > 0 and cfg.use_log_bias:
            log_bias = torch.log(count_past.clamp(min=1.0)).view(1, 1, 1, m_past)
            attn_logits[..., :m_past] = attn_logits[..., :m_past] + log_bias

        # Causal mask within the current chunk.
        if q_len > 1:
            causal = torch.triu(
                torch.ones((q_len, q_len), device=device, dtype=torch.bool),
                diagonal=1,
            )
            attn_logits[..., m_past:] = attn_logits[..., m_past:].masked_fill(causal, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_output = torch.matmul(attn_weights.to(dtype=v_rep.dtype), v_rep)

        # 5. Update salience scores and append chunk to streaming cache.
        with torch.no_grad():
            # Max across query heads, sum over query positions in chunk -> (m_past + q_len,)
            sal_total = attn_weights[0].sum(dim=1).max(dim=0).values
            sal_past = sal_total[:m_past] if m_past > 0 else None
            sal_cur = sal_total[m_past:]

        layer_cache.apply_salience_feedback(sal_past, slices)
        k_to_store = k_cur_rot if cfg.rope_mode == "post_rope" else k_pre
        layer_cache.append_chunk(k_to_store, v, chunk_pos, attn_salience_chunk=sal_cur)

        # 6. Output projection.
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, num_heads * head_dim)
        attn_output = attn_module.o_proj(attn_output)

        returned_weights = attn_weights if output_attentions else None
        if return_arity == 3:
            return attn_output, returned_weights, None
        return attn_output, returned_weights

    return patched_forward


@contextlib.contextmanager
def patched_multiscale_attention(
    model,
    config: CacheConfig,
    model_mergers: Optional[ModelMergers] = None,
) -> Iterator[MultiScaleKVCache]:
    """Context manager that temporarily patches a CausalLM to use MultiScaleKVCache."""
    base_model = getattr(model, "model", model)
    layers = base_model.layers
    num_layers = len(layers)

    cfg_m = model.config
    num_heads = int(cfg_m.num_attention_heads)
    num_kv_heads = int(getattr(cfg_m, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg_m, "head_dim", cfg_m.hidden_size // num_heads))
    inv_freq, attention_scaling = extract_rope_params(model, head_dim)
    return_arity = _detect_attn_return_arity(layers[0])

    layer_mergers = list(model_mergers.layers) if model_mergers is not None else None
    cache = MultiScaleKVCache(num_layers=num_layers, config=config, layer_mergers=layer_mergers)

    orig_forwards = []
    try:
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            orig_forwards.append((attn, attn.forward))
            attn.forward = _make_patched_forward(
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

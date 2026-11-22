"""AMD-friendly device selection, dtype defaults, and safe linear algebra helpers."""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def get_best_device(requested: str = "auto") -> Tuple[torch.device, str]:
    """Select the best available device with AMD ROCm and DirectML support.

    Note: PyTorch built with ROCm exposes AMD GPUs under the standard ``cuda`` backend.
    On Windows without ROCm, ``torch_directml`` can optionally be used.
    """
    req = requested.lower().strip()
    if req != "auto":
        if req == "dml" or req == "directml":
            import torch_directml  # type: ignore

            return torch_directml.device(), "directml"
        return torch.device(req), req

    if torch.cuda.is_available():
        backend = "rocm" if getattr(torch.version, "hip", None) is not None else "cuda"
        name = torch.cuda.get_device_name(0)
        logger.info("Using %s device: %s", backend.upper(), name)
        return torch.device("cuda"), backend

    try:
        import torch_directml  # type: ignore

        dev = torch_directml.device()
        logger.info("Using DirectML device: %s", dev)
        return dev, "directml"
    except ImportError:
        pass

    logger.info("No GPU backend detected; falling back to CPU.")
    return torch.device("cpu"), "cpu"


def get_best_dtype(device: torch.device, backend: str, requested: str = "auto") -> torch.dtype:
    """Pick a numerically safe dtype for ROCm, DirectML, CUDA, or CPU."""
    req = requested.lower().strip()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if req != "auto":
        if req not in mapping:
            raise ValueError(f"Unsupported dtype '{requested}'. Choose from auto, fp32, fp16, bf16.")
        return mapping[req]

    if backend in ("rocm", "cuda"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    if backend == "directml":
        # DirectML frequently has limited bf16/fp16 op coverage; fp16 works for matmuls on newer builds,
        # but fp32 is the safest default unless explicitly overridden via --dtype fp16.
        return torch.float32

    return torch.float32


def safe_cayley_orthogonal(skew_param: torch.Tensor) -> torch.Tensor:
    """Compute strictly orthogonal matrix Q = (I - A)(I + A)^{-1} from parameter P.

    Args:
        skew_param: Tensor of shape (..., D, D). Skew-symmetric part A = 0.5 * (P - P^T) is used.
                    At P = 0, returns exact identity I.

    Returns:
        Orthogonal tensor Q of shape (..., D, D) in float32.
    """
    orig_device = skew_param.device
    p = skew_param.float()
    a = 0.5 * (p - p.transpose(-1, -2))
    d = a.shape[-1]
    eye = torch.eye(d, dtype=torch.float32, device="cpu").to(a.device).expand_as(a)

    # Since (I + A) and (I - A) commute, (I + A)^{-1}(I - A) == (I - A)(I + A)^{-1}.
    try:
        return torch.linalg.solve(eye + a, eye - a)
    except RuntimeError:
        # Fallback for backends (e.g., DirectML) where batched linalg.solve may be unsupported.
        a_cpu = a.detach().cpu() if not a.requires_grad else a.cpu()
        eye_cpu = torch.eye(d, dtype=torch.float32).expand_as(a_cpu)
        q_cpu = torch.linalg.solve(eye_cpu + a_cpu, eye_cpu - a_cpu)
        return q_cpu.to(orig_device)


def get_peak_vram_mb(backend: str) -> float:
    """Return peak allocated GPU memory in MB when supported by the backend."""
    if backend in ("rocm", "cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    return 0.0


def reset_peak_vram(backend: str) -> None:
    if backend in ("rocm", "cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

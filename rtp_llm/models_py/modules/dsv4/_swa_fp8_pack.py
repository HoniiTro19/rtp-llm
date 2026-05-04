"""Per-row FP8 e4m3 SWA pack/unpack helpers.

TEMP -- pending real fp8 attention rewrite, see plan section 1.1.

This is a hack to unblock the v4_flash_tp1_dp1_ep1_sm100_fp8 smoke. The
framework SWA_KV pool is allocated as ``uint8 [num_slots, 584]`` when the
FP8 KV cache layout is enabled; ``write_kv_to_pool`` requires src and
pool to share their last dim (it dtype-casts but does not reshape), so
we round-trip the bf16 SWA source through a per-row absmax FP8 e4m3
quant + dequant on the SWA write/read sites only. CSA/HCA/INDEXER are
out of scope for this hack -- INDEXER already has a real Triton FP8
quant kernel and the others are not on the immediate crash path.

Slot byte layout (584 B per slot):
  bytes [0  : 512] -- fp8 e4m3 NoPE (512 elements * 1 B)
  bytes [512: 576] -- zero-pad (was rope; SWA source carries no rope)
  bytes [576: 580] -- per-slot fp32 scale
  bytes [580: 584] -- zero-pad

Quant: per-row absmax; scale = max(absmax, eps) / 448.0; value
quant = clamp(round(x/scale), -448, 448).to(float8_e4m3fn). Dequant
multiplies by scale and casts back to bf16.
"""

from __future__ import annotations

import torch

_SLOT_BYTES = 584
_NOPE_BYTES = 512  # head_dim, 1 B per fp8 element
_SCALE_OFF = 576  # fp32 scale offset
_FP8_MAX = 448.0  # float8_e4m3fn dynamic-range max
_EPS = 1e-6


def pack_swa_fp8(src_bf16: torch.Tensor) -> torch.Tensor:
    """Per-row FP8 e4m3 quant of an SWA source row.

    TEMP -- pending real fp8 attention rewrite, see plan section 1.1.

    Args:
        src_bf16: ``[N, 512]`` (bf16 or any float dtype). Empty N=0 OK.
    Returns:
        ``[N, 584]`` uint8 slot bytes laid out as described in the module
        docstring. ``[N, 512]`` head_dim assumption is hard-coded.
    """
    assert (
        src_bf16.dim() == 2 and src_bf16.shape[1] == _NOPE_BYTES
    ), f"pack_swa_fp8 expects [N, {_NOPE_BYTES}], got {tuple(src_bf16.shape)}"
    n = src_bf16.shape[0]
    device = src_bf16.device
    out = torch.zeros((n, _SLOT_BYTES), dtype=torch.uint8, device=device)
    if n == 0:
        return out

    src_f32 = src_bf16.to(torch.float32)
    absmax = src_f32.abs().amax(dim=1, keepdim=True)  # [N, 1]
    scale = torch.clamp(absmax, min=_EPS) / _FP8_MAX  # [N, 1]
    q = (src_f32 / scale).clamp(min=-_FP8_MAX, max=_FP8_MAX)
    q_fp8 = q.to(torch.float8_e4m3fn)  # [N, 512]
    q_bytes = q_fp8.view(torch.uint8)  # [N, 512]

    out[:, 0:_NOPE_BYTES] = q_bytes
    # bytes [512:576] left zero (rope pad).
    scale_bytes = scale.to(torch.float32).contiguous().view(torch.uint8).view(n, 4)
    out[:, _SCALE_OFF : _SCALE_OFF + 4] = scale_bytes
    # bytes [580:584] left zero.
    return out


def unpack_swa_fp8(slots_uint8: torch.Tensor) -> torch.Tensor:
    """Dequant FP8 SWA slots back to bf16.

    TEMP -- pending real fp8 attention rewrite, see plan section 1.1.

    Args:
        slots_uint8: ``[N, 584]`` uint8 slot bytes from the pool.
    Returns:
        ``[N, 512]`` bf16. Empty N=0 OK.
    """
    assert (
        slots_uint8.dim() == 2 and slots_uint8.shape[1] == _SLOT_BYTES
    ), f"unpack_swa_fp8 expects [N, {_SLOT_BYTES}], got {tuple(slots_uint8.shape)}"
    n = slots_uint8.shape[0]
    device = slots_uint8.device
    if n == 0:
        return torch.empty((0, _NOPE_BYTES), dtype=torch.bfloat16, device=device)

    nope_bytes = slots_uint8[:, 0:_NOPE_BYTES].contiguous()
    q_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # [N, 512]
    val_f32 = q_fp8.to(torch.float32)

    scale_bytes = slots_uint8[:, _SCALE_OFF : _SCALE_OFF + 4].contiguous()
    scale_f32 = scale_bytes.view(torch.float32).view(n, 1)  # [N, 1]

    out = (val_f32 * scale_f32).to(torch.bfloat16)
    return out

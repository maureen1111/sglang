"""SM120 BF16 batched GEMM for GLM MLA absorb/unabsorb projections.

The stock torch.bmm path in CUDA 13 selects an SM80 WMMA kernel for these
shapes.  This Triton kernel preserves BF16 inputs/output and FP32 accumulation,
and is bitwise identical to torch.bmm for the validated GLM-5.2 shapes.
"""

import torch
import triton
import triton.language as tl


_VALID_SHAPES = {(192, 512), (512, 256)}
_logged_shape = set()


@triton.jit
def _sm120_bmm_bf16_kernel(
    a,
    b,
    out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_oh,
    stride_om,
    stride_on,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = (
        a
        + head * stride_ah
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b
        + head * stride_bh
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        av = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            b_ptrs,
            mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(av, bv, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    out_ptrs = (
        out
        + head * stride_oh
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def can_use_sm120_absorb_bmm(a: torch.Tensor, b: torch.Tensor) -> bool:
    if not (a.is_cuda and b.is_cuda and a.dtype == b.dtype == torch.bfloat16):
        return False
    if a.ndim != 3 or b.ndim != 3:
        return False
    if a.shape[0] != 64 or b.shape[0] != 64 or a.shape[2] != b.shape[1]:
        return False
    if (a.shape[2], b.shape[2]) not in _VALID_SHAPES:
        return False
    return a.stride(2) == 1 and (b.stride(1) == 1 or b.stride(2) == 1)


def sm120_absorb_bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    h, m, k = a.shape
    n = b.shape[2]
    shape = (h, k, n)
    if shape not in _logged_shape:
        print(f"Enabled SM120 BF16 absorb BMM for H/K/N={shape}", flush=True)
        _logged_shape.add(shape)
    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 128), h)
    _sm120_bmm_bf16_kernel[grid](
        a,
        b,
        out,
        m,
        N=n,
        K=k,
        stride_ah=a.stride(0),
        stride_am=a.stride(1),
        stride_ak=a.stride(2),
        stride_bh=b.stride(0),
        stride_bk=b.stride(1),
        stride_bn=b.stride(2),
        stride_oh=out.stride(0),
        stride_om=out.stride(1),
        stride_on=out.stride(2),
        BM=64,
        BN=128,
        BK=64,
        num_warps=8,
        num_stages=3,
    )


@triton.jit
def _copy_q_rope_tail_kernel(
    q_rope,
    q_all,
    num_items,
    rope_dim: tl.constexpr,
    out_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_items * rope_dim
    item = offsets // rope_dim
    column = offsets - item * rope_dim
    value = tl.load(q_rope + offsets, mask=mask)
    tl.store(q_all + item * out_dim + (out_dim - rope_dim) + column, value, mask=mask)


def can_use_sm120_absorb_bmm_preconcat(
    a: torch.Tensor, b: torch.Tensor, q_rope: torch.Tensor
) -> bool:
    return (
        can_use_sm120_absorb_bmm(a, b)
        and a.shape[2] == 192
        and b.shape[2] == 512
        and q_rope.is_cuda
        and q_rope.dtype == torch.bfloat16
        and q_rope.ndim == 3
        and q_rope.shape == (a.shape[1], a.shape[0], 64)
        and q_rope.is_contiguous()
    )


def sm120_absorb_bmm_preconcat(
    a: torch.Tensor, b: torch.Tensor, q_rope: torch.Tensor
) -> torch.Tensor:
    """Write the absorbed Q and RoPE tail directly into one contiguous Q tensor.

    The regular path materializes a contiguous [M,H,512] BMM output and then
    copies both that output and the [M,H,64] RoPE tail in a separate concat
    kernel.  The custom BMM already supports arbitrary output strides, so it
    can write the first 512 columns of [M,H,576] directly; only the 64-column
    tail still needs a copy.
    """
    heads, tokens, _ = a.shape
    q_all = torch.empty(
        (tokens, heads, 576), dtype=torch.bfloat16, device=a.device
    )
    sm120_absorb_bmm(a, b, q_all[..., :512].transpose(0, 1))
    num_items = tokens * heads
    _copy_q_rope_tail_kernel[(triton.cdiv(num_items * 64, 1024),)](
        q_rope,
        q_all,
        num_items,
        rope_dim=64,
        out_dim=576,
        BLOCK=1024,
        num_warps=8,
    )
    return q_all

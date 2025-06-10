# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.utils import input_guard

import warnings
def config_prune(configs):

    if torch.version.hip:
        try:
            # set warp size based on gcn architecure 
            gcn_arch_name = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx10" in gcn_arch_name or "gfx11" in gcn_arch_name:
                # radeon
                warp_size = 32
            else:
                # instinct
                warp_size = 64
        except AttributeError as e:
            # fall back to crude method to set warp size
            device_name = torch.cuda.get_device_properties(0).name
            if 'instinct' in device_name.lower():
                warp_size = 64
            else:
                warp_size = 32
            warnings.warn(f"{e}, warp size set to {warp_size} based on device name: {device_name}", UserWarning)

    else:
        # cuda 
        warp_size = 32    

    max_block_sz = 1024
    max_num_warps = max_block_sz // warp_size
    pruned_configs = [config for config in configs if config.num_warps <= max_num_warps]
    return pruned_configs

configs_autotune = [
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ]

pruned_configs_autotune = config_prune(configs_autotune)


@triton.autotune(
    configs=pruned_configs_autotune,
    key=['N']
)
@triton.jit
def l2norm_fwd_kernel(
    X,
    Y,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    i_m = tl.program_id(0)
    X += i_m * N
    Y += i_m * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + i_m, rstd)
    # Normalize and apply linear transformation
    y = x * rstd
    # Write output
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=pruned_configs_autotune,
    key=['N']
)
@triton.jit
def l2norm_bwd_kernel(
    X,
    DY,
    DX,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    i_m = tl.program_id(0)
    X += i_m * N
    DX += i_m * N
    DY += i_m * N

    # Y += i_m * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + i_m, rstd)
    # Normalize and apply linear transformation
    # y = x * rstd
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.where(mask, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    N = x.shape[-1]
    M = x.shape[0]
    # rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    l2norm_fwd_kernel[(M,)](
        x,
        y,
        N,
        eps,
        BLOCK_N,
    )
    return y.reshape(x_shape_og)


def l2norm_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-5
):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape
    # allocate output
    dx = torch.empty_like(x)
    M = x.shape[0]
    N = x.shape[-1]
    # rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    l2norm_bwd_kernel[(M,)](
        x,
        dy,
        dx,
        N,
        eps,
        BLOCK_N,
    )
    return dx.reshape(x_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        eps=1e-6,
        output_dtype=None
    ):
        y = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(x)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = l2norm_bwd(x, dy, ctx.eps)
        return dx, None, None


def l2_norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)

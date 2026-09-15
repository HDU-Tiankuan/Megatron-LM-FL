# Copyright (c) 2026, BAAI. All rights reserved.
"""Ascend in-place gradient scaling for Megatron's clipping call site."""

import torch


def _scale_grads(grads, clip_coeff):
    """Batch dense NPU gradients without changing norm or coefficient semantics.

    This call site aliases the input/output lists and discards the overflow
    buffer. Therefore self-copies and the scale kernel's finite scan are not
    observable. Non-finite detection in the optimizer is left unchanged.
    Unsupported inputs delegate before any mutation; kernel errors propagate.
    """
    from megatron.core.optimizer.clip_grads import _scale_grads as original

    if not grads:
        return original.__wrapped__(grads, clip_coeff)
    first = grads[0]
    supported = callable(getattr(torch, "_foreach_mul_", None)) and all(
        type(g) is torch.Tensor
        and g.device.type == "npu"
        and g.device == first.device
        and g.dtype == first.dtype
        and g.dtype in (torch.float32, torch.bfloat16)
        and g.layout == torch.strided
        and g.is_contiguous()
        for g in grads
    )
    if not supported:
        return original.__wrapped__(grads, clip_coeff)
    # Foreach kernels may update list entries concurrently. Preserve the
    # sequential reference semantics for duplicate or overlapping views.
    # Contiguity above makes each nonempty view a single byte interval.
    intervals = sorted((g.data_ptr(), g.numel() * g.element_size()) for g in grads if g.numel())
    if any(
        start + size > following for (start, size), (following, _) in zip(intervals, intervals[1:])
    ):
        return original.__wrapped__(grads, clip_coeff)
    if isinstance(clip_coeff, torch.Tensor):
        clip_coeff.clamp_max_(1.0)
        # Preserve the reference NPU path's host scalar conversion.
        scalar = clip_coeff.item()
    elif clip_coeff < 1.0:
        scalar = clip_coeff
    else:
        return None
    torch._foreach_mul_(grads, scalar)

"""Experimental Adaptive Gradient Recovery (AGR) for PyTorch.

Forward: tanh(clamp(z, -limit, limit)), identical to the hard-clamp baseline.
Backward: surrogate gradient allows only inward-directed recovery outside the
clamp. This is NOT the mathematical derivative of the forward operation.

Status: experimental; not enabled in the production PAW core.
"""
import torch


class _AGR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, limit):
        ctx.save_for_backward(z)
        ctx.limit = float(limit)
        return torch.tanh(z.clamp(-ctx.limit, ctx.limit))

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        limit = ctx.limit
        clipped = z.clamp(-limit, limit)
        g = grad_output * (1.0 - torch.tanh(clipped).square())
        # An SGD descent step is -g: positive g moves z inward above +limit,
        # and negative g moves z inward below -limit.
        outward = ((z > limit) & (g < 0)) | ((z < -limit) & (g > 0))
        return torch.where(outward, torch.zeros_like(g), g), None


def adaptive_gradient_recovery(z, limit=1.25):
    """Apply experimental AGR to a tensor of preactivations."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return _AGR.apply(z, float(limit))


class AdaptiveGradientRecovery(torch.nn.Module):
    def __init__(self, limit=1.25):
        super().__init__()
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = float(limit)

    def forward(self, z):
        return adaptive_gradient_recovery(z, self.limit)

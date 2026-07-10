"""
LoRA (Low-Rank Adaptation) implemented from scratch, no PEFT/Tinker library.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear. The base weight is frozen; only the
    small A/B matrices are trainable.

    output = x @ W_frozen + scaling * (x @ A^T @ B^T)
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        # A: random init. B: zero init -> LoRA starts as a no-op.
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * (1 / math.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        # Two small matmuls through the rank-r bottleneck. B@A is NEVER
        # materialized as a full (out x in) matrix during training.
        h = self.dropout(x) @ self.lora_A.T          # (..., in) -> (..., r)
        lora_out = h @ self.lora_B.T                  # (..., r) -> (..., out)
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold LoRA into a plain nn.Linear (one-time matmul, done once)."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                            bias=self.base.bias is not None)
        delta = self.scaling * (self.lora_B @ self.lora_A)   # built ONCE, here
        merged.weight.copy_(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def inject_lora(model: nn.Module, target_names=("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
                 r: int = 8, alpha: int = 16):
    """Walk the model, swap named nn.Linear children for LoRALinear wrappers."""
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in target_names:
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
    return model


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

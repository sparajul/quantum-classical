"""models/classical_mlp.py — Classical MLP matching Acorn's make_mlp style."""
from __future__ import annotations

from typing import List, Optional

import torch.nn as nn


def make_classical_mlp(
    input_size: int,
    sizes: List[int],
    hidden_activation: str = "ReLU",
    output_activation: Optional[str] = None,
    layernorm: bool = False,
    batchnorm: bool = False,
    output_layer_norm: bool = False,
    output_batch_norm: bool = False,
    track_running_stats: bool = False,
    input_dropout: float = 0.0,
    hidden_dropout: float = 0.0,
    **kwargs,
) -> nn.Sequential:
    """
    Classical MLP — Acorn-style norm placement.

    Layer order: Linear → [LayerNorm|BatchNorm] → Activation → [Dropout]
    Norm is placed BEFORE the activation (pre-activation / pre-norm style),
    which stabilises gradient flow vs. the post-activation placement.

    LayerNorm uses elementwise_affine=False (non-learnable scale/shift).
    BatchNorm uses eps=6e-5, track_running_stats=False (Acorn defaults).
    """
    if not sizes:
        raise ValueError("sizes cannot be empty")

    hidden_act_cls = getattr(nn, hidden_activation, None)
    if hidden_act_cls is None:
        raise ValueError(f"Unknown hidden_activation: '{hidden_activation}'")

    output_act_cls = None
    if output_activation:
        output_act_cls = getattr(nn, output_activation, None)
        if output_act_cls is None:
            raise ValueError(f"Unknown output_activation: '{output_activation}'")

    n_layers = len(sizes)
    all_sizes = [input_size] + list(sizes)

    layers: List[nn.Module] = []

    # Hidden layers (all but the last)
    for i in range(n_layers - 1):
        if i == 0 and input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        layers.append(nn.Linear(all_sizes[i], all_sizes[i + 1]))
        if layernorm:
            layers.append(nn.LayerNorm(all_sizes[i + 1], elementwise_affine=False))
        if batchnorm:
            layers.append(nn.BatchNorm1d(
                all_sizes[i + 1],
                eps=6e-5,
                track_running_stats=track_running_stats,
                affine=True,
            ))
        layers.append(hidden_act_cls())
        if hidden_dropout > 0:
            layers.append(nn.Dropout(hidden_dropout))

    # Output layer
    layers.append(nn.Linear(all_sizes[-2], all_sizes[-1]))
    if output_act_cls is not None:
        if output_layer_norm:
            layers.append(nn.LayerNorm(all_sizes[-1], elementwise_affine=False))
        if output_batch_norm:
            layers.append(nn.BatchNorm1d(
                all_sizes[-1],
                eps=6e-5,
                track_running_stats=track_running_stats,
                affine=True,
            ))
        layers.append(output_act_cls())

    return nn.Sequential(*layers)

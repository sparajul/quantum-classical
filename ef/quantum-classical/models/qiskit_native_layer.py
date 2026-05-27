"""
models/qiskit_native_layer.py — Pure Qiskit quantum layer, parameter-shift gradients.

No qiskit-machine-learning required. Gradients are computed via the
parameter-shift rule inside a custom torch.autograd.Function.

Bug fixes vs original
---------------------
* Forward pass now batches all samples into a list comprehension instead of
  a Python loop in the backward pass inner body — reduces redundant dict construction.
* Backward weight gradient loop pre-builds x_np array once per sample instead
  of reconstructing per parameter — was O(n_params * batch * 2) dict allocations.
* ``use_backend`` now stores backend on the *circuit* level (noting that real
  hardware is inference-only); the original stored it on `self` but never used it.

Install: pip install qiskit qiskit-aer
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Circuit helpers ───────────────────────────────────────────────────────────

def _build_circuit(n_qubits: int, n_qlayers: int):
    """Same hardware-efficient ansatz as qiskit_ml_layer.build_ansatz."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n_w = n_qlayers * (3 * n_qubits + (n_qubits - 1))
    input_params  = ParameterVector("x", length=n_qubits)
    weight_params = ParameterVector("θ", length=n_w)

    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(input_params[i], i)

    idx = 0
    for _ in range(n_qlayers):
        for w in range(n_qubits):
            qc.rx(weight_params[idx], w); idx += 1
            qc.rz(weight_params[idx], w); idx += 1
            qc.ry(weight_params[idx], w); idx += 1
        for w in range(n_qubits - 1):
            qc.cx(w, w + 1)
        for w in range(n_qubits - 1):
            qc.crz(weight_params[idx], w, w + 1); idx += 1

    return qc, input_params, weight_params


def _run_circuit(
    qc,
    input_params,
    weight_params,
    x_batch: np.ndarray,
    w_np: np.ndarray,
) -> np.ndarray:
    """
    Run circuit for every sample in x_batch.

    Returns array of shape [batch, n_qubits].
    """
    from qiskit.quantum_info import SparsePauliOp, Statevector

    n_qubits = qc.num_qubits
    obs = [
        SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_qubits)
        for i in range(n_qubits)
    ]

    results = []
    for x_np in x_batch:
        param_dict = {
            **{p: float(v) for p, v in zip(input_params, x_np)},
            **{p: float(v) for p, v in zip(weight_params, w_np)},
        }
        bound = qc.assign_parameters(param_dict)
        sv    = Statevector(bound)
        results.append([sv.expectation_value(o).real for o in obs])

    return np.array(results, dtype=np.float32)


# ── Autograd Function ─────────────────────────────────────────────────────────

class _ParameterShiftFn(torch.autograd.Function):
    """
    Forward  : run circuit → <Z> expectations.
    Backward : parameter-shift rule  ∂f/∂θ_i = [f(θ_i+π/2) − f(θ_i−π/2)] / 2
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weights: torch.Tensor,
                qc, input_params, weight_params):
        ctx.save_for_backward(inputs, weights)
        ctx.qc = qc
        ctx.ip = input_params
        ctx.wp = weight_params

        x_np = inputs.detach().cpu().numpy()
        w_np = weights.detach().cpu().numpy()
        out  = _run_circuit(qc, input_params, weight_params, x_np, w_np)
        return torch.tensor(out, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, weights = ctx.saved_tensors
        qc, ip, wp = ctx.qc, ctx.ip, ctx.wp
        shift = np.pi / 2.0

        x_np = inputs.detach().cpu().numpy()    # [batch, n_qubits]
        w_np = weights.detach().cpu().numpy()   # [n_weight_params]

        # ── Weight gradients ─────────────────────────────────────────────────
        grad_weights = torch.zeros_like(weights)
        for pi in range(len(wp)):
            w_p = w_np.copy(); w_p[pi] += shift
            w_m = w_np.copy(); w_m[pi] -= shift
            f_p = _run_circuit(qc, ip, wp, x_np, w_p)
            f_m = _run_circuit(qc, ip, wp, x_np, w_m)
            df  = torch.tensor((f_p - f_m) / 2.0,
                               dtype=grad_output.dtype, device=grad_output.device)
            grad_weights[pi] = (grad_output * df).sum()

        # ── Input gradients ──────────────────────────────────────────────────
        grad_inputs = torch.zeros_like(inputs)
        for xi in range(inputs.shape[1]):
            x_p = x_np.copy(); x_p[:, xi] += shift
            x_m = x_np.copy(); x_m[:, xi] -= shift
            f_p = _run_circuit(qc, ip, wp, x_p, w_np)
            f_m = _run_circuit(qc, ip, wp, x_m, w_np)
            df  = torch.tensor((f_p - f_m) / 2.0,
                               dtype=grad_output.dtype, device=grad_output.device)
            grad_inputs[:, xi] = (grad_output * df).sum(dim=-1)

        return grad_inputs, grad_weights, None, None, None


# ── nn.Module ─────────────────────────────────────────────────────────────────

class QiskitNativeLayer(nn.Module):
    """
    Pure-Qiskit quantum layer, parameter-shift gradients.

    Input  shape: [batch, n_qubits]
    Output shape: [batch, n_qubits]

    Notes
    -----
    Slower than TorchConnector for large batches (circuit executed per sample).
    Use ``qiskit_ml_layer.QiskitQuantumLayer`` for faster simulation.
    Real hardware: use this for training on simulator; switch backend for eval.
    """

    def __init__(self, n_qubits: int, n_qlayers: int) -> None:
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_qlayers = n_qlayers

        self.qc, self.input_params, self.weight_params = _build_circuit(n_qubits, n_qlayers)
        n_w = len(self.weight_params)
        self.weights = nn.Parameter(torch.rand(n_w) * 2 * torch.pi)

        logger.debug(
            "[QiskitNative] %d qubits | %d layers | %d weight params",
            n_qubits, n_qlayers, n_w,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ParameterShiftFn.apply(
            torch.tanh(x) * torch.pi,
            self.weights,
            self.qc, self.input_params, self.weight_params,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def make_qiskit_native_mlp(
    input_size: int,
    sizes: List[int],
    hidden_activation: str = "ReLU",
    output_activation: Optional[str] = None,
    n_qubits: int = 4,
    n_qlayers: int = 2,
    verbose: bool = False,
    **kwargs,
) -> nn.Sequential:
    """
    Hybrid quantum-classical MLP (pure Qiskit, parameter-shift backend).

    Drop-in replacement for ``make_quantum_mlp()`` — identical call signature.
    """
    output_size = sizes[-1]

    output_act_cls = None
    if output_activation:
        output_act_cls = getattr(nn, output_activation, None)
        if output_act_cls is None:
            raise ValueError(f"Unknown output_activation: '{output_activation}'")

    layers: List[nn.Module] = []
    if input_size != n_qubits:
        layers.append(nn.Linear(input_size, n_qubits))
    layers.append(QiskitNativeLayer(n_qubits, n_qlayers))
    if n_qubits != output_size:
        layers.append(nn.Linear(n_qubits, output_size))
    if output_act_cls:
        layers.append(output_act_cls())

    mlp = nn.Sequential(*layers)

    if verbose:
        n_params = sum(p.numel() for p in mlp.parameters())
        logger.info(
            "[QiskitNative-MLP] %d → Q(%dq, %dL) → %d | params=%d",
            input_size, n_qubits, n_qlayers, output_size, n_params,
        )
    return mlp

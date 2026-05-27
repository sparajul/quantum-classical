"""
models/qiskit_ml_layer.py — Qiskit quantum layer via TorchConnector.

Uses ``qiskit-machine-learning`` EstimatorQNN wrapped with TorchConnector,
giving standard autograd support without manual parameter-shift code.

Install: pip install qiskit qiskit-machine-learning
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared ansatz (used by both Qiskit backends)
# ─────────────────────────────────────────────────────────────────────────────

def build_ansatz(n_qubits: int, n_qlayers: int):
    """
    Hardware-efficient ansatz:
      encoding  : RY(x_i) per qubit
      per layer : RX+RZ+RY rotations → CNOT ladder → CRZ pairs

    Returns (circuit, input_params, weight_params).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n_crz = n_qubits - 1
    n_weight_params = n_qlayers * (3 * n_qubits + n_crz)

    input_params  = ParameterVector("x", length=n_qubits)
    weight_params = ParameterVector("θ", length=n_weight_params)

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


def build_estimator_qnn(n_qubits: int, n_qlayers: int):
    """Wrap the ansatz in an EstimatorQNN measuring <Z> on each qubit."""
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    qc, input_params, weight_params = build_ansatz(n_qubits, n_qlayers)

    observables = [
        SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n_qubits)
        for i in range(n_qubits)
    ]

    return EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params.params,
        weight_params=weight_params.params,
        estimator=StatevectorEstimator(),
        input_gradients=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# nn.Module
# ─────────────────────────────────────────────────────────────────────────────

class QiskitQuantumLayer(nn.Module):
    """
    Qiskit quantum layer via TorchConnector.

    Encodes inputs via tanh(x)*π → RY rotations, applies n_qlayers of
    trainable rotations + entanglement, returns <Z> expectation values.

    Input  shape: [batch, n_qubits]
    Output shape: [batch, n_qubits]
    """

    def __init__(self, n_qubits: int, n_qlayers: int) -> None:
        super().__init__()
        from qiskit_machine_learning.connectors import TorchConnector

        self.n_qubits  = n_qubits
        self.n_qlayers = n_qlayers
        self.qnn = TorchConnector(build_estimator_qnn(n_qubits, n_qlayers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(torch.tanh(x) * torch.pi)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_qiskit_ml_mlp(
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
    Hybrid quantum-classical MLP (Qiskit-ML / TorchConnector backend).

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
    layers.append(QiskitQuantumLayer(n_qubits, n_qlayers))
    if n_qubits != output_size:
        layers.append(nn.Linear(n_qubits, output_size))
    if output_act_cls:
        layers.append(output_act_cls())

    mlp = nn.Sequential(*layers)

    if verbose:
        n_params = sum(p.numel() for p in mlp.parameters())
        logger.info(
            "[QiskitML-MLP] %d → Q(%dq, %dL) → %d | params=%d",
            input_size, n_qubits, n_qlayers, output_size, n_params,
        )
    return mlp

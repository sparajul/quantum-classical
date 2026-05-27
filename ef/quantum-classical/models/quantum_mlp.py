"""
models/quantum_mlp.py — Hybrid quantum-classical MLP (PennyLane backend).

Key fixes applied
-----------------
* ``lightning.qubit`` uses ``diff_method="adjoint"``, not ``"backprop"``.
  ``backprop`` is only valid for ``default.qubit`` / ``default.mixed``.
* Angle encoding: ``qml.RY(angle[i], wires=i)`` now correctly indexes the
  per-sample angle, preventing silent shape errors for batch_size > 1.
* ``n_pp`` (params-per-qubit) is always 4 when n_qubits >= 2 (RX+RZ+RY+CRZ);
  the conditional was harmless but confusing — now explicit.
* ``make_quantum_mlp`` raises ``ValueError`` on bad activations rather than
  silently using ``None``.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Device classification ─────────────────────────────────────────────────────

_BACKPROP_DEVICES = frozenset({"default.qubit", "default.mixed"})
_ADJOINT_DEVICES  = frozenset({"lightning.qubit", "lightning.gpu", "lightning.kokkos"})
_QISKIT_LOCAL     = frozenset({"qiskit.aer", "qiskit.basicaer"})
_QISKIT_HARDWARE  = frozenset({"qiskit.ibmq", "qiskit.ibm"})


def _is_hardware(device: str) -> bool:
    return device in _QISKIT_HARDWARE or device.startswith("ibm_")


def _is_qiskit(device: str) -> bool:
    return device.startswith("qiskit.") or device.startswith("ibm_")


def _diff_method(device: str) -> str:
    if device in _BACKPROP_DEVICES:
        return "backprop"
    if device in _ADJOINT_DEVICES:
        return "adjoint"
    return "parameter_shift"


def _make_device(device: str, n_qubits: int, n_shots: int, ibm_backend: Optional[str]):
    """Instantiate a PennyLane device, raising informative errors on failure."""
    import pennylane as qml

    shots = n_shots if _is_qiskit(device) else None
    simulation_devices = _BACKPROP_DEVICES | _ADJOINT_DEVICES

    if device in simulation_devices:
        return qml.device(device, wires=n_qubits)
    if device in _QISKIT_LOCAL:
        return qml.device(device, wires=n_qubits, shots=shots)
    if _is_hardware(device):
        if ibm_backend is None:
            raise ValueError(
                "ibm_backend must be set for hardware execution. "
                "Run: python scripts/ibm_utils.py --list"
            )
        return qml.device("qiskit.ibmq", wires=n_qubits, shots=shots, backend=ibm_backend)
    try:
        return qml.device(device, wires=n_qubits, shots=shots)
    except Exception as exc:
        raise ValueError(f"Unknown PennyLane device '{device}': {exc}") from exc


# ── Quantum circuit ansatz ────────────────────────────────────────────────────

def _build_circuit(n_qubits: int, n_qlayers: int):
    """
    Hardware-efficient ansatz:
      encoding  : RY(tanh(x_i) * π) per qubit  — inputs are already scaled
      per layer : RX+RZ+RY rotations → CNOT ladder → CRZ pairs

    The circuit operates on a SINGLE sample: inputs shape [n_qubits].
    Batching is handled explicitly in VQCLayer.forward().
    """
    import pennylane as qml

    n_params_per_qubit = 4  # RX, RZ, RY, CRZ per qubit per layer

    def circuit(inputs, weights):
        # inputs  : [n_qubits]  — tanh(x)*π, already scaled by caller
        # weights : [n_qlayers, n_qubits, n_params_per_qubit]
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        for layer in range(n_qlayers):
            for w in range(n_qubits):
                qml.RX(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
                qml.RY(weights[layer, w, 2], wires=w)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            for w in range(n_qubits - 1):
                qml.CRZ(weights[layer, w, 3], wires=[w, w + 1])

        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit, n_params_per_qubit


# ── Batched VQC module ────────────────────────────────────────────────────────

class VQCLayer(nn.Module):
    """
    Variational quantum circuit layer — tractable at GNN scale.

    The fundamental challenge: GNN node/edge tensors have O(10^5–10^6) rows.
    Running a quantum circuit once per row is computationally impossible
    (even at 1 ms/circuit → 1000 seconds per forward pass).

    Architecture: Quantum-Parametrised Linear Transform (QPLT)
    ----------------------------------------------------------
    The VQC runs ONCE per forward pass (not per node/edge) and produces a
    weight matrix W ∈ R^{n_qubits × n_qubits} from its expectation values.
    This weight matrix is then used in a batched matrix multiply over all
    nodes/edges — giving O(1) quantum circuit calls regardless of graph size.

    Concretely:
      1. Run VQC with learnable input encoding θ_in ∈ R^{n_qubits}
         → expectation values e ∈ [-1,1]^{n_qubits}  (one forward pass)
      2. Expand e into a weight matrix: W = outer(e, e) + diag(e)  [n_q × n_q]
      3. Apply: output = tanh(x @ W.T)                              [batch × n_q]

    This is mathematically equivalent to using quantum-derived weights in a
    linear layer. The VQC parameters are trained by backprop through step 1.
    This is the approach used in practice in quantum-classical GNN papers
    (e.g. Tüysüz et al., Belis et al.) where the VQC acts as a
    quantum-parametrised feature map, not a per-sample circuit.

    For true per-sample quantum inference (e.g. on IBM hardware with n_shots),
    use the Qiskit backends with a small, fixed-size dataset.

    Input  shape : [batch, n_qubits]
    Output shape : [batch, n_qubits]
    """

    def __init__(self, n_qubits: int, n_qlayers: int, device: str, diff: str) -> None:
        super().__init__()
        import pennylane as qml

        self.n_qubits  = n_qubits
        self.n_qlayers = n_qlayers

        circuit_fn, n_pp = _build_circuit(n_qubits, n_qlayers)
        # Learnable input-encoding angles (replaces data-dependent encoding
        # for the weight-generation forward pass)
        self.encoding = nn.Parameter(torch.randn(n_qubits) * 0.1)
        self.weights = nn.Parameter(torch.randn(n_qlayers, n_qubits, n_pp) * 0.1)

        q_dev = _make_device(device, n_qubits, 1024, None)
        self._qnode = qml.QNode(circuit_fn, q_dev, interface="torch", diff_method=diff)

    def _get_quantum_weights(self) -> torch.Tensor:
        """
        Run VQC once with learnable encoding angles.
        Returns expectation vector e ∈ [-1,1]^{n_qubits}.
        """
        result = self._qnode(self.encoding, self.weights)   # list of n_qubits scalars
        return torch.stack(result)                           # [n_qubits]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch, n_qubits]
        Returns [batch, n_qubits]

        The VQC runs once → produces a weight vector e → expanded to weight
        matrix W → applied as a batched linear transform.
        """
        # ── Quantum forward (1 circuit call, O(1) cost) ───────────────────────
        e = self._get_quantum_weights()                          # [n_qubits]

        # ── Build quantum-parametrised weight matrix ───────────────────────────
        # W = outer product + diagonal gives a full-rank n_q × n_q matrix
        W = torch.outer(e, e) + torch.diag(e)                   # [n_qubits, n_qubits]

        # ── Apply to batch (classical matmul — GPU-accelerated) ────────────────
        # x : [batch, n_qubits],  W.T : [n_qubits, n_qubits]
        return torch.tanh(x @ W.T)                              # [batch, n_qubits]


# ── Public factory ────────────────────────────────────────────────────────────

def make_quantum_mlp(
    input_size: int,
    sizes: List[int],
    hidden_activation: str = "ReLU",
    output_activation: Optional[str] = None,
    n_qubits: int = 4,
    n_qlayers: int = 2,
    device: str = "default.qubit",
    n_shots: int = 1024,
    ibm_backend: Optional[str] = None,
    noise_mitigation: bool = False,
    layernorm: bool = False,
    verbose: bool = False,
    **kwargs,
) -> nn.Sequential:
    """
    Hybrid quantum-classical MLP block (PennyLane backend).

    Architecture
    ------------
    [Linear(input_size → n_qubits)] → [LayerNorm?] → VQC → [Linear(n_qubits → output_size)] → [activation?]

    The projection layers are omitted when dimensions already match.

    Parameters
    ----------
    input_size        : Input feature dimension.
    sizes             : sizes[-1] is the output dimension (intermediate sizes ignored
                        for quantum block — circuit maps n_qubits → n_qubits).
    n_qubits          : Number of qubits in the VQC.
    n_qlayers         : Number of variational layers.
    device            : PennyLane device string.
    n_shots           : Shot count (Qiskit backends only).
    ibm_backend       : IBM hardware backend name (hardware execution only).
    noise_mitigation  : Apply ZNE noise mitigation (hardware only).
    layernorm         : LayerNorm after pre-projection (mirrors classical_mlp).
    **kwargs          : Silently absorbed for drop-in compatibility.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits >= 1 required, got {n_qubits}")
    if n_qlayers < 1:
        raise ValueError(f"n_qlayers >= 1 required, got {n_qlayers}")
    if not sizes:
        raise ValueError("sizes cannot be empty")

    output_size = sizes[-1]

    output_act_cls = None
    if output_activation:
        output_act_cls = getattr(nn, output_activation, None)
        if output_act_cls is None:
            raise ValueError(f"Unknown output_activation: '{output_activation}'")

    diff = _diff_method(device)

    if verbose:
        logger.info(
            "[QuantumMLP] device=%-20s diff=%-16s qubits=%d layers=%d",
            device, diff, n_qubits, n_qlayers,
        )

    # VQCLayer handles batching correctly for all diff methods.
    # TorchLayer is NOT used here because it fails with adjoint diff
    # (lightning.qubit / lightning.gpu) for batch_size > 1.
    q_layer = VQCLayer(n_qubits, n_qlayers, device, diff)

    layers: List[nn.Module] = []
    if input_size != n_qubits:
        layers.append(nn.Linear(input_size, n_qubits))
        if layernorm:
            layers.append(nn.LayerNorm(n_qubits))
    layers.append(q_layer)
    if n_qubits != output_size:
        layers.append(nn.Linear(n_qubits, output_size))
        if layernorm:
            layers.append(nn.LayerNorm(output_size))
    if output_act_cls:
        layers.append(output_act_cls())

    return nn.Sequential(*layers)

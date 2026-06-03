"""
models/quantum_mlp.py — Hybrid quantum-classical MLP (CTD 2025 architecture).

Structure per MLP
-----------------
  Linear(input_size → n_qubits)
  QTorchLayer  ×  (len(sizes) - 1)
  Linear(n_qubits → sizes[-1])

QTorchLayer
-----------
  Angle-encodes each sample's features into qubit rotations, applies trainable
  variational layers, and returns Pauli-Z expectation values.  Data enters the
  circuit directly as rotation angles — the circuit runs once per sample (not
  once per batch).

  Device auto-selection: lightning.gpu → lightning.qubit → default.qubit.
  adjoint differentiation (lightning) avoids storing per-gate intermediate
  states, keeping memory O(2^n_qubits × batch) instead of O(2^n × batch × gates).
  For default.qubit (backprop fallback), forward is chunked to cap CPU RAM.

Comparison with classical MLP
------------------------------
  Component       Classical MLP        Quantum Layer
  ─────────────── ──────────────────── ──────────────────────
  Input mapping   Linear layer         Feature encoding (RY)
  Nonlinearity    ReLU                 Entangling gates (CNOT, CRZ)
  Parameters      Weights / biases     Rotation angles
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Device selection ──────────────────────────────────────────────────────────

def _preload_custatevec() -> None:
    """
    Pre-load libcustatevec into the process with RTLD_GLOBAL so that
    lightning_gpu_ops.so can find it via dlopen when lightning.gpu initialises.
    custatevec ships as a Python package (custatevec-cu12) and its .so is not
    on the system linker path by default.
    """
    try:
        import ctypes, os
        import cuquantum
        lib = os.path.join(os.path.dirname(cuquantum.__file__), "lib", "libcustatevec.so.1")
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            logger.debug("Pre-loaded %s", lib)
    except Exception as e:
        logger.debug("custatevec pre-load skipped: %s", e)


def _make_pennylane_qnode(circuit, n_qubits: int):
    """
    Try lightning.gpu → lightning.qubit → default.qubit.

    Returns (qnode, cpu_transfer: bool, chunk_size: int).
    adjoint diff avoids storing per-gate states → O(2^n × batch) memory.
    cpu_transfer=True means forward() moves input tensors to CPU before the
    circuit and back to the original device afterward (gradient-preserving).
    chunk_size controls how many samples are passed per QNode call:
      lightning.gpu  → 1_000_000  (full batch in one call; cuStateVec batches natively)
      lightning.qubit → 512       (CPU C++; keep small to bound RAM)
      default.qubit  → 256        (Python sim; keep very small)
    """
    import warnings
    import pennylane as qml

    _preload_custatevec()

    candidates = [
        ("lightning.gpu",   "adjoint",  True, 1_000_000),
        ("lightning.qubit", "adjoint",  True, 512),
        ("default.qubit",   "backprop", True, 256),
    ]
    for dev_name, diff, cpu_transfer, chunk_size in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # suppress libnvidia-ml warnings
                dev   = qml.device(dev_name, wires=n_qubits)
                qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff)
            logger.info("[QTorchLayer] device=%s  diff_method=%s  chunk_size=%s",
                        dev_name, diff,
                        "unlimited" if chunk_size >= 1_000_000 else chunk_size)
            return qnode, cpu_transfer, chunk_size
        except Exception as e:
            logger.debug("[QTorchLayer] %s unavailable: %s", dev_name, e)
            continue
    raise RuntimeError("No PennyLane device could be initialised.")


# ── Per-sample variational quantum circuit layer ──────────────────────────────

class QTorchLayer(nn.Module):
    """
    Per-sample variational quantum circuit as a neural network layer.

    Each sample's feature vector x ∈ ℝ^n_qubits is angle-encoded into
    qubit rotations via RY(x_i) on wire i, then processed by n_qlayers of
    trainable variational gates (RX, RZ, RY, CNOT, CRZ).  The output is the
    vector of Pauli-Z expectation values ⟨Z_i⟩ ∈ [-1,1]^n_qubits.

    Input / output shape : [batch, n_qubits]
    Trainable parameters : weights  [n_qlayers, n_qubits, 4]  (rotation angles)

    data_reuploading : re-encode input before each variational layer (Pérez-Salinas
        et al. 2020). Adds Fourier frequency modes → provably more expressive.
    ring_entanglement : add CNOT(n-1→0) after the linear CNOT ladder, closing the
        ring and giving every qubit equal connectivity.
    chunk_size : max samples passed to the quantum circuit at once. Only active
        for default.qubit (backprop); lightning devices handle large batches natively.
    """

    def __init__(self, n_qubits: int, n_qlayers: int,
                 data_reuploading: bool = False,
                 ring_entanglement: bool = False) -> None:
        super().__init__()
        import pennylane as qml

        n_params_per_qubit = 4  # RX, RZ, RY, CRZ per qubit per layer

        def _encode(inputs):
            # Scale by π/2: Tanh-normalised inputs in [-1,1] → angles in [-π/2,π/2]
            # where d⟨Z⟩/dθ = -sin(θ) ≈ ±1 (maximum gradient, never vanishes).
            for i in range(n_qubits):
                qml.RY(inputs[..., i] * (math.pi / 2), wires=i)

        def _entangle(weights, layer):
            for w in range(n_qubits):
                qml.RX(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
                qml.RY(weights[layer, w, 2], wires=w)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            if ring_entanglement and n_qubits >= 3:
                qml.CNOT(wires=[n_qubits - 1, 0])
            for w in range(n_qubits - 1):
                qml.CRZ(weights[layer, w, 3], wires=[w, w + 1])

        def circuit(inputs, weights):
            _encode(inputs)
            for layer in range(n_qlayers):
                _entangle(weights, layer)
                if data_reuploading and layer < n_qlayers - 1:
                    _encode(inputs)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        qnode, self._cpu_transfer, self._chunk_size = _make_pennylane_qnode(circuit, n_qubits)

        # Near-zero init: variational circuit ≈ identity at epoch 0, so
        # output is dominated by the data encoding angles (not random noise).
        init_weights = {"weights": lambda s: torch.rand_like(s) * 0.1}
        weight_shapes = {"weights": (n_qlayers, n_qubits, n_params_per_qubit)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes,
                                         init_method=init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [batch, n_qubits] → [batch, n_qubits]"""
        device = x.device
        x_in = x.cpu() if self._cpu_transfer else x
        if x_in.shape[0] <= self._chunk_size:
            out = self.qlayer(x_in)
        else:
            out = torch.cat([self.qlayer(c) for c in x_in.split(self._chunk_size)])
        return out.to(device) if self._cpu_transfer else out


# ── Batch-level quantum transform (QPLT) ─────────────────────────────────────

class QPLTLayer(nn.Module):
    """
    Quantum-Parametrised Linear Transform — runs circuit ONCE per forward call.

    Instead of encoding each sample into the circuit (per-sample VQC), QPLT
    runs the circuit once with learnable angles self.encoding to produce
    expectation values e, builds a weight matrix W, and applies it to the
    full batch via GPU matmul:

        e = circuit(self.encoding)          # [n_qubits] — ONE circuit call
        W = outer(e, e) + diag(e)           # [n_qubits, n_qubits]
        output = tanh(x @ W.T)             # [batch, n_qubits]

    This replaces O(batch_size) QNode calls with 1, making it practical for
    graphs with tens of thousands of edges.

    Data features enter through the classical matmul, not the circuit.
    Gradient flows: loss → output → W → e → circuit params (encoding + weights).

    Input / output shape : [batch, n_qubits]
    Trainable parameters : encoding [n_qubits]           (circuit input angles)
                           weights  [n_qlayers, n_qubits, 4]  (variational)
    """

    def __init__(self, n_qubits: int, n_qlayers: int,
                 ring_entanglement: bool = False) -> None:
        super().__init__()
        import pennylane as qml

        self._n_qubits = n_qubits
        self.encoding  = nn.Parameter(torch.zeros(n_qubits))

        def _entangle(weights, layer):
            for w in range(n_qubits):
                qml.RX(weights[layer, w, 0], wires=w)
                qml.RZ(weights[layer, w, 1], wires=w)
                qml.RY(weights[layer, w, 2], wires=w)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            if ring_entanglement and n_qubits >= 3:
                qml.CNOT(wires=[n_qubits - 1, 0])
            for w in range(n_qubits - 1):
                qml.CRZ(weights[layer, w, 3], wires=[w, w + 1])

        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[..., i] * (math.pi / 2), wires=i)
            for layer in range(n_qlayers):
                _entangle(weights, layer)
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        qnode, self._cpu_transfer, _ = _make_pennylane_qnode(circuit, n_qubits)
        weight_shapes = {"weights": (n_qlayers, n_qubits, 4)}
        init_weights  = {"weights": lambda s: torch.rand_like(s) * 0.1}
        self.qlayer   = qml.qnn.TorchLayer(qnode, weight_shapes,
                                           init_method=init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [batch, n_qubits] → [batch, n_qubits]"""
        enc = self.encoding.cpu() if self._cpu_transfer else self.encoding
        e   = self.qlayer(enc.unsqueeze(0)).squeeze(0)              # [n_qubits]
        # Fix 1: normalise W so pre-tanh values stay ~O(1) at init (e≈1 → W/n ≈ I)
        W   = (torch.outer(e, e) + torch.diag(e)) / self._n_qubits  # [n_qubits, n_qubits]
        # Fix 3: residual connection — gradient flows even when W is near-zero
        return torch.tanh(x @ W.T.to(x.device)) + x                 # [batch, n_qubits]


# ── Public factory ────────────────────────────────────────────────────────────

def make_quantum_mlp(
    input_size: int,
    sizes: List[int],
    n_qubits: int = 4,
    n_qlayers: int = 1,
    output_activation: Optional[str] = None,
    verbose: bool = False,
    data_reuploading: bool = False,
    ring_entanglement: bool = False,
    # kept for drop-in API compatibility with make_classical_mlp callers:
    hidden_activation: str = "ReLU",
    layernorm: bool = False,
    device: str = "default.qubit",
    **kwargs,
) -> nn.Sequential:
    """
    Hybrid quantum-classical MLP (CTD 2025 architecture).

    Layout
    ------
        Linear(input_size → n_qubits)
        QTorchLayer  ×  (len(sizes) - 1)
        Linear(n_qubits → sizes[-1])
        [output_activation]

    Parameters
    ----------
    input_size  : input feature dimension
    sizes       : hidden + output dims, e.g. [16, 16, 16].
                  depth = len(sizes); number of QTorchLayers = len(sizes) - 1.
    n_qubits    : quantum circuit width = intermediate feature dimension.
    n_qlayers   : variational layers inside each QTorchLayer.
    **kwargs    : silently absorbed (n_shots, ibm_backend, use_vqc, …).
    """
    if not sizes:
        raise ValueError("sizes cannot be empty")
    if n_qubits < 1:
        raise ValueError(f"n_qubits >= 1 required, got {n_qubits}")
    if n_qlayers < 1:
        raise ValueError(f"n_qlayers >= 1 required, got {n_qlayers}")

    output_act_cls = None
    if output_activation:
        output_act_cls = getattr(nn, output_activation, None)
        if output_act_cls is None:
            raise ValueError(f"Unknown output_activation: '{output_activation}'")

    n_qtl = len(sizes) - 1  # number of QTorchLayer instances in this MLP

    if verbose:
        logger.info(
            "[QuantumMLP] input=%d  n_qubits=%d  n_qlayers=%d  depth=%d  out=%d",
            input_size, n_qubits, n_qlayers, n_qtl, sizes[-1],
        )

    layers: List[nn.Module] = []
    layers.append(nn.Linear(input_size, n_qubits))
    # Tanh normalises the projection to [-1, 1] so QPLT receives values in
    # the same range as the learnable encoding angles (scaled to [-π/2, π/2]).
    layers.append(nn.Tanh())
    for _ in range(n_qtl):
        layers.append(QPLTLayer(n_qubits, n_qlayers,
                                ring_entanglement=ring_entanglement))
    layers.append(nn.Linear(n_qubits, sizes[-1]))
    if output_act_cls is not None:
        layers.append(output_act_cls())

    return nn.Sequential(*layers)

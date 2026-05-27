"""
models/backend_factory.py — Unified factory for quantum MLP blocks.

Supported backends
------------------
  pennylane     : PennyLane TorchLayer  (fastest simulation, best GPU support)
  qiskit-ml     : qiskit-machine-learning TorchConnector (clean autograd)
  qiskit-native : Pure Qiskit + manual parameter-shift  (hardware-swap friendly)

Bug fix: original file was missing the final ``return`` statement for the
``qiskit-native`` branch — it fell off the end of the function returning None.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("pennylane", "qiskit-ml", "qiskit-native")


def make_quantum_mlp_for_backend(
    backend: str,
    input_size: int,
    sizes: List[int],
    hidden_activation: str = "ReLU",
    output_activation: Optional[str] = None,
    n_qubits: int = 4,
    n_qlayers: int = 2,
    device: str = "default.qubit",
    verbose: bool = False,
    **kwargs,
) -> nn.Sequential:
    """
    Unified factory — returns a fully differentiable hybrid quantum-classical block.

    Parameters
    ----------
    backend           : One of ``"pennylane"``, ``"qiskit-ml"``, ``"qiskit-native"``.
    input_size        : Input feature dimension.
    sizes             : sizes[-1] is the output dimension.
    n_qubits          : Number of qubits.
    n_qlayers         : Number of variational layers.
    device            : PennyLane device string (ignored for Qiskit backends).
    verbose           : Log architecture details.
    **kwargs          : Forwarded to the chosen backend factory (e.g. n_shots).

    Raises
    ------
    ValueError  : Unknown backend name.
    ImportError : Required library not installed.
    """
    backend = backend.lower().strip()

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown quantum backend '{backend}'. "
            f"Choose one of {SUPPORTED_BACKENDS}."
        )

    logger.debug(
        "Building quantum MLP | backend=%s | %d→%d | qubits=%d layers=%d",
        backend, input_size, sizes[-1], n_qubits, n_qlayers,
    )

    common = dict(
        input_size=input_size,
        sizes=sizes,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        n_qubits=n_qubits,
        n_qlayers=n_qlayers,
        verbose=verbose,
        **kwargs,
    )

    if backend == "pennylane":
        try:
            from models.quantum_mlp import make_quantum_mlp
        except ImportError as exc:
            raise ImportError(
                "PennyLane backend requires: pip install pennylane"
            ) from exc
        return make_quantum_mlp(**common, device=device)

    if backend == "qiskit-ml":
        try:
            from models.qiskit_ml_layer import make_qiskit_ml_mlp
        except ImportError as exc:
            raise ImportError(
                "qiskit-ml backend requires: pip install qiskit qiskit-machine-learning"
            ) from exc
        return make_qiskit_ml_mlp(**common)

    # backend == "qiskit-native"
    try:
        from models.qiskit_native_layer import make_qiskit_native_mlp
    except ImportError as exc:
        raise ImportError(
            "qiskit-native backend requires: pip install qiskit qiskit-aer"
        ) from exc
    return make_qiskit_native_mlp(**common)

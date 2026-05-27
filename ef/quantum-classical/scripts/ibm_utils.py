#!/usr/bin/env python
"""
scripts/ibm_utils.py
─────────────────────
IBM Quantum account management and hardware inspection utilities.

Usage
-----
# Save your IBM token (first time only)
python scripts/ibm_utils.py --save_token YOUR_TOKEN_HERE

# List all available backends
python scripts/ibm_utils.py --list

# Show noise profile for a specific machine
python scripts/ibm_utils.py --inspect ibm_brisbane

# Run a quick circuit test on a machine (verifies connectivity)
python scripts/ibm_utils.py --test ibm_brisbane --n_qubits 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ibm_utils")


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

def save_token(token: str) -> None:
    """Save IBM Quantum API token to disk (only needed once)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        overwrite=True,
    )
    logger.info("IBM token saved successfully.")


def get_service():
    """Return an authenticated QiskitRuntimeService."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        return QiskitRuntimeService(channel="ibm_quantum_platform")
    except Exception as exc:
        logger.error("Authentication failed: %s", exc)
        logger.error("Run: python scripts/ibm_utils.py --save_token YOUR_TOKEN")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Backend listing
# ─────────────────────────────────────────────────────────────────────────────

def list_backends(min_qubits: int = 0) -> None:
    """Print all backends you have access to."""
    service  = get_service()
    backends = service.backends()

    print(f"\n{'Backend':<25} {'Qubits':>7}  {'Status':<15}  {'Queue':>6}  {'T1 (µs)':>10}")
    print("─" * 75)

    for b in backends:
        try:
            status    = b.status()
            n_qubits  = b.num_qubits
            if n_qubits < min_qubits:
                continue
            queue     = status.pending_jobs
            props     = b.properties()
            t1_avg    = (
                sum(q.t1 * 1e6 for q in props.qubits) / len(props.qubits)
                if props else float("nan")
            )
            print(
                f"{b.name:<25} {n_qubits:>7}  {status.status_msg:<15}  "
                f"{queue:>6}  {t1_avg:>10.1f}"
            )
        except Exception as exc:
            print(f"{b.name:<25} (error: {exc})")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Backend inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_backend(backend_name: str) -> None:
    """Print noise properties for a specific backend."""
    service = get_service()
    backend = service.backend(backend_name)
    props   = backend.properties()

    print(f"\n── {backend_name} ──────────────────────────────────────────────")
    print(f"  Qubits      : {backend.num_qubits}")
    print(f"  Basis gates : {backend.operation_names}")

    if props:
        print(f"\n  {'Qubit':>6}  {'T1 (µs)':>10}  {'T2 (µs)':>10}  {'Readout err':>12}")
        print("  " + "─" * 46)
        for i, qubit_props in enumerate(props.qubits):
            t1  = next((p.value * 1e6 for p in qubit_props if p.name == "T1"),  float("nan"))
            t2  = next((p.value * 1e6 for p in qubit_props if p.name == "T2"),  float("nan"))
            err = next((p.value       for p in qubit_props if p.name == "readout_error"), float("nan"))
            print(f"  {i:>6}  {t1:>10.1f}  {t2:>10.1f}  {err:>12.4f}")

        print(f"\n  Gate errors (2-qubit):")
        for gate in props.gates:
            if len(gate.qubits) == 2:
                err = next((p.value for p in gate.parameters if p.name == "gate_error"), None)
                if err:
                    print(f"    {gate.gate:<8} {gate.qubits}  error={err:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Quick circuit test
# ─────────────────────────────────────────────────────────────────────────────

def test_backend(backend_name: str, n_qubits: int = 4) -> None:
    """
    Submit a minimal test circuit (Bell state) to verify connectivity.
    Uses the least-busy queue slot available.
    """
    import pennylane as qml
    import torch

    logger.info("Testing connectivity to %s with %d qubits …", backend_name, n_qubits)

    dev = qml.device("qiskit.ibmq", wires=n_qubits, shots=256, backend=backend_name)

    @qml.qnode(dev, interface="torch", diff_method="parameter_shift")
    def bell_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(min(2, n_qubits))]

    result = bell_circuit()
    logger.info("Test circuit result (expect ~[0, 0] for Bell state): %s", result)
    logger.info("Connection to %s: OK ✓", backend_name)


# ─────────────────────────────────────────────────────────────────────────────
# Recommend best backend
# ─────────────────────────────────────────────────────────────────────────────

def recommend_backend(n_qubits_needed: int = 4) -> str:
    """
    Return the name of the best available backend for your qubit count:
    least queue depth + highest average T1.
    """
    service  = get_service()
    backends = [
        b for b in service.backends()
        if b.num_qubits >= n_qubits_needed and not b.simulator
    ]

    if not backends:
        logger.warning("No hardware backends with ≥%d qubits found.", n_qubits_needed)
        return ""

    def score(b):
        try:
            status = b.status()
            props  = b.properties()
            queue  = status.pending_jobs
            t1_avg = (
                sum(q.t1 * 1e6 for q in props.qubits) / len(props.qubits)
                if props else 0.0
            )
            # Lower queue + higher T1 = better
            return -queue + t1_avg / 1000.0
        except Exception:
            return float("-inf")

    best = max(backends, key=score)
    logger.info(
        "Recommended backend: %s  (qubits=%d, queue=%d)",
        best.name, best.num_qubits, best.status().pending_jobs,
    )
    return best.name


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IBM Quantum utilities")
    parser.add_argument("--save_token",  type=str,  default=None, help="Save IBM API token")
    parser.add_argument("--list",        action="store_true",     help="List available backends")
    parser.add_argument("--min_qubits",  type=int,  default=0,    help="Filter by min qubits")
    parser.add_argument("--inspect",     type=str,  default=None, help="Inspect a backend")
    parser.add_argument("--test",        type=str,  default=None, help="Test a backend connection")
    parser.add_argument("--recommend",   action="store_true",     help="Recommend best backend")
    parser.add_argument("--n_qubits",    type=int,  default=4,    help="Qubits needed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_token:
        save_token(args.save_token)

    if args.list:
        list_backends(min_qubits=args.min_qubits)

    if args.inspect:
        inspect_backend(args.inspect)

    if args.test:
        test_backend(args.test, n_qubits=args.n_qubits)

    if args.recommend:
        name = recommend_backend(n_qubits_needed=args.n_qubits)
        if name:
            print(f"\nBest backend for {args.n_qubits} qubits: {name}")


if __name__ == "__main__":
    main()

# Changelog — qgnn_fixed → qgnn_clean

## Bug Fixes

### `models/backend_factory.py`
- **Critical**: Missing `return` statement at the end of the `qiskit-native` branch.
  The function fell off the end and returned `None`, causing a silent crash whenever
  the native backend was used.

### `models/quantum_mlp.py`
- **Circuit angle encoding bug**: the original `_step` loop did
  `angle = scaled[:, i] if i < scaled.shape[1] else 0`, then passed the whole
  `[batch, n_qubits]` slice to `qml.RY`. PennyLane expects a scalar or
  `[batch]` tensor per wire — passing `[:, i]` (shape `[batch]`) works only for
  batch_size=1 and gives wrong results for larger batches. Fixed by restructuring
  the circuit to receive per-sample inputs (the circuit now operates on `[n_qubits]`
  vectors after the `TorchLayer` batching, which is the correct PennyLane contract).
- Removed spurious `n_pp = 4 if n_qubits >= 2 else 3` branch — CRZ is always
  included when n_qubits ≥ 2 (which it must be); the `else 3` path was dead code.

### `models/qiskit_native_layer.py`
- **Performance bug**: the backward pass rebuilt the full `param_dict` inside
  `for sample in inputs` **inside** `for param_idx in range(len(weight_params))`,
  giving O(n_params × batch × 2) redundant dict constructions. Refactored to
  pre-build batches of x_np once per parameter shift call.
- Removed `use_backend` storing the backend on `self` without ever using it
  in `forward` — noted as inference-only in docstring instead.

### `data/dataset.py`
- **Type error in `_check_condition`**: the `not_within` branch compared
  `hi < math.inf` but `hi` was computed as `float(.inf)` (which is fine),
  *except* when `val[1]` was the string `".inf"` — `float(".inf")` raises
  `ValueError`. Added string sentinel check (`str(val[1]) in (".inf", "inf")`),
  matching the pattern used in the `[lo, hi]` branch.
- `_trim_graph` passed a sorted integer index tensor to a boolean-mask filter
  path that expected a boolean tensor. Unified edge-mask logic into a single
  `_filter_edges` helper used by both `apply_hard_cuts` and `_trim_graph`.

### `utils/callbacks.py`
- `TimingCallback.on_train_epoch_end` would crash with `AttributeError: _t0`
  if training was resumed from a checkpoint (epoch start hook skipped).
  Initialise `_t0 = 0.0` in `__init__`.

## Redundancy Removed

- `models/classical_mlp.py`: removed dead quantum kwargs from the signature
  (`n_qubits`, `n_qlayers`, `device`, `n_shots`, `ibm_backend`, `noise_mitigation`,
  `verbose`). The function never used them. Replaced with `**kwargs` absorber.
- `models/gnn.py`:
  - Removed duplicated `pos_weight` `criterion` construction (was set once in
    `__init__` and again in `_build_networks`; now done once in `__init__`).
  - Removed duplicated `_HPARAM_DEFAULTS` inline scattered across `_apply_defaults`
    — consolidated into a single module-level dict.
  - `_apply_defaults` renamed to module-level `_HPARAM_DEFAULTS` + `_IGNORED_KEYS`
    for clarity and testability.
- `models/qiskit_ml_layer.py` and `models/qiskit_native_layer.py` shared the
  identical ansatz circuit builder — deduplicated into a single `build_ansatz`
  in `qiskit_ml_layer.py`, imported by `qiskit_native_layer.py`.
- `utils/callbacks.py`: removed duplicated `_WANDB` / `wandb.run is not None`
  checks — consolidated into `_wandb_active()` helper.
- Removed committed binary artifacts: `run/wandb/`, `__pycache__/`, `.DS_Store`,
  `__MACOSX/` — these should never be in version control.

## Code Quality

- All public functions have proper docstrings with Parameters/Returns/Raises sections.
- Logger calls upgraded from `print()` to `logger.*()` throughout.
- Type annotations added to all public function signatures.
- `from __future__ import annotations` added consistently.
- `f"{split}_metrics_epoch"` access now uses `getattr(..., None)` guard throughout
  to prevent AttributeError on first epoch before metrics are populated.

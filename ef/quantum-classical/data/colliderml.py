"""Load ColliderML tracker_hits + particles parquet files into per-event tensors."""

import numpy as np
import torch
import pyarrow.parquet as pq
from pathlib import Path


def _to_cylindrical(x, y, z):
    """Cartesian mm → (r mm, phi rad, z mm, eta)."""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    # pseudorapidity from spatial coords
    p_mag = np.sqrt(r**2 + z**2)
    cos_theta = np.where(p_mag > 0, z / p_mag, 0.0)
    # clamp to avoid log(0)
    cos_theta = np.clip(cos_theta, -1 + 1e-7, 1 - 1e-7)
    eta = -0.5 * np.log((1 - cos_theta) / (1 + cos_theta))
    return r, phi, z, eta


def load_event(hits_row: dict, particles_row: dict) -> dict:
    """Parse one event from pydict rows → dict of float32/long tensors.

    Returns keys: r, phi, z, eta, particle_id, volume_id, layer_id, pt
    All arrays are 1-D, length = number of hits in the event.
    """
    x = np.array(hits_row["x"], dtype=np.float32)
    y = np.array(hits_row["y"], dtype=np.float32)
    z = np.array(hits_row["z"], dtype=np.float32)
    particle_id = np.array(hits_row["particle_id"], dtype=np.int64)
    volume_id = np.array(hits_row["volume_id"], dtype=np.int64)
    layer_id = np.array(hits_row["layer_id"], dtype=np.int64)

    r, phi, z_cyl, eta = _to_cylindrical(x, y, z)

    # Build pt lookup from particles table
    p_ids = np.array(particles_row["particle_id"], dtype=np.int64)
    px = np.array(particles_row["px"], dtype=np.float32)
    py = np.array(particles_row["py"], dtype=np.float32)
    pt_vals = np.sqrt(px**2 + py**2)
    pt_map = dict(zip(p_ids.tolist(), pt_vals.tolist()))
    pt = np.array([pt_map.get(pid, 0.0) for pid in particle_id.tolist()], dtype=np.float32)

    return {
        "r": torch.from_numpy(r),
        "phi": torch.from_numpy(phi),
        "z": torch.from_numpy(z_cyl),
        "eta": torch.from_numpy(eta),
        "particle_id": torch.from_numpy(particle_id),
        "volume_id": torch.from_numpy(volume_id),
        "layer_id": torch.from_numpy(layer_id),
        "pt": torch.from_numpy(pt),
        "event_id": hits_row["event_id"],
    }


class ColliderMLEvents:
    """Iterator over events from ColliderML parquet shards.

    Args:
        hits_dir: directory containing tracker_hits *.parquet shards
        particles_dir: directory containing particles *.parquet shards
        max_events: stop after this many events (None = all)
    """

    def __init__(self, hits_dir, particles_dir, max_events=None):
        self.hits_files = sorted(Path(hits_dir).glob("*.parquet"))
        self.particles_files = sorted(Path(particles_dir).glob("*.parquet"))
        assert len(self.hits_files) == len(self.particles_files), \
            "Mismatched number of hits/particles shards"
        self.max_events = max_events

    def __iter__(self):
        n_yielded = 0
        for hf, pf in zip(self.hits_files, self.particles_files):
            hits_table = pq.read_table(hf).to_pydict()
            parts_table = pq.read_table(pf).to_pydict()
            n_rows = len(hits_table["event_id"])
            for row in range(n_rows):
                hits_row = {k: hits_table[k][row] for k in hits_table}
                parts_row = {k: parts_table[k][row] for k in parts_table}
                yield load_event(hits_row, parts_row)
                n_yielded += 1
                if self.max_events is not None and n_yielded >= self.max_events:
                    return

    def __len__(self):
        # Each shard holds 1000 events
        total = len(self.hits_files) * 1000
        return total if self.max_events is None else min(total, self.max_events)

"""Build PyG graph files from ColliderML hits.

Three methods:
  embedding (recommended — Acorn style)
    Radius search in embedding space: for each hit, find all other hits
    within L2 distance r_infer. Handles all detector geometries including
    barrel/endcap transitions that geometric cuts miss. With a well-trained
    HNM embedding, achieves >95% efficiency at purity ~5-20%.
    The GNN classifier then cleans up the false positives.

  geometric
    Adjacent detector-layer doublets with phi/eta geometric cuts, then
    optionally filtered by embedding cosine similarity. Can miss barrel-endcap
    transitions. Kept for comparison.

  knn
    k-NN in embedding space. Good for small k but efficiency caps at k.

Usage:
    # Acorn-style embedding method (recommended):
    python scripts/build_graphs.py \\
        --hits-dir data/openml/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits \\
        --particles-dir data/openml/ttbar_pu0_particles/data/ttbar_pu0_particles \\
        --embedding outputs/embedding.pt \\
        --output-dir data/graphs \\
        --method embedding --r-infer 1.0 --k-infer 500 \\
        --split 800 100 100

Output layout:
    data/graphs/train_set/*.pyg
    data/graphs/val_set/*.pyg
    data/graphs/test_set/*.pyg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.colliderml import ColliderMLEvents
from models.embedding import HitEmbedding, build_edges_radius


# ─────────────────────────────────────────────────────────────────────────────
# Geometric cut — applied after any build method
# ─────────────────────────────────────────────────────────────────────────────

def _apply_geometric_cuts(data: Data,
                           phi_slope_cut: float,
                           z_slope_cut: float,
                           require_outward: bool) -> Data:
    """
    Prune edges that violate detector geometry constraints.

    phi_slope_cut  : max |Δphi| in radians. Tight cut removes high-angle
                     combinatorics. Typical value: 0.1 rad for pt > 1 GeV.
    z_slope_cut    : max |Δz/Δr|. Removes edges inconsistent with tracker
                     acceptance. Endcap pairs (|Δr| < 1 mm) bypass this cut.
    require_outward: if True, keep only edges with Δr > 0 (outward-pointing).
                     Eliminates ~50% of background with zero efficiency loss.
    """
    src, dst = data.edge_index
    keep = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=src.device)

    dr = data.r[dst] - data.r[src]

    if require_outward:
        keep &= dr > 0

    raw  = data.phi[dst] - data.phi[src]
    dphi = torch.atan2(torch.sin(raw), torch.cos(raw))
    if phi_slope_cut < float("inf"):
        keep &= dphi.abs() < phi_slope_cut

    if z_slope_cut < float("inf"):
        dz      = data.z[dst] - data.z[src]
        has_dr  = dr.abs() > 1.0          # mm — skip endcap pairs
        z_slope = torch.where(has_dr, (dz / dr.clamp(min=1.0)).abs(),
                              torch.zeros_like(dz))
        keep &= z_slope < z_slope_cut

    if keep.all():
        return data

    data.edge_index = data.edge_index[:, keep]
    data.y          = data.y[keep]
    for attr in ("dr", "dz", "dphi", "deta", "weights"):
        if hasattr(data, attr):
            setattr(data, attr, getattr(data, attr)[keep])
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Geometric doublet graph builder
# ─────────────────────────────────────────────────────────────────────────────

def _adjacent_layer_pairs(layer_key: np.ndarray,
                          r_np: np.ndarray,
                          z_np: np.ndarray) -> list[tuple[int, int]]:
    """Return (layer_a, layer_b) pairs to connect, sorted outward by mean radius."""
    unique_layers = np.unique(layer_key)
    # Sort by mean r of hits in each layer (outward direction)
    mean_r = {l: r_np[layer_key == l].mean() for l in unique_layers}
    sorted_layers = sorted(unique_layers, key=lambda l: mean_r[l])
    return list(zip(sorted_layers[:-1], sorted_layers[1:]))


def build_geometric_graph(event: dict, model: HitEmbedding,
                          phi_slope_cut: float, z_slope_cut: float,
                          sim_threshold: float,
                          min_pt: float, device: torch.device) -> Data:
    """
    1. Filter low-pT hits.
    2. Group hits by detector layer, sort layers outward by mean r.
    3. For each adjacent layer pair: create all hit-pair candidates.
    4. Apply geometric cuts: |Δphi| < phi_slope_cut, |Δz/Δr| < z_slope_cut.
    5. Score each edge with embedding cosine similarity; keep >= sim_threshold.
    6. Label edges: y=1 if same particle_id.
    """
    r   = event["r"].numpy().astype(np.float32)
    phi = event["phi"].numpy().astype(np.float32)
    z   = event["z"].numpy().astype(np.float32)
    eta = event["eta"].numpy().astype(np.float32)
    pid = event["particle_id"].numpy()
    pt  = event["pt"].numpy().astype(np.float32)
    vol = event["volume_id"].numpy()
    lay = event["layer_id"].numpy()

    # pT filter
    mask = (pt >= min_pt)
    if mask.sum() < 4:
        return None
    r, phi, z, eta, pid, vol, lay = (
        r[mask], phi[mask], z[mask], eta[mask],
        pid[mask], vol[mask], lay[mask]
    )

    layer_key = vol * 1000 + lay
    adj_pairs = _adjacent_layer_pairs(layer_key, r, z)

    all_src, all_dst = [], []

    for l1, l2 in adj_pairs:
        idx1 = np.where(layer_key == l1)[0]
        idx2 = np.where(layer_key == l2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue

        # All cross-layer candidate pairs (vectorised)
        src = np.repeat(idx1, len(idx2))
        dst = np.tile(idx2, len(idx1))

        # ── Geometric cuts ────────────────────────────────────────────────
        # Δphi wrapped to (-π, π)
        dphi = phi[dst] - phi[src]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dphi_abs = np.abs(dphi)

        # z-slope = Δz / Δr; skip for endcap pairs (small dr) to avoid div-by-zero
        dr = r[dst] - r[src]
        dz = z[dst] - z[src]
        has_dr = np.abs(dr) > 1.0
        z_slope = np.where(has_dr, np.abs(dz / np.where(has_dr, dr, 1.0)), 0.0)

        # Endcap pairs (small dr) pass z_slope automatically (set to 0)
        keep = (dphi_abs < phi_slope_cut) & (z_slope < z_slope_cut)

        all_src.append(src[keep])
        all_dst.append(dst[keep])

    if not all_src or sum(len(s) for s in all_src) == 0:
        return None

    src_np = np.concatenate(all_src)
    dst_np = np.concatenate(all_dst)

    r_t   = torch.from_numpy(r)
    phi_t = torch.from_numpy(phi)
    z_t   = torch.from_numpy(z)
    eta_t = torch.from_numpy(eta)

    # ── Embedding similarity filter (skip if threshold is 0) ─────────────────
    if sim_threshold > 0:
        feats = HitEmbedding.preprocess(r_t.to(device), phi_t.to(device),
                                        z_t.to(device), eta_t.to(device))
        with torch.no_grad():
            emb = model(feats)
        src_t = torch.from_numpy(src_np).long()
        dst_t = torch.from_numpy(dst_np).long()
        sim   = (emb[src_t] * emb[dst_t]).sum(dim=-1).cpu()
        keep  = (sim >= sim_threshold).numpy()
        src_np, dst_np = src_np[keep], dst_np[keep]
        if len(src_np) == 0:
            return None

    edge_index = torch.stack([
        torch.from_numpy(src_np.copy()).long(),
        torch.from_numpy(dst_np.copy()).long(),
    ])
    pid_t = torch.from_numpy(pid)
    y     = (pid_t[edge_index[0]] == pid_t[edge_index[1]]).float()

    n_nodes = int(mask.sum())
    return Data(
        r=r_t, phi=phi_t, z=z_t, eta=eta_t,
        edge_index=edge_index,
        y=y,
        particle_id=pid_t,
        pt=torch.from_numpy(pt[mask] if isinstance(pt, np.ndarray) else pt),
        num_nodes=n_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-based radius graph builder (Acorn style — recommended)
# ─────────────────────────────────────────────────────────────────────────────

def build_embedding_graph(event: dict, model: HitEmbedding,
                          r_infer: float, k_infer: int,
                          min_pt: float, device: torch.device) -> Data:
    """
    Build a graph via pure radius search in embedding space (Acorn style).

    For each hit, find all other hits within L2 distance r_infer in the
    learned embedding. No geometric cuts — the embedding handles all
    detector topologies including barrel/endcap transitions.

    Set r_infer slightly larger than r_train used during embedding training
    (typically r_infer = 1.5 * r_train) to avoid missing true doublets at
    the boundary.

    Args:
        r_infer  : L2 radius in embedding space (e.g. 1.0).
        k_infer  : Max neighbours per hit (e.g. 500 — set large to avoid
                   missing hits at high local density).
    """
    r   = event["r"].to(device)
    phi = event["phi"].to(device)
    z   = event["z"].to(device)
    eta = event["eta"].to(device)
    pid = event["particle_id"].to(device)
    pt  = event["pt"].to(device)

    mask = pt >= min_pt
    if mask.sum() < 4:
        return None

    r, phi, z, eta, pid = r[mask], phi[mask], z[mask], eta[mask], pid[mask]

    feats = HitEmbedding.preprocess(r, phi, z, eta)
    with torch.no_grad():
        emb = model(feats)

    edge_index = build_edges_radius(emb, r_max=r_infer, k_max=k_infer)
    if edge_index.shape[1] == 0:
        return None

    pid_t = pid
    y = ((pid_t[edge_index[0]] == pid_t[edge_index[1]]) &
         (pid_t[edge_index[0]] != 0)).float()

    return Data(
        r=r.cpu(), phi=phi.cpu(), z=z.cpu(), eta=eta.cpu(),
        edge_index=edge_index.cpu(),
        y=y.cpu(),
        particle_id=pid_t.cpu(),
        pt=pt[mask].cpu(),
        num_nodes=int(mask.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# k-NN graph builder (original method)
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_graph(event: dict, model: HitEmbedding,
                    k: int, min_pt: float,
                    sim_threshold: float,
                    device: torch.device) -> Data:
    from torch_geometric.nn import knn_graph

    r   = event["r"].to(device)
    phi = event["phi"].to(device)
    z   = event["z"].to(device)
    eta = event["eta"].to(device)
    pid = event["particle_id"].to(device)
    pt  = event["pt"].to(device)

    mask = (pt >= min_pt)
    if mask.sum() < 4:
        return None
    r, phi, z, eta, pid = r[mask], phi[mask], z[mask], eta[mask], pid[mask]

    feats = HitEmbedding.preprocess(r, phi, z, eta)
    with torch.no_grad():
        emb = model(feats)

    edge_index = knn_graph(emb, k=k, loop=False)

    # Optional similarity filter
    if sim_threshold > 0:
        src, dst = edge_index
        sim  = (emb[src] * emb[dst]).sum(dim=-1)
        keep = sim >= sim_threshold
        edge_index = edge_index[:, keep]

    src, dst = edge_index
    y = (pid[src] == pid[dst]).float()

    n_nodes = int(mask.sum())
    return Data(
        r=r.cpu(), phi=phi.cpu(), z=z.cpu(), eta=eta.cpu(),
        edge_index=edge_index.cpu(),
        y=y.cpu(),
        particle_id=pid.cpu(),
        pt=pt[mask].cpu(),
        num_nodes=n_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hits-dir",      required=True)
    p.add_argument("--particles-dir", required=True)
    p.add_argument("--embedding",     required=True)
    p.add_argument("--output-dir",    default="data/graphs")
    p.add_argument("--split", type=int, nargs=3, default=[800, 100, 100],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--min-pt",  type=float, default=0.5)

    # Method
    p.add_argument("--method", default="embedding",
                   choices=["embedding", "geometric", "knn"],
                   help="Graph construction method (default: embedding)")

    # Embedding method options (Acorn style — recommended)
    p.add_argument("--r-infer", type=float, default=1.0,
                   help="L2 radius in embedding space for graph construction "
                        "(embedding method). Set ~1.5×r_train used in training.")
    p.add_argument("--k-infer", type=int, default=500,
                   help="Max neighbours per hit for embedding radius search. "
                        "Set large (500+) to avoid missing hits at high density.")

    # Geometric cuts — applied after any build method
    p.add_argument("--phi-slope-cut",   type=float, default=float("inf"),
                   help="Max |Δphi| in radians. ~0.1 for pt>1 GeV; inf = off (default)")
    p.add_argument("--z-slope-cut",     type=float, default=float("inf"),
                   help="Max |Δz/Δr|. ~10 for ATLAS acceptance; inf = off (default)")
    p.add_argument("--require-outward", action="store_true",
                   help="Keep only edges with Δr > 0 (outward-pointing). Recommended.")
    p.add_argument("--sim-threshold", type=float, default=0.0,
                   help="Min embedding cosine similarity to keep edge; 0.0 = disabled")

    # k-NN method options
    p.add_argument("--k", type=int, default=50,
                   help="Nearest neighbours (knn method only)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Method : {args.method}")
    if args.method == "embedding":
        print(f"  r_infer         = {args.r_infer}")
        print(f"  k_infer         = {args.k_infer}")
    elif args.method == "geometric":
        print(f"  sim_threshold   = {args.sim_threshold}")
    else:
        print(f"  k               = {args.k}")
        print(f"  sim_threshold   = {args.sim_threshold}")
    print(f"Geometric cuts:")
    print(f"  require_outward = {args.require_outward}")
    print(f"  phi_slope_cut   = {args.phi_slope_cut}")
    print(f"  z_slope_cut     = {args.z_slope_cut}")
    print()

    # Load embedding
    ckpt      = torch.load(args.embedding, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    model = HitEmbedding(
        embed_dim=ckpt_args.get("embed_dim", 16),
        hidden=ckpt_args.get("hidden", 256),
        n_blocks=ckpt_args.get("n_blocks", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded embedding from {args.embedding}")
    if "val_eff" in ckpt:
        print(f"  eff@r_train at save : {ckpt['val_eff']:.3f}")
    if "val_pur" in ckpt:
        print(f"  purity at save      : {ckpt['val_pur']:.3f}")
    print()

    total_events = sum(args.split)
    print(f"Loading {total_events} events...")
    events = list(ColliderMLEvents(
        args.hits_dir, args.particles_dir, max_events=total_events))

    output_dir  = Path(args.output_dir)
    split_names = ["train_set", "val_set", "test_set"]

    idx = 0
    for split_name, n in zip(split_names, args.split):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_events = events[idx: idx + n]
        idx += n
        print(f"Building {split_name} ({n} events)...")

        n_saved = n_skipped = 0
        for i, ev in enumerate(split_events):
            if args.method == "embedding":
                data = build_embedding_graph(
                    ev, model,
                    r_infer=args.r_infer,
                    k_infer=args.k_infer,
                    min_pt=args.min_pt,
                    device=device,
                )
            elif args.method == "geometric":
                data = build_geometric_graph(
                    ev, model,
                    phi_slope_cut=args.phi_slope_cut,
                    z_slope_cut=args.z_slope_cut,
                    sim_threshold=args.sim_threshold,
                    min_pt=args.min_pt,
                    device=device,
                )
            else:
                data = build_knn_graph(
                    ev, model,
                    k=args.k,
                    min_pt=args.min_pt,
                    sim_threshold=args.sim_threshold,
                    device=device,
                )

            if data is None:
                n_skipped += 1
                continue

            data = _apply_geometric_cuts(
                data,
                phi_slope_cut=args.phi_slope_cut,
                z_slope_cut=args.z_slope_cut,
                require_outward=args.require_outward,
            )
            if data.edge_index.shape[1] == 0:
                n_skipped += 1
                continue

            out_path = split_dir / f"event{ev['event_id']:06d}.pyg"
            torch.save(data, out_path)
            n_saved += 1

            if (i + 1) % 100 == 0 or i == 0:
                n_true  = data.y.sum().int().item()
                purity  = data.y.float().mean().item()
                print(f"  [{i+1:4d}/{n}] event {ev['event_id']}: "
                      f"{data.num_nodes} hits, {data.num_edges} edges, "
                      f"{n_true} true ({100*purity:.1f}% purity)")

        print(f"  Saved {n_saved}  Skipped {n_skipped}\n")

    print(f"Done. Graphs saved to {output_dir}/")
    print("Set input_dir in configs/default.yaml to:", output_dir.resolve())


if __name__ == "__main__":
    main()

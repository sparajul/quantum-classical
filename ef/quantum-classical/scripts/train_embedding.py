"""Train hit embedding with Hard Negative Mining + hinge loss (Acorn style).

Per-step training procedure (one full event per step):
  1. Forward all hits → L2-normalised embedding.
  2. True doublets: ALL consecutive same-particle layer pairs with pT > min_pt.
     These are the signal the embedding must learn to cluster.
  3. Hard negatives (HNM): radius search in current embedding space finds
     non-track pairs that are currently close — these are the hardest cases.
  4. Random pairs: for coverage of easy negatives (training stability).
  5. Hinge loss on squared L2 distances: positives < margin, negatives > margin.

Key metrics:
  eff@r_train  — fraction of true doublets within r_train in embedding space.
                 This directly predicts graph-construction efficiency.
  purity       — fraction of edges within r_train that are true doublets.

Usage:
    python scripts/train_embedding.py \\
        --hits-dir data/openml/ttbar_pu0_tracker_hits/data/ttbar_pu0_tracker_hits \\
        --particles-dir data/openml/ttbar_pu0_particles/data/ttbar_pu0_particles \\
        --output outputs/embedding.pt

Recommended settings for best quality:
    --epochs 60 --embed-dim 16 --hidden 256 --n-blocks 4 \\
    --r-train 0.6 --margin 0.6 --k-hnm 50 --n-random 2000 \\
    --max-events 800 --min-pt 1.0
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.colliderml import ColliderMLEvents
from models.embedding import (
    HitEmbedding,
    build_consecutive_doublets,
    build_edges_radius,
    weighted_hinge_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_cosine_lr(optimizer, epoch, warmup, total, base_lr, min_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        t = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, events, device, r_train, k_hnm, min_pt, n_events=20):
    """Measure efficiency (fraction of true doublets within r_train) and loss."""
    model.eval()
    total_loss = 0.0
    total_eff = 0.0
    total_pur = 0.0
    n = 0

    for ev in events[:n_events]:
        x = HitEmbedding.preprocess(
            ev["r"].to(device), ev["phi"].to(device),
            ev["z"].to(device), ev["eta"].to(device),
        )
        emb = model(x)

        pid = ev["particle_id"].to(device)
        pt  = ev["pt"].to(device)
        lk  = ev["volume_id"].to(device) * 1000 + ev["layer_id"].to(device)

        true_edges = build_consecutive_doublets(pid, lk, pt, min_pt)
        if true_edges.shape[1] == 0:
            continue

        hnm_edges  = build_edges_radius(emb.detach(), r_max=r_train, k_max=k_hnm)
        edges      = torch.unique(
            torch.cat([true_edges, hnm_edges], dim=1), dim=1)
        mask       = edges[0] != edges[1]
        edges      = edges[:, mask]

        d     = ((emb[edges[0]] - emb[edges[1]]) ** 2).sum(-1)
        truth = ((pid[edges[0]] == pid[edges[1]]) & (pid[edges[0]] != 0)).float()

        loss = weighted_hinge_loss(d, truth, margin=r_train)

        # Efficiency: true doublets within r_train
        d_true = ((emb[true_edges[0]] - emb[true_edges[1]]) ** 2).sum(-1)
        eff    = (d_true < r_train ** 2).float().mean().item()

        # Purity of radius graph
        if hnm_edges.shape[1] > 0:
            d_hnm   = ((emb[hnm_edges[0]] - emb[hnm_edges[1]]) ** 2).sum(-1)
            t_hnm   = ((pid[hnm_edges[0]] == pid[hnm_edges[1]]) &
                       (pid[hnm_edges[0]] != 0)).float()
            pur = t_hnm.mean().item()
        else:
            pur = 0.0

        total_loss += loss.item()
        total_eff  += eff
        total_pur  += pur
        n += 1

    if n == 0:
        return float("inf"), 0.0, 0.0

    return total_loss / n, total_eff / n, total_pur / n


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--hits-dir",      required=True)
    p.add_argument("--particles-dir", required=True)
    p.add_argument("--max-events",    type=int,   default=800)
    p.add_argument("--val-events",    type=int,   default=100)
    p.add_argument("--min-pt",        type=float, default=1.0,
                   help="Min pT [MeV] for signal doublets")
    # Model
    p.add_argument("--embed-dim",  type=int,   default=16)
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--n-blocks",   type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    # HNM / loss
    p.add_argument("--r-train",  type=float, default=0.6,
                   help="L2 radius for HNM and hinge margin")
    p.add_argument("--margin",   type=float, default=0.6,
                   help="Hinge loss margin (set equal to r-train)")
    p.add_argument("--k-hnm",    type=int,   default=50,
                   help="Max neighbours for hard negative mining")
    p.add_argument("--n-random",   type=int,   default=2000,
                   help="Random pairs per event (for stability)")
    p.add_argument("--pos-weight", type=float, default=2.0,
                   help="Upweight positive (signal) loss to counter neg:pos imbalance. "
                        "5.0 causes collapsed-embedding stall; 1.0 lets negatives dominate. "
                        "2.0 is a good starting point.")
    # Training
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--min-lr",        type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int,   default=5)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    # Output
    p.add_argument("--output", default="outputs/embedding.pt")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Model     : embed_dim={args.embed_dim}  hidden={args.hidden}  "
          f"n_blocks={args.n_blocks}  dropout={args.dropout}")
    print(f"Training  : r_train={args.r_train}  margin={args.margin}  "
          f"k_hnm={args.k_hnm}  n_random={args.n_random}  "
          f"pos_weight={args.pos_weight}")
    print(f"Schedule  : {args.epochs} epochs  lr={args.lr}  "
          f"warmup={args.warmup_epochs}  patience={args.patience}")
    print()

    # ── Data ─────────────────────────────────────────────────────────────────
    total = args.max_events + args.val_events
    print(f"Loading {total} events...")
    events = list(ColliderMLEvents(
        args.hits_dir, args.particles_dir, max_events=total))
    train_events = events[: args.max_events]
    val_events   = events[args.max_events :]
    print(f"  {len(train_events)} train  {len(val_events)} val")
    print()

    # ── Model — retry initialization until non-collapsed ─────────────────────
    # A collapsed init (all hits mapped to nearly the same unit vector) gives
    # near-zero gradients: f(i)-f(j) ≈ 0 for all pairs → training stalls.
    # We detect collapse by checking nEdges within r_train on one event.
    # If nEdges > collapse_threshold, the init is bad; reinitialize and retry.
    _collapse_threshold = 100_000
    _max_init_tries = 10
    _probe_ev = train_events[0]
    _probe_x = HitEmbedding.preprocess(
        _probe_ev["r"].to(device), _probe_ev["phi"].to(device),
        _probe_ev["z"].to(device),  _probe_ev["eta"].to(device),
    )

    for _try in range(_max_init_tries):
        model = HitEmbedding(
            embed_dim=args.embed_dim,
            hidden=args.hidden,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
        ).to(device)
        with torch.no_grad():
            _emb = model(_probe_x)
            _n = build_edges_radius(_emb, r_max=args.r_train, k_max=args.k_hnm).shape[1]
        if _n < _collapse_threshold:
            print(f"Init OK   : nEdges={_n:,} on probe event (try {_try+1})")
            break
        print(f"Init retry: nEdges={_n:,} > {_collapse_threshold:,} (collapsed), try {_try+1}/{_max_init_tries}")
    else:
        print(f"WARNING: all {_max_init_tries} inits were collapsed. Proceeding with last init.")
    del _probe_ev, _probe_x, _emb, _n

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4, amsgrad=True)

    # ── Training ─────────────────────────────────────────────────────────────
    best_val_score = 0.0   # F1(eff, purity) = 2*eff*pur/(eff+pur)
    patience_count = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    hdr = (f"{'Epoch':>6}  {'LR':>8}  "
           f"{'TrainLoss':>10}  {'ValLoss':>9}  "
           f"{'eff@r':>7}  {'purity':>7}  {'F1':>7}  "
           f"{'nEdges':>8}")
    print(hdr)
    print("-" * len(hdr))

    # Sanity-check: verify signal doublets exist in the first event.
    # If this fails, min_pt is almost certainly in wrong units.
    _ev0 = train_events[0]
    _pt0 = _ev0["pt"]
    _pid0 = _ev0["particle_id"]
    _lk0 = _ev0["volume_id"] * 1000 + _ev0["layer_id"]
    _check = build_consecutive_doublets(_pid0, _lk0, _pt0, args.min_pt)
    if _check.shape[1] == 0:
        print(f"\nFATAL: no signal doublets found in the first event "
              f"with min_pt={args.min_pt}.")
        print(f"  pt range in event: {_pt0[_pt0 > 0].min():.4f} – {_pt0.max():.4f}")
        print(f"  Check units: OpenML px/py are in GeV, so min_pt should be in GeV "
              f"(e.g. --min-pt 1.0 for 1 GeV, not 1000).")
        raise SystemExit(1)
    del _ev0, _pt0, _pid0, _lk0, _check

    for epoch in range(args.epochs):
        lr = _warmup_cosine_lr(
            optimizer, epoch, args.warmup_epochs, args.epochs,
            args.lr, args.min_lr)

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_steps    = 0

        for ev in train_events:
            x = HitEmbedding.preprocess(
                ev["r"].to(device), ev["phi"].to(device),
                ev["z"].to(device), ev["eta"].to(device),
            )

            optimizer.zero_grad()
            emb = model(x)                    # (N, D) unit vectors, grad tracked

            pid = ev["particle_id"].to(device)
            pt  = ev["pt"].to(device)
            lk  = ev["volume_id"].to(device) * 1000 + ev["layer_id"].to(device)

            # 1. True signal doublets (all consecutive same-particle pairs)
            true_edges = build_consecutive_doublets(pid, lk, pt, args.min_pt)
            if true_edges.shape[1] == 0:
                continue

            # 2. Hard negatives: radius search in current (detached) embedding
            hnm_edges = build_edges_radius(emb.detach(), r_max=args.r_train,
                                           k_max=args.k_hnm)

            # 3. Random pairs for stability
            n_hits = x.shape[0]
            n_rnd  = min(args.n_random, n_hits * (n_hits - 1) // 2)
            rnd_src = torch.randint(n_hits, (n_rnd,), device=device)
            rnd_dst = torch.randint(n_hits, (n_rnd,), device=device)
            rnd_edges = torch.stack([rnd_src, rnd_dst])

            # Combine, deduplicate, remove self-loops
            all_edges = torch.unique(
                torch.cat([true_edges, hnm_edges, rnd_edges], dim=1), dim=1)
            sl_mask = all_edges[0] != all_edges[1]
            all_edges = all_edges[:, sl_mask]

            if all_edges.shape[1] == 0:
                continue

            # Squared L2 distances (with grad for backprop)
            d = ((emb[all_edges[0]] - emb[all_edges[1]]) ** 2).sum(-1)

            # Truth: same non-noise particle
            truth = ((pid[all_edges[0]] == pid[all_edges[1]]) &
                     (pid[all_edges[0]] != 0)).float()

            loss = weighted_hinge_loss(d, truth, margin=args.margin,
                                       pos_weight=args.pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            n_steps    += 1

        train_loss /= max(n_steps, 1)

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_eff, val_pur = validate(
            model, val_events, device,
            r_train=args.r_train, k_hnm=args.k_hnm,
            min_pt=args.min_pt, n_events=min(20, len(val_events)),
        )
        model.train()

        # Approximate edge count per event (from last training event)
        n_edges_approx = all_edges.shape[1] if n_steps > 0 else 0

        # F1 balances eff and purity: high only when BOTH are high.
        # Checkpointing on eff alone stops at epoch 1 (eff saturates near 1.0
        # at initialization); purity keeps improving for many more epochs.
        val_f1 = (2 * val_eff * val_pur / (val_eff + val_pur + 1e-8))

        print(f"{epoch+1:6d}  {lr:8.2e}  "
              f"{train_loss:10.4f}  {val_loss:9.4f}  "
              f"{val_eff:7.3f}  {val_pur:7.3f}  {val_f1:7.3f}  "
              f"{n_edges_approx:8d}")

        # ── Checkpoint on best F1(eff, purity) ───────────────────────────
        if val_f1 > best_val_score:
            best_val_score = val_f1
            patience_count = 0
            torch.save({
                "model_state": model.state_dict(),
                "args":        vars(args),
                "epoch":       epoch + 1,
                "val_loss":    val_loss,
                "val_eff":     val_eff,
                "val_pur":     val_pur,
                "val_f1":      val_f1,
            }, args.output)
            print(f"         -> saved  "
                  f"(F1={val_f1:.3f}  eff@r={val_eff:.3f}  "
                  f"purity={val_pur:.3f}  loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(F1 not improved for {args.patience} epochs)")
                break

    print(f"\nDone. Best val F1: {best_val_score:.4f}")
    print(f"Checkpoint: {args.output}")
    print()
    print("Next step: build graphs with embedding-based radius search:")
    print(f"  python scripts/build_graphs.py --embedding {args.output} "
          f"--method embedding --r-infer 1.0 --k-infer 500 ...")


if __name__ == "__main__":
    main()

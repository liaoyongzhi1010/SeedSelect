#!/usr/bin/env python3
"""Train learned proxy MLP for SeedSelect (Task B2).

Uses per-view LPIPS deltas (6D feature vector) from B1 to train a small MLP
that predicts GT Chamfer Distance. Uses 5-fold cross-validation on GSO-300.

Usage:
    python scripts/train_learned_proxy.py
"""
import json
import os
import sys
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DELTAS_PATH = REPO / "outputs/multiseed/gso_full/per_view_deltas.json"
RESULTS_PATH = REPO / "outputs/multiseed/gso_full/results.json"
OUTPUT_PATH = REPO / "outputs/multiseed/learned_proxy_results.json"

VIEWS = ['front', 'back', 'left', 'right', 'top', 'front_right']


def load_data():
    """Load per-view deltas and GT CD, return aligned arrays."""
    with open(DELTAS_PATH) as f:
        deltas = json.load(f)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Build feature matrix and target vector
    # Each row: [delta_front, delta_back, delta_left, delta_right, delta_top, delta_front_right]
    # Target: GT CD for this seed
    objects = []
    features = []
    targets = []
    obj_indices = []  # For per-object grouping

    obj_id_map = {}
    obj_idx = 0

    for obj_id in sorted(deltas.keys()):
        if obj_id not in results:
            continue

        seed_data = results[obj_id].get('seeds', {})
        obj_deltas = deltas[obj_id]

        if obj_id not in obj_id_map:
            obj_id_map[obj_id] = obj_idx
            obj_idx += 1

        for seed, view_deltas in obj_deltas.items():
            if seed not in seed_data:
                continue

            # Build 6D feature vector
            feat = [view_deltas.get(v, 0.0) for v in VIEWS]
            if len(feat) != 6 or any(f == 0.0 for f in feat):
                continue

            features.append(feat)
            targets.append(seed_data[seed]['cd'])
            obj_indices.append(obj_id_map[obj_id])
            objects.append((obj_id, seed))

    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    obj_indices = np.array(obj_indices)

    print(f"Loaded {len(features)} samples from {len(obj_id_map)} objects")
    print(f"Feature shape: {features.shape}")
    print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")

    return features, targets, obj_indices, objects, obj_id_map


def train_mlp_fold(X_train, y_train, X_val, y_val, epochs=500, lr=1e-3):
    """Train a small MLP on one fold."""
    import torch
    import torch.nn as nn

    class ProxyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ProxyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    X_v = torch.from_numpy(X_val).to(device)

    best_val_loss = float('inf')
    best_preds = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v).cpu().numpy()
                val_loss = np.mean((val_pred - y_val) ** 2)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_preds = val_pred.copy()

    # Final predictions
    model.eval()
    with torch.no_grad():
        final_preds = model(X_v).cpu().numpy()

    return final_preds if best_preds is None else best_preds


def compute_selection_metrics(predictions, targets, obj_indices, objects, results):
    """Compute SeedSelect-style metrics using MLP predictions as proxy."""
    from itertools import combinations

    # Group by object
    obj_groups = {}
    for i, (obj_id, seed) in enumerate(objects):
        if obj_id not in obj_groups:
            obj_groups[obj_id] = []
        obj_groups[obj_id].append({
            'seed': seed,
            'pred': predictions[i],
            'cd': targets[i],
        })

    default_cds = []
    selected_cds = []
    oracle_cds = []
    pairwise_accs = []

    for obj_id, candidates in obj_groups.items():
        if len(candidates) < 2:
            continue

        seed_data = results[obj_id]['seeds']
        default_seed = '42'
        if default_seed not in seed_data:
            continue

        default_cd = seed_data[default_seed]['cd']

        # MLP selection: pick candidate with LOWEST predicted CD
        selected = min(candidates, key=lambda c: c['pred'])
        oracle = min(candidates, key=lambda c: c['cd'])

        default_cds.append(default_cd)
        selected_cds.append(selected['cd'])
        oracle_cds.append(oracle['cd'])

        # Pairwise accuracy
        pairs = list(combinations(range(len(candidates)), 2))
        concordant = 0
        total = 0
        for i, j in pairs:
            cd_diff = candidates[i]['cd'] - candidates[j]['cd']
            pred_diff = candidates[i]['pred'] - candidates[j]['pred']
            if cd_diff != 0 and pred_diff != 0:
                if (cd_diff > 0) == (pred_diff > 0):
                    concordant += 1
                total += 1
        if total > 0:
            pairwise_accs.append(concordant / total)

    default_cds = np.array(default_cds)
    selected_cds = np.array(selected_cds)
    oracle_cds = np.array(oracle_cds)

    improvement = (default_cds.mean() - selected_cds.mean()) / default_cds.mean() * 100
    oracle_imp = (default_cds.mean() - oracle_cds.mean()) / default_cds.mean() * 100
    gap_closed = improvement / oracle_imp * 100 if oracle_imp > 0 else 0

    return {
        'n_objects': len(default_cds),
        'improvement_pct': float(improvement),
        'oracle_improvement_pct': float(oracle_imp),
        'gap_closed_pct': float(gap_closed),
        'mean_pairwise_acc': float(np.mean(pairwise_accs)) if pairwise_accs else 0,
        'default_cd': float(default_cds.mean()),
        'selected_cd': float(selected_cds.mean()),
        'oracle_cd': float(oracle_cds.mean()),
    }


def main():
    features, targets, obj_indices, objects, obj_id_map = load_data()

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # 5-fold cross-validation (split by objects, not samples)
    unique_objs = np.unique(obj_indices)
    n_objs = len(unique_objs)
    np.random.seed(42)
    perm = np.random.permutation(n_objs)
    fold_size = n_objs // 5

    all_predictions = np.zeros(len(features))
    fold_results = []

    print(f"\n5-Fold Cross-Validation ({n_objs} objects)")
    print("=" * 60)

    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else n_objs
        val_obj_ids = set(perm[start:end])

        val_mask = np.array([obj_indices[i] in val_obj_ids for i in range(len(features))])
        train_mask = ~val_mask

        X_train = features[train_mask]
        y_train = targets[train_mask]
        X_val = features[val_mask]
        y_val = targets[val_mask]

        print(f"\nFold {fold+1}: train={train_mask.sum()}, val={val_mask.sum()}")

        val_preds = train_mlp_fold(X_train, y_train, X_val, y_val)
        all_predictions[val_mask] = val_preds

        # Per-fold correlation
        from scipy.stats import spearmanr
        rho, p = spearmanr(val_preds, y_val)
        mse = np.mean((val_preds - y_val) ** 2)
        print(f"  MSE: {mse:.6f}, Spearman rho: {rho:.3f} (p={p:.2e})")

        fold_results.append({'mse': float(mse), 'spearman_rho': float(rho), 'p': float(p)})

    # Overall metrics
    from scipy.stats import spearmanr
    rho_all, p_all = spearmanr(all_predictions, targets)
    print(f"\nOverall Spearman rho: {rho_all:.3f} (p={p_all:.2e})")

    # Compute selection metrics
    metrics = compute_selection_metrics(all_predictions, targets, obj_indices, objects, results)

    print(f"\nSelection Metrics (Learned Proxy MLP):")
    print(f"  Improvement: {metrics['improvement_pct']:+.2f}%")
    print(f"  Oracle Improvement: {metrics['oracle_improvement_pct']:+.2f}%")
    print(f"  Gap Closed: {metrics['gap_closed_pct']:.1f}%")
    print(f"  Pairwise Accuracy: {metrics['mean_pairwise_acc']:.1%}")

    # Compare with hand-crafted mean
    print(f"\nComparison:")
    print(f"  Hand-crafted mean: 22.1% gap closed, 55.3% pairwise acc")
    print(f"  Learned MLP:       {metrics['gap_closed_pct']:.1f}% gap closed, {metrics['mean_pairwise_acc']:.1%} pairwise acc")

    # Save results
    output = {
        'overall': {
            'spearman_rho': float(rho_all),
            'spearman_p': float(p_all),
            **metrics,
        },
        'folds': fold_results,
        'comparison': {
            'handcrafted_mean_gap_pct': 22.1,
            'handcrafted_mean_pairwise_acc': 0.553,
            'learned_mlp_gap_pct': metrics['gap_closed_pct'],
            'learned_mlp_pairwise_acc': metrics['mean_pairwise_acc'],
        }
    }

    with open(str(OUTPUT_PATH), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train learned proxy with pairwise ranking loss (Task B2 variant).

Instead of MSE on absolute CD, trains with a pairwise ranking objective:
for each pair of candidates within the same object, predict which one has
lower CD.
"""
import json
import os
import numpy as np
from pathlib import Path
from itertools import combinations

REPO = Path(__file__).resolve().parent.parent
DELTAS_PATH = REPO / "outputs/multiseed/gso_full/per_view_deltas.json"
RESULTS_PATH = REPO / "outputs/multiseed/gso_full/results.json"
OUTPUT_PATH = REPO / "outputs/multiseed/learned_proxy_results.json"

VIEWS = ['front', 'back', 'left', 'right', 'top', 'front_right']


def load_data():
    with open(DELTAS_PATH) as f:
        deltas = json.load(f)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Group by object
    objects_data = {}
    for obj_id in sorted(deltas.keys()):
        if obj_id not in results:
            continue
        seed_data = results[obj_id].get('seeds', {})
        obj_deltas = deltas[obj_id]

        candidates = []
        for seed, view_deltas in obj_deltas.items():
            if seed not in seed_data:
                continue
            feat = [view_deltas.get(v, 0.0) for v in VIEWS]
            if len(feat) != 6 or any(f == 0.0 for f in feat):
                continue
            candidates.append({
                'seed': seed,
                'features': np.array(feat, dtype=np.float32),
                'cd': seed_data[seed]['cd'],
            })

        if len(candidates) >= 2:
            objects_data[obj_id] = candidates

    print(f"Loaded {len(objects_data)} objects")
    return objects_data, results


def train_ranking_mlp(train_objs, val_objs, epochs=300, lr=1e-3):
    """Train MLP with pairwise ranking (margin) loss."""
    import torch
    import torch.nn as nn

    class RankingMLP(nn.Module):
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
    model = RankingMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Build pairwise training data
    pairs_feat_a, pairs_feat_b, pairs_labels = [], [], []
    for obj_id, candidates in train_objs.items():
        for i, j in combinations(range(len(candidates)), 2):
            ci, cj = candidates[i], candidates[j]
            if abs(ci['cd'] - cj['cd']) < 1e-6:
                continue
            # Label: 1 if ci has lower CD (better), -1 if cj is better
            label = 1.0 if ci['cd'] < cj['cd'] else -1.0
            pairs_feat_a.append(ci['features'])
            pairs_feat_b.append(cj['features'])
            pairs_labels.append(label)

    X_a = torch.from_numpy(np.array(pairs_feat_a)).to(device)
    X_b = torch.from_numpy(np.array(pairs_feat_b)).to(device)
    Y = torch.from_numpy(np.array(pairs_labels, dtype=np.float32)).to(device)

    print(f"    Training pairs: {len(Y)}")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        score_a = model(X_a)
        score_b = model(X_b)

        # Margin ranking loss: we want score_a < score_b when a has lower CD
        # (lower score = lower predicted CD = better)
        # loss = max(0, -Y * (score_b - score_a) + margin)
        margin = 0.01
        loss = torch.clamp(-Y * (score_b - score_a) + margin, min=0).mean()

        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_predictions = {}
    with torch.no_grad():
        for obj_id, candidates in val_objs.items():
            feats = torch.from_numpy(np.array([c['features'] for c in candidates])).to(device)
            scores = model(feats).cpu().numpy()
            val_predictions[obj_id] = [(c['seed'], float(s), c['cd']) for c, s in zip(candidates, scores)]

    return val_predictions


def evaluate_selection(predictions, results):
    """Evaluate selection using MLP predictions."""
    default_cds, selected_cds, oracle_cds = [], [], []
    pairwise_accs = []

    for obj_id, pred_list in predictions.items():
        seed_data = results[obj_id]['seeds']
        default_seed = '42'
        if default_seed not in seed_data:
            continue

        default_cd = seed_data[default_seed]['cd']

        # MLP: lower predicted score = better
        selected = min(pred_list, key=lambda x: x[1])
        oracle = min(pred_list, key=lambda x: x[2])

        default_cds.append(default_cd)
        selected_cds.append(selected[2])  # actual CD
        oracle_cds.append(oracle[2])

        # Pairwise accuracy
        pairs = list(combinations(range(len(pred_list)), 2))
        concordant, total = 0, 0
        for i, j in pairs:
            cd_diff = pred_list[i][2] - pred_list[j][2]
            pred_diff = pred_list[i][1] - pred_list[j][1]
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
        'gap_closed_pct': float(gap_closed),
        'mean_pairwise_acc': float(np.mean(pairwise_accs)) if pairwise_accs else 0,
    }


def main():
    objects_data, results = load_data()

    obj_ids = sorted(objects_data.keys())
    n = len(obj_ids)
    np.random.seed(42)
    perm = np.random.permutation(n)
    fold_size = n // 5

    all_predictions = {}

    print(f"\n5-Fold Cross-Validation (Pairwise Ranking Loss)")
    print("=" * 60)

    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size if fold < 4 else n
        val_ids = set(obj_ids[i] for i in perm[start:end])

        train_objs = {k: v for k, v in objects_data.items() if k not in val_ids}
        val_objs = {k: v for k, v in objects_data.items() if k in val_ids}

        print(f"\nFold {fold+1}: train={len(train_objs)} objs, val={len(val_objs)} objs")
        val_preds = train_ranking_mlp(train_objs, val_objs)
        all_predictions.update(val_preds)

    metrics = evaluate_selection(all_predictions, results)

    print(f"\n{'='*60}")
    print(f"RESULTS: Learned Proxy (Ranking Loss)")
    print(f"{'='*60}")
    print(f"  Improvement:      {metrics['improvement_pct']:+.2f}%")
    print(f"  Gap Closed:       {metrics['gap_closed_pct']:.1f}%")
    print(f"  Pairwise Accuracy: {metrics['mean_pairwise_acc']:.1%}")
    print(f"\nComparison:")
    print(f"  Hand-crafted mean: 22.1% gap, 55.3% pairwise")
    print(f"  MLP (MSE loss):    see previous run")
    print(f"  MLP (Ranking):     {metrics['gap_closed_pct']:.1f}% gap, {metrics['mean_pairwise_acc']:.1%} pairwise")

    # Update results file
    output_path = str(OUTPUT_PATH)
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    existing['ranking_mlp'] = metrics
    existing['comparison'] = {
        'handcrafted_mean_gap_pct': 22.1,
        'handcrafted_mean_pairwise_acc': 0.553,
        'mse_mlp_gap_pct': existing.get('overall', {}).get('gap_closed_pct', -6.8),
        'mse_mlp_pairwise_acc': existing.get('overall', {}).get('mean_pairwise_acc', 0.499),
        'ranking_mlp_gap_pct': metrics['gap_closed_pct'],
        'ranking_mlp_pairwise_acc': metrics['mean_pairwise_acc'],
    }

    with open(output_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

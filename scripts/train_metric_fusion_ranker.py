#!/usr/bin/env python3
"""Train a multi-metric fusion ranker and evaluate as a stronger verifier baseline.

Object-level 5-fold CV:
  - Train pairwise logistic ranker on metric features.
  - Predict per-seed utility on held-out objects.
  - Select best seed and report CD improvement / gap / pairwise accuracy.
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


DEFAULT_RESULTS = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/results.json"
DEFAULT_SCORES = "/root/eccv/.worktrees/seedselect80/code/outputs/multimetric/difix_multimetric_scores_gso300.json"
DEFAULT_OUT = "/root/eccv/.worktrees/seedselect80/code/outputs/fusion_ranker/fusion_ranker_results.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion ranking baseline")
    parser.add_argument("--results_json", default=DEFAULT_RESULTS)
    parser.add_argument("--scores_json", default=DEFAULT_SCORES)
    parser.add_argument(
        "--feature_keys",
        default="difix_mv_mean,difix_mv_ssim,difix_mv_l1,difix_front,difix_front_ssim,difix_front_l1",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_seed", default="42")
    parser.add_argument("--out_json", default=DEFAULT_OUT)
    return parser.parse_args()


def build_data(results_json, scores_json, feature_keys):
    results = load_json(results_json)
    scores = load_json(scores_json)["scores"]

    objects = sorted(results.keys())
    gt = {}
    feats = {}
    for obj in objects:
        seed_payload = results[obj].get("seeds", {})
        gt_obj = {str(s): float(v["cd"]) for s, v in seed_payload.items() if "cd" in v}
        if len(gt_obj) < 2:
            continue

        # Shared seeds across all features.
        shared = set(gt_obj.keys())
        for key in feature_keys:
            if key not in scores or obj not in scores[key]:
                shared = set()
                break
            shared &= set(str(s) for s in scores[key][obj].keys())
        if len(shared) < 2:
            continue

        gt[obj] = {s: gt_obj[s] for s in sorted(shared)}
        feats[obj] = {}
        for s in sorted(shared):
            feats[obj][s] = np.array([float(scores[k][obj][s]) for k in feature_keys], dtype=np.float32)

    return gt, feats, sorted(gt.keys())


def make_pairwise_samples(obj_ids, gt, feats):
    x, y = [], []
    for obj in obj_ids:
        seeds = sorted(gt[obj].keys())
        for si, sj in combinations(seeds, 2):
            fi = feats[obj][si]
            fj = feats[obj][sj]
            # label 1 means i better (lower CD).
            yi = 1 if gt[obj][si] < gt[obj][sj] else 0

            x.append(fi - fj)
            y.append(yi)
            x.append(fj - fi)
            y.append(1 - yi)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def evaluate_model(model, obj_ids, gt, feats, default_seed):
    selected_cds = []
    default_cds = []
    oracle_cds = []
    per_obj_pairwise = []

    wins = ties = losses = 0
    for obj in obj_ids:
        seeds = sorted(gt[obj].keys())
        X = np.stack([feats[obj][s] for s in seeds], axis=0)
        utility = X @ model.coef_.ravel()
        sel_idx = int(np.argmax(utility))
        sel_seed = seeds[sel_idx]

        d_seed = default_seed if default_seed in seeds else seeds[0]
        o_seed = min(seeds, key=lambda s: gt[obj][s])

        cd_d = gt[obj][d_seed]
        cd_s = gt[obj][sel_seed]
        cd_o = gt[obj][o_seed]
        default_cds.append(cd_d)
        selected_cds.append(cd_s)
        oracle_cds.append(cd_o)

        if cd_s < cd_d:
            wins += 1
        elif cd_s > cd_d:
            losses += 1
        else:
            ties += 1

        # Pairwise ranking accuracy from utility.
        c = 0
        d = 0
        for i, j in combinations(range(len(seeds)), 2):
            pred = utility[i] > utility[j]
            truth = gt[obj][seeds[i]] < gt[obj][seeds[j]]
            if pred == truth:
                c += 1
            else:
                d += 1
        per_obj_pairwise.append(c / (c + d) if (c + d) else 0.5)

    default_cds = np.asarray(default_cds)
    selected_cds = np.asarray(selected_cds)
    oracle_cds = np.asarray(oracle_cds)
    pair = np.asarray(per_obj_pairwise)

    default_mean = float(default_cds.mean())
    selected_mean = float(selected_cds.mean())
    oracle_mean = float(oracle_cds.mean())
    improvement = (default_mean - selected_mean) / max(default_mean, 1e-12) * 100.0
    gap_closed = (default_mean - selected_mean) / max(default_mean - oracle_mean, 1e-12) * 100.0
    try:
        p = float(wilcoxon(selected_cds, default_cds, alternative="less").pvalue)
    except ValueError:
        p = 1.0

    return {
        "n_objects": int(len(obj_ids)),
        "default_cd": default_mean,
        "selected_cd": selected_mean,
        "oracle_cd": oracle_mean,
        "improvement_pct": float(improvement),
        "gap_closed_pct": float(gap_closed),
        "mean_pairwise_acc_pct": float(pair.mean() * 100.0),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "p_value": p,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    feature_keys = [x.strip() for x in args.feature_keys.split(",") if x.strip()]

    gt, feats, objects = build_data(args.results_json, args.scores_json, feature_keys)
    if len(objects) < args.n_splits:
        raise RuntimeError(f"Not enough objects ({len(objects)}) for {args.n_splits}-fold CV.")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_reports = []
    all_test_objects = []
    all_selected = []
    all_default = []
    all_oracle = []
    all_pairwise = []
    total_wins = total_ties = total_losses = 0
    predicted_scores = {}

    objects_arr = np.asarray(objects)
    for fold_id, (tr, te) in enumerate(kf.split(objects_arr), start=1):
        train_objs = objects_arr[tr].tolist()
        test_objs = objects_arr[te].tolist()
        x_train, y_train = make_pairwise_samples(train_objs, gt, feats)

        model = LogisticRegression(
            fit_intercept=False,
            max_iter=2000,
            solver="lbfgs",
            random_state=args.seed + fold_id,
        )
        model.fit(x_train, y_train)

        report = evaluate_model(model, test_objs, gt, feats, args.default_seed)
        report["fold"] = fold_id
        report["n_train_pairs"] = int(len(y_train))
        report["weights"] = {k: float(v) for k, v in zip(feature_keys, model.coef_.ravel())}
        fold_reports.append(report)

        # Recompute per-object details for merged summary.
        for obj in test_objs:
            seeds = sorted(gt[obj].keys())
            X = np.stack([feats[obj][s] for s in seeds], axis=0)
            utility = X @ model.coef_.ravel()
            sel_seed = seeds[int(np.argmax(utility))]
            d_seed = args.default_seed if args.default_seed in seeds else seeds[0]
            o_seed = min(seeds, key=lambda s: gt[obj][s])

            cd_d = gt[obj][d_seed]
            cd_s = gt[obj][sel_seed]
            cd_o = gt[obj][o_seed]

            all_default.append(cd_d)
            all_selected.append(cd_s)
            all_oracle.append(cd_o)
            all_test_objects.append(obj)

            if cd_s < cd_d:
                total_wins += 1
            elif cd_s > cd_d:
                total_losses += 1
            else:
                total_ties += 1

            c = 0
            d = 0
            for i, j in combinations(range(len(seeds)), 2):
                pred = utility[i] > utility[j]
                truth = gt[obj][seeds[i]] < gt[obj][seeds[j]]
                if pred == truth:
                    c += 1
                else:
                    d += 1
            all_pairwise.append(c / (c + d) if (c + d) else 0.5)

            predicted_scores[obj] = {s: float(u) for s, u in zip(seeds, utility)}

    all_default = np.asarray(all_default)
    all_selected = np.asarray(all_selected)
    all_oracle = np.asarray(all_oracle)
    all_pairwise = np.asarray(all_pairwise)

    overall = {
        "n_objects": int(len(all_test_objects)),
        "default_cd": float(all_default.mean()),
        "selected_cd": float(all_selected.mean()),
        "oracle_cd": float(all_oracle.mean()),
        "improvement_pct": float((all_default.mean() - all_selected.mean()) / max(all_default.mean(), 1e-12) * 100.0),
        "gap_closed_pct": float((all_default.mean() - all_selected.mean()) / max(all_default.mean() - all_oracle.mean(), 1e-12) * 100.0),
        "mean_pairwise_acc_pct": float(all_pairwise.mean() * 100.0),
        "wins": int(total_wins),
        "ties": int(total_ties),
        "losses": int(total_losses),
        "p_value": float(wilcoxon(all_selected, all_default, alternative="less").pvalue),
        "feature_keys": feature_keys,
    }

    payload = {"overall": overall, "folds": fold_reports, "predicted_scores": predicted_scores}
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {args.out_json}")
    print(
        f"Fusion ranker: improv={overall['improvement_pct']:+.2f}% | "
        f"gap={overall['gap_closed_pct']:.1f}% | "
        f"pairwise={overall['mean_pairwise_acc_pct']:.1f}% | "
        f"p={overall['p_value']:.3g}"
    )


if __name__ == "__main__":
    main()

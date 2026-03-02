#!/usr/bin/env python3
"""Confidence-based guardrail for SeedSelect.

Idea:
If the top-1 proxy score is only marginally better than top-2, abstain and
fallback to the default seed (seed 42). This controls catastrophic picks.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_RESULTS = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/results.json"
DEFAULT_SCORES = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/difix_multiview_scores.json"
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "guardrail")


@dataclass
class EvalRow:
    threshold: float
    abstain_rate_pct: float
    improvement_pct: float
    gap_closed_pct: float
    win_rate_pct: float
    worst_pick_rate_pct: float
    p_value: float
    selected_cd: float
    default_cd: float
    oracle_cd: float
    n_objects: int


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def score_margin(scores: Dict[str, float]) -> float:
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return 0.0
    return float(vals[0] - vals[1])


def evaluate(
    gt: Dict[str, Dict[str, float]],
    proxy: Dict[str, Dict[str, float]],
    threshold: float,
    default_seed: str,
) -> EvalRow:
    default_cds = []
    selected_cds = []
    oracle_cds = []

    wins = 0
    worst_pick = 0
    abstains = 0
    used = 0

    for obj, gt_obj in gt.items():
        if obj not in proxy:
            continue
        sc_obj = proxy[obj]
        shared = sorted(set(gt_obj.keys()) & set(sc_obj.keys()))
        if len(shared) < 2:
            continue

        seed_default = default_seed if default_seed in shared else shared[0]
        shared_scores = {s: float(sc_obj[s]) for s in shared}
        ranking = sorted(shared_scores.items(), key=lambda x: x[1], reverse=True)
        top_seed = ranking[0][0]
        margin = ranking[0][1] - ranking[1][1]

        if margin < threshold:
            selected = seed_default
            abstains += 1
        else:
            selected = top_seed

        oracle = min(shared, key=lambda s: gt_obj[s])
        worst = max(shared, key=lambda s: gt_obj[s])

        cd_d = float(gt_obj[seed_default])
        cd_s = float(gt_obj[selected])
        cd_o = float(gt_obj[oracle])

        default_cds.append(cd_d)
        selected_cds.append(cd_s)
        oracle_cds.append(cd_o)

        if cd_s < cd_d:
            wins += 1
        if selected == worst:
            worst_pick += 1

        used += 1

    default_cds = np.asarray(default_cds)
    selected_cds = np.asarray(selected_cds)
    oracle_cds = np.asarray(oracle_cds)

    default_mean = float(np.mean(default_cds))
    selected_mean = float(np.mean(selected_cds))
    oracle_mean = float(np.mean(oracle_cds))

    improvement = (default_mean - selected_mean) / max(default_mean, 1e-12) * 100.0
    gap = (default_mean - selected_mean) / max(default_mean - oracle_mean, 1e-12) * 100.0
    abstain_rate = abstains / max(used, 1) * 100.0
    win_rate = wins / max(used, 1) * 100.0
    worst_rate = worst_pick / max(used, 1) * 100.0

    try:
        pval = float(wilcoxon(selected_cds, default_cds, alternative="less").pvalue)
    except ValueError:
        pval = 1.0

    return EvalRow(
        threshold=float(threshold),
        abstain_rate_pct=float(abstain_rate),
        improvement_pct=float(improvement),
        gap_closed_pct=float(gap),
        win_rate_pct=float(win_rate),
        worst_pick_rate_pct=float(worst_rate),
        p_value=pval,
        selected_cd=selected_mean,
        default_cd=default_mean,
        oracle_cd=oracle_mean,
        n_objects=int(used),
    )


def rows_to_csv(rows: List[EvalRow], path: str):
    fieldnames = [
        "threshold",
        "abstain_rate_pct",
        "improvement_pct",
        "gap_closed_pct",
        "win_rate_pct",
        "worst_pick_rate_pct",
        "p_value",
        "selected_cd",
        "default_cd",
        "oracle_cd",
        "n_objects",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate confidence-abstain guardrail")
    parser.add_argument("--results_json", default=DEFAULT_RESULTS)
    parser.add_argument("--scores_json", default=DEFAULT_SCORES)
    parser.add_argument("--score_key", default="scores.difix_mv_mean")
    parser.add_argument("--default_seed", default="42")
    parser.add_argument("--target_worst_pct", type=float, default=10.0)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt_raw = load_json(args.results_json)
    gt = {}
    for obj, payload in gt_raw.items():
        seeds = payload.get("seeds", {})
        cds = {str(s): float(v["cd"]) for s, v in seeds.items() if "cd" in v}
        if len(cds) >= 2:
            gt[obj] = cds

    scores_raw = load_json(args.scores_json)
    proxy = scores_raw
    for key in args.score_key.split("."):
        proxy = proxy[key]
    proxy = {
        obj: {str(seed): float(score) for seed, score in seed_scores.items()}
        for obj, seed_scores in proxy.items()
        if isinstance(seed_scores, dict)
    }

    margins = []
    for obj in sorted(set(gt.keys()) & set(proxy.keys())):
        shared = sorted(set(gt[obj].keys()) & set(proxy[obj].keys()))
        if len(shared) < 2:
            continue
        margins.append(score_margin({s: proxy[obj][s] for s in shared}))

    margins = np.asarray(margins)
    quantiles = np.linspace(0.0, 0.9, 10)
    thresholds = sorted(set([0.0] + [float(np.quantile(margins, q)) for q in quantiles]))

    rows = [evaluate(gt, proxy, t, default_seed=args.default_seed) for t in thresholds]
    rows = sorted(rows, key=lambda x: x.threshold)

    feasible = [r for r in rows if r.worst_pick_rate_pct <= args.target_worst_pct]
    if feasible:
        best = max(feasible, key=lambda r: r.improvement_pct)
    else:
        best = max(rows, key=lambda r: (r.improvement_pct - 0.05 * r.worst_pick_rate_pct))

    json_path = os.path.join(args.out_dir, "guardrail_abstain_results.json")
    csv_path = os.path.join(args.out_dir, "guardrail_abstain_results.csv")
    rows_to_csv(rows, csv_path)
    with open(json_path, "w") as f:
        json.dump(
            {
                "target_worst_pct": args.target_worst_pct,
                "best": best.__dict__,
                "sweep": [r.__dict__ for r in rows],
            },
            f,
            indent=2,
        )

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print("\nBest threshold under constraint:")
    print(
        f"  thr={best.threshold:.6f} | abstain={best.abstain_rate_pct:.1f}% | "
        f"improv={best.improvement_pct:+.2f}% | worst={best.worst_pick_rate_pct:.1f}% | "
        f"p={best.p_value:.3g}"
    )


if __name__ == "__main__":
    main()

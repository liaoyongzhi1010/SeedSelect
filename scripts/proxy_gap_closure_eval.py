#!/usr/bin/env python3
"""Close proxy-only gap with independent GT geometry evaluation.

Evaluates SeedSelect selection on:
  - InstantMesh (GSO-300, GT CD from results.json)
  - LGM         (GSO-300, GT CD from gt_eval_results.json)

Both use the same proxy-style seed scorer (Difix-style scores for InstantMesh,
LGM stored seed scores for LGM), then report independent CD improvement with CI.
"""

import argparse
import csv
import json
import os
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
from scipy.stats import wilcoxon

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUT = "/root/eccv/.worktrees/seedselect80/code/outputs/proxy_gap_closure"


def parse_args():
    parser = argparse.ArgumentParser(description="Independent GT geometry closure")
    parser.add_argument(
        "--instant_results_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/results.json",
    )
    parser.add_argument(
        "--instant_scores_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/difix_multiview_scores.json",
    )
    parser.add_argument("--instant_score_key", default="scores.difix_mv_mean")
    parser.add_argument(
        "--lgm_gt_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/lgm_gso/gt_eval_results.json",
    )
    parser.add_argument(
        "--lgm_scores_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/lgm_gso/seedselect_results.json",
    )
    parser.add_argument("--default_seed", default="42")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def nested_get(d: dict, key_path: str):
    cur = d
    for key in key_path.split("."):
        cur = cur[key]
    return cur


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(float(np.mean(values[idx])))
    lo, hi = np.percentile(np.asarray(stats), [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_ratio(default_cd: np.ndarray, selected_cd: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(default_cd)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = float(np.mean(default_cd[idx]))
        s = float(np.mean(selected_cd[idx]))
        vals.append((d - s) / max(d, 1e-12) * 100.0)
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi)


def pairwise_acc(gt_obj: Dict[str, float], sc_obj: Dict[str, float], higher_is_better: bool = True) -> float:
    seeds = sorted(set(gt_obj.keys()) & set(sc_obj.keys()))
    c, d = 0, 0
    for si, sj in combinations(seeds, 2):
        gt_pref = gt_obj[si] < gt_obj[sj]
        pred_pref = sc_obj[si] > sc_obj[sj] if higher_is_better else sc_obj[si] < sc_obj[sj]
        if gt_obj[si] == gt_obj[sj] or sc_obj[si] == sc_obj[sj]:
            continue
        if gt_pref == pred_pref:
            c += 1
        else:
            d += 1
    if c + d == 0:
        return 0.5
    return c / (c + d)


def evaluate_from_gt_and_scores(
    gt_cd: Dict[str, Dict[str, float]],
    scores: Dict[str, Dict[str, float]],
    default_seed: str = "42",
    higher_is_better: bool = True,
    bootstrap: int = 4000,
    seed: int = 42,
) -> dict:
    objs = sorted(set(gt_cd.keys()) & set(scores.keys()))
    default_arr = []
    selected_arr = []
    oracle_arr = []
    pairwise_arr = []
    per_object = []

    wins = ties = losses = severe = 0
    for obj in objs:
        gt_obj = gt_cd[obj]
        sc_obj = scores[obj]
        shared = sorted(set(gt_obj.keys()) & set(sc_obj.keys()))
        if len(shared) < 2:
            continue
        if default_seed not in shared:
            continue

        if higher_is_better:
            selected = max(shared, key=lambda s: sc_obj[s])
        else:
            selected = min(shared, key=lambda s: sc_obj[s])
        oracle = min(shared, key=lambda s: gt_obj[s])
        default = default_seed

        cd_d = float(gt_obj[default])
        cd_s = float(gt_obj[selected])
        cd_o = float(gt_obj[oracle])
        delta = cd_d - cd_s
        severe_flag = (cd_s - cd_d) / max(cd_d, 1e-12) > 0.05
        if severe_flag:
            severe += 1

        if cd_s < cd_d:
            wins += 1
        elif cd_s > cd_d:
            losses += 1
        else:
            ties += 1

        default_arr.append(cd_d)
        selected_arr.append(cd_s)
        oracle_arr.append(cd_o)
        pairwise_arr.append(pairwise_acc(gt_obj, sc_obj, higher_is_better))
        per_object.append(
            {
                "object_id": obj,
                "default_seed": default,
                "selected_seed": selected,
                "oracle_seed": oracle,
                "default_cd": cd_d,
                "selected_cd": cd_s,
                "oracle_cd": cd_o,
                "delta_cd": delta,
                "degrade_pct": (cd_s - cd_d) / max(cd_d, 1e-12) * 100.0,
            }
        )

    default_arr = np.asarray(default_arr, dtype=np.float64)
    selected_arr = np.asarray(selected_arr, dtype=np.float64)
    oracle_arr = np.asarray(oracle_arr, dtype=np.float64)
    pairwise_arr = np.asarray(pairwise_arr, dtype=np.float64)

    d_mean = float(np.mean(default_arr))
    s_mean = float(np.mean(selected_arr))
    o_mean = float(np.mean(oracle_arr))
    improvement_pct = (d_mean - s_mean) / max(d_mean, 1e-12) * 100.0
    gap_closed_pct = (d_mean - s_mean) / max(d_mean - o_mean, 1e-12) * 100.0
    delta_ci = bootstrap_ci(default_arr - selected_arr, bootstrap, seed + 10)
    imp_ci = bootstrap_ci_ratio(default_arr, selected_arr, bootstrap, seed + 20)
    try:
        pval = float(wilcoxon(selected_arr, default_arr, alternative="less").pvalue)
    except ValueError:
        pval = 1.0

    return {
        "n_objects": int(len(default_arr)),
        "default_cd": d_mean,
        "selected_cd": s_mean,
        "oracle_cd": o_mean,
        "improvement_pct": float(improvement_pct),
        "improvement_pct_ci95": imp_ci,
        "delta_cd_mean": float(np.mean(default_arr - selected_arr)),
        "delta_cd_ci95": delta_ci,
        "gap_closed_pct": float(gap_closed_pct),
        "pairwise_acc_pct": float(np.mean(pairwise_arr) * 100.0),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "severe_degrade_rate_pct": float(severe / max(len(default_arr), 1) * 100.0),
        "wilcoxon_p": pval,
        "per_object": per_object,
    }


def build_instant_gt_scores(results_json: str, scores_json: str, score_key: str):
    results = load_json(results_json)
    score_pack = load_json(scores_json)
    score = nested_get(score_pack, score_key)

    gt = {}
    sc = {}
    for obj, payload in results.items():
        if obj not in score:
            continue
        cds = {str(s): float(v["cd"]) for s, v in payload.get("seeds", {}).items() if "cd" in v}
        if len(cds) < 2:
            continue
        gt[obj] = cds
        sc[obj] = {str(s): float(v) for s, v in score[obj].items()}
    return gt, sc


def build_lgm_gt_scores(lgm_gt_json: str, lgm_scores_json: str):
    gt_pack = load_json(lgm_gt_json)
    score_pack = load_json(lgm_scores_json)
    gt = {}
    for obj, seeds in gt_pack.get("per_object", {}).items():
        cds = {}
        for s, payload in seeds.items():
            if isinstance(payload, dict) and "cd" in payload:
                cds[str(s)] = float(payload["cd"])
        if len(cds) >= 2:
            gt[obj] = cds
    sc = {obj: {str(s): float(v) for s, v in seeds.items()} for obj, seeds in score_pack.get("scores", {}).items()}
    return gt, sc


def save_per_object_csv(path: str, rows):
    fieldnames = [
        "backbone",
        "object_id",
        "default_seed",
        "selected_seed",
        "oracle_seed",
        "default_cd",
        "selected_cd",
        "oracle_cd",
        "delta_cd",
        "degrade_pct",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_bar_ci(path: str, inst: dict, lgm: dict):
    names = ["InstantMesh", "LGM"]
    vals = [inst["improvement_pct"], lgm["improvement_pct"]]
    ci = [inst["improvement_pct_ci95"], lgm["improvement_pct_ci95"]]
    yerr_low = [vals[i] - ci[i][0] for i in range(2)]
    yerr_high = [ci[i][1] - vals[i] for i in range(2)]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    colors = ["#1f77b4", "#ff7f0e"]
    ax.bar(names, vals, color=colors, alpha=0.9)
    ax.errorbar(names, vals, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.axhline(0.0, color="#999999", linewidth=1.0)
    ax.set_ylabel("CD Improvement (%)")
    ax.set_title("Independent GT Geometry Evaluation (with 95% CI)")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_scatter(path: str, inst_rows, lgm_rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=False, sharey=False)
    for ax, rows, title, color in [
        (axes[0], inst_rows, "InstantMesh", "#1f77b4"),
        (axes[1], lgm_rows, "LGM", "#ff7f0e"),
    ]:
        x = np.asarray([r["default_cd"] for r in rows], dtype=np.float64)
        y = np.asarray([r["selected_cd"] for r in rows], dtype=np.float64)
        mn = min(float(np.min(x)), float(np.min(y)))
        mx = max(float(np.max(x)), float(np.max(y)))
        ax.scatter(x, y, s=10, alpha=0.5, color=color)
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        ax.set_xlabel("Default CD")
        ax.set_ylabel("SeedSelect CD")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.suptitle("Per-object GT-CD: default vs selected")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    instant_gt, instant_sc = build_instant_gt_scores(args.instant_results_json, args.instant_scores_json, args.instant_score_key)
    lgm_gt, lgm_sc = build_lgm_gt_scores(args.lgm_gt_json, args.lgm_scores_json)

    instant = evaluate_from_gt_and_scores(
        instant_gt,
        instant_sc,
        default_seed=args.default_seed,
        higher_is_better=True,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    lgm = evaluate_from_gt_and_scores(
        lgm_gt,
        lgm_sc,
        default_seed=args.default_seed,
        higher_is_better=True,
        bootstrap=args.bootstrap,
        seed=args.seed + 1000,
    )

    rows = []
    for name, report in [("InstantMesh", instant), ("LGM", lgm)]:
        rows.append(
            {
                "backbone": name,
                "n_objects": report["n_objects"],
                "default_cd": report["default_cd"],
                "selected_cd": report["selected_cd"],
                "oracle_cd": report["oracle_cd"],
                "improvement_pct": report["improvement_pct"],
                "improvement_ci95_lo": report["improvement_pct_ci95"][0],
                "improvement_ci95_hi": report["improvement_pct_ci95"][1],
                "gap_closed_pct": report["gap_closed_pct"],
                "pairwise_acc_pct": report["pairwise_acc_pct"],
                "wins": report["wins"],
                "ties": report["ties"],
                "losses": report["losses"],
                "severe_degrade_rate_pct": report["severe_degrade_rate_pct"],
                "wilcoxon_p": report["wilcoxon_p"],
            }
        )

    summary_csv = os.path.join(args.out_dir, "proxy_gap_closure_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    per_object_rows = []
    for row in instant["per_object"]:
        r = dict(row)
        r["backbone"] = "InstantMesh"
        per_object_rows.append(r)
    for row in lgm["per_object"]:
        r = dict(row)
        r["backbone"] = "LGM"
        per_object_rows.append(r)
    per_obj_csv = os.path.join(args.out_dir, "proxy_gap_closure_per_object.csv")
    save_per_object_csv(per_obj_csv, per_object_rows)

    bar_fig = os.path.join(args.out_dir, "proxy_gap_closure_bar_ci.pdf")
    scatter_fig = os.path.join(args.out_dir, "proxy_gap_closure_scatter.pdf")
    plot_bar_ci(bar_fig, instant, lgm)
    plot_scatter(scatter_fig, instant["per_object"], lgm["per_object"])

    out_json = os.path.join(args.out_dir, "proxy_gap_closure_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "InstantMesh": {k: v for k, v in instant.items() if k != "per_object"},
                "LGM": {k: v for k, v in lgm.items() if k != "per_object"},
                "artifacts": {
                    "summary_csv": summary_csv,
                    "per_object_csv": per_obj_csv,
                    "bar_fig": bar_fig,
                    "scatter_fig": scatter_fig,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved {out_json}")
    print(f"Saved {summary_csv}")
    print("Independent geometry summary:")
    for row in rows:
        print(
            f"  {row['backbone']}: n={row['n_objects']}, "
            f"improv={row['improvement_pct']:+.2f}% "
            f"[{row['improvement_ci95_lo']:+.2f}, {row['improvement_ci95_hi']:+.2f}], "
            f"p={row['wilcoxon_p']:.3g}, severe={row['severe_degrade_rate_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()


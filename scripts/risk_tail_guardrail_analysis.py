#!/usr/bin/env python3
"""Risk-tail analysis across K and guardrail risk-coverage evaluation.

Outputs:
  - Tail risk vs K (P_fail, P_severe, CVaR@5, Q5/Q1)
  - Guardrail risk-coverage sweep with val threshold selection and frozen test
"""

import argparse
import csv
import hashlib
import json
import os
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUT = "/root/eccv/.worktrees/seedselect80/code/outputs/risk_tail"


def parse_args():
    parser = argparse.ArgumentParser(description="Risk-tail + guardrail analysis")
    parser.add_argument(
        "--k_results_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_k8/results.json",
    )
    parser.add_argument(
        "--k_scores_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_k8/difix_multiview_scores.json",
    )
    parser.add_argument("--k_score_key", default="scores.difix_mv_mean")
    parser.add_argument("--k_values", default="2,4,6,8,10,12,14,16")
    parser.add_argument(
        "--guard_results_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/results.json",
    )
    parser.add_argument(
        "--guard_scores_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/difix_multiview_scores.json",
    )
    parser.add_argument("--guard_score_key", default="scores.difix_mv_mean")
    parser.add_argument("--default_seed", default="42")
    parser.add_argument("--severe_threshold_pct", type=float, default=5.0)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def nested_get(d, key_path: str):
    cur = d
    for k in key_path.split("."):
        cur = cur[k]
    return cur


def parse_int_list(txt: str):
    return [int(x.strip()) for x in txt.split(",") if x.strip()]


def stable_split(obj_ids: List[str], val_ratio: float):
    val, test = [], []
    for obj in sorted(obj_ids):
        h = hashlib.md5(obj.encode("utf-8")).hexdigest()
        x = int(h[:8], 16) / float(0xFFFFFFFF)
        if x < val_ratio:
            val.append(obj)
        else:
            test.append(obj)
    return val, test


def risk_metrics_from_delta(delta_pct: np.ndarray, severe_threshold_pct: float):
    if len(delta_pct) == 0:
        return {
            "n": 0,
            "improvement_pct_mean": 0.0,
            "P_fail": 0.0,
            "P_severe": 0.0,
            "CVaR5": 0.0,
            "Q1": 0.0,
            "Q5": 0.0,
            "median": 0.0,
        }
    sorted_delta = np.sort(delta_pct)
    tail_n = max(1, int(np.ceil(0.05 * len(sorted_delta))))
    cvar5 = float(np.mean(sorted_delta[:tail_n]))
    return {
        "n": int(len(delta_pct)),
        "improvement_pct_mean": float(np.mean(delta_pct)),
        "P_fail": float(np.mean(delta_pct < 0.0)),
        "P_severe": float(np.mean(delta_pct < -severe_threshold_pct)),
        "CVaR5": cvar5,
        "Q1": float(np.percentile(delta_pct, 1)),
        "Q5": float(np.percentile(delta_pct, 5)),
        "median": float(np.percentile(delta_pct, 50)),
    }


def eval_k_tail(results_json: str, scores_json: str, score_key: str, k_values: List[int], default_seed: str, severe_threshold_pct: float):
    results = load_json(results_json)
    scores = nested_get(load_json(scores_json), score_key)

    out = []
    for k in k_values:
        deltas = []
        for obj, payload in results.items():
            if obj not in scores:
                continue
            gt = {str(s): float(v["cd"]) for s, v in payload.get("seeds", {}).items() if "cd" in v}
            sc = {str(s): float(v) for s, v in scores[obj].items()}
            if default_seed not in gt:
                continue
            order = [s for s in sc.keys() if s in gt]
            if len(order) < k:
                continue
            cand = order[:k]
            selected = max(cand, key=lambda s: sc[s])

            d_cd = gt[default_seed]
            s_cd = gt[selected]
            delta_pct = (d_cd - s_cd) / max(d_cd, 1e-12) * 100.0
            deltas.append(delta_pct)
        metrics = risk_metrics_from_delta(np.asarray(deltas, dtype=np.float64), severe_threshold_pct)
        metrics["K"] = k
        out.append(metrics)
    return out


def guardrail_sweep(results_json: str, scores_json: str, score_key: str, default_seed: str, severe_threshold_pct: float, val_ratio: float):
    results = load_json(results_json)
    scores = nested_get(load_json(scores_json), score_key)

    # Build object-level records.
    records = {}
    for obj, payload in results.items():
        if obj not in scores:
            continue
        gt = {str(s): float(v["cd"]) for s, v in payload.get("seeds", {}).items() if "cd" in v}
        sc = {str(s): float(v) for s, v in scores[obj].items()}
        shared = sorted(set(gt.keys()) & set(sc.keys()))
        if len(shared) < 2 or default_seed not in shared:
            continue
        ranked = sorted(shared, key=lambda s: sc[s], reverse=True)
        best = ranked[0]
        second = ranked[1]
        margin = sc[best] - sc[second]

        d_cd = gt[default_seed]
        s_cd = gt[best]
        delta_pct = (d_cd - s_cd) / max(d_cd, 1e-12) * 100.0
        records[obj] = {
            "margin": float(margin),
            "delta_pct_accept": float(delta_pct),
            "delta_pct_fallback": 0.0,  # fallback = default
        }

    obj_ids = sorted(records.keys())
    val_ids, test_ids = stable_split(obj_ids, val_ratio=val_ratio)

    val_margins = np.asarray([records[o]["margin"] for o in val_ids], dtype=np.float64)
    # 0..90% quantile thresholds
    thresholds = sorted(set([0.0] + [float(np.quantile(val_margins, q)) for q in np.linspace(0.0, 0.9, 19)]))

    def eval_subset(ids: List[str], thr: float):
        deltas = []
        accepted = 0
        for o in ids:
            rec = records[o]
            if rec["margin"] >= thr:
                deltas.append(rec["delta_pct_accept"])
                accepted += 1
            else:
                deltas.append(rec["delta_pct_fallback"])
        m = risk_metrics_from_delta(np.asarray(deltas, dtype=np.float64), severe_threshold_pct)
        m["coverage"] = float(accepted / max(len(ids), 1))
        m["threshold"] = float(thr)
        return m

    val_curve = [eval_subset(val_ids, t) for t in thresholds]
    test_curve = [eval_subset(test_ids, t) for t in thresholds]

    # Select threshold on val only.
    feasible = [m for m in val_curve if m["P_severe"] <= severe_threshold_pct / 100.0]
    if feasible:
        selected = max(feasible, key=lambda m: m["improvement_pct_mean"])
    else:
        selected = max(val_curve, key=lambda m: (m["improvement_pct_mean"] - 0.8 * m["P_severe"] * 100.0))
    selected_thr = selected["threshold"]
    frozen_test = eval_subset(test_ids, selected_thr)

    return {
        "n_objects": len(obj_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "thresholds": thresholds,
        "val_curve": val_curve,
        "test_curve": test_curve,
        "selected_on_val": selected,
        "frozen_test": frozen_test,
    }


def save_csv(path: str, rows: List[dict], field_order: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_tail_vs_k(path: str, rows: List[dict]):
    ks = [r["K"] for r in rows]
    p_fail = [100.0 * r["P_fail"] for r in rows]
    p_severe = [100.0 * r["P_severe"] for r in rows]
    cvar = [r["CVaR5"] for r in rows]
    q5 = [r["Q5"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax = axes[0]
    ax.plot(ks, p_fail, "o-", label="$P_{fail}(K)$")
    ax.plot(ks, p_severe, "s-", label="$P_{severe}(K)$")
    ax.set_xlabel("K")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Failure Tail vs K")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ks, cvar, "o-", label="CVaR@5%")
    ax.plot(ks, q5, "s--", label="Q5")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xlabel("K")
    ax.set_ylabel("Delta% (default - selected)")
    ax.set_title("Tail Magnitude vs K")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_risk_coverage(path: str, sweep: dict):
    val = sweep["val_curve"]
    test = sweep["test_curve"]
    thr = sweep["selected_on_val"]["threshold"]
    frozen = sweep["frozen_test"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    ax = axes[0]
    ax.plot([100 * m["coverage"] for m in val], [100 * m["P_severe"] for m in val], "o-", label="Val severe-risk")
    ax.plot([100 * m["coverage"] for m in test], [100 * m["P_severe"] for m in test], "s--", label="Test severe-risk")
    ax.scatter([100 * frozen["coverage"]], [100 * frozen["P_severe"]], marker="*", s=120, label=f"Frozen test @thr={thr:.4f}")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Severe risk (%)")
    ax.set_title("Risk-Coverage (Severe)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot([100 * m["coverage"] for m in val], [m["improvement_pct_mean"] for m in val], "o-", label="Val improvement")
    ax.plot([100 * m["coverage"] for m in test], [m["improvement_pct_mean"] for m in test], "s--", label="Test improvement")
    ax.scatter([100 * frozen["coverage"]], [frozen["improvement_pct_mean"]], marker="*", s=120, label="Frozen test")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Mean improvement (%)")
    ax.set_title("Quality-Coverage")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    k_values = parse_int_list(args.k_values)
    tail_rows = eval_k_tail(
        results_json=args.k_results_json,
        scores_json=args.k_scores_json,
        score_key=args.k_score_key,
        k_values=k_values,
        default_seed=args.default_seed,
        severe_threshold_pct=args.severe_threshold_pct,
    )
    sweep = guardrail_sweep(
        results_json=args.guard_results_json,
        scores_json=args.guard_scores_json,
        score_key=args.guard_score_key,
        default_seed=args.default_seed,
        severe_threshold_pct=args.severe_threshold_pct,
        val_ratio=args.val_ratio,
    )

    tail_csv = os.path.join(args.out_dir, "risk_tail_vs_k.csv")
    save_csv(
        tail_csv,
        tail_rows,
        field_order=["K", "n", "improvement_pct_mean", "P_fail", "P_severe", "CVaR5", "Q1", "Q5", "median"],
    )

    val_csv = os.path.join(args.out_dir, "risk_coverage_val.csv")
    test_csv = os.path.join(args.out_dir, "risk_coverage_test.csv")
    save_csv(
        val_csv,
        sweep["val_curve"],
        field_order=["threshold", "coverage", "improvement_pct_mean", "P_fail", "P_severe", "CVaR5", "Q1", "Q5", "median", "n"],
    )
    save_csv(
        test_csv,
        sweep["test_curve"],
        field_order=["threshold", "coverage", "improvement_pct_mean", "P_fail", "P_severe", "CVaR5", "Q1", "Q5", "median", "n"],
    )

    fig_tail = os.path.join(args.out_dir, "risk_tail_vs_k.pdf")
    fig_cov = os.path.join(args.out_dir, "risk_coverage_guardrail.pdf")
    plot_tail_vs_k(fig_tail, tail_rows)
    plot_risk_coverage(fig_cov, sweep)

    out_json = os.path.join(args.out_dir, "risk_tail_guardrail_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "settings": {
                    "k_values": k_values,
                    "severe_threshold_pct": args.severe_threshold_pct,
                    "val_ratio": args.val_ratio,
                },
                "tail_vs_k": tail_rows,
                "guardrail_sweep": sweep,
                "artifacts": {
                    "tail_csv": tail_csv,
                    "val_csv": val_csv,
                    "test_csv": test_csv,
                    "tail_fig": fig_tail,
                    "risk_coverage_fig": fig_cov,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved {out_json}")
    print(
        f"Guardrail frozen-test: threshold={sweep['selected_on_val']['threshold']:.6f}, "
        f"coverage={100*sweep['frozen_test']['coverage']:.1f}%, "
        f"improvement={sweep['frozen_test']['improvement_pct_mean']:+.2f}%, "
        f"P_severe={100*sweep['frozen_test']['P_severe']:.1f}%"
    )


if __name__ == "__main__":
    main()


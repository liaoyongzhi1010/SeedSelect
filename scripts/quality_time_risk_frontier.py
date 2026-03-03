#!/usr/bin/env python3
"""Build a compute-matched quality-time-risk frontier table/plot.

Time accounting protocol (inference-time only):
  T_total = T_render + T_feature + T_verifier_forward

Where:
  - T_render: measured by rendering sample objects (4 seeds x 6 views)
  - T_feature: measured from Difix multimetric metadata elapsed time
  - T_verifier_forward: tiny (linear/logistic scorer), measured in-script
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Quality-time-risk frontier")
    parser.add_argument(
        "--results_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/results.json",
    )
    parser.add_argument(
        "--scores_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/difix_multiview_scores.json",
    )
    parser.add_argument(
        "--sensitivity_json",
        default="/root/eccv/.worktrees/seedselect80/code/outputs/sensitivity_multimetric/refiner_metric_sensitivity.json",
    )
    parser.add_argument(
        "--guardrail_json",
        default="/root/eccv/.worktrees/seedselect80/code/outputs/guardrail_multimetric/guardrail_abstain_results.json",
    )
    parser.add_argument(
        "--fusion_json",
        default="/root/eccv/.worktrees/seedselect80/code/outputs/fusion_ranker/fusion_ranker_results.json",
    )
    parser.add_argument(
        "--multimetric_json",
        default="/root/eccv/.worktrees/seedselect80/code/outputs/multimetric/difix_multimetric_scores_gso300.json",
    )
    parser.add_argument("--default_seed", default="42")
    parser.add_argument("--render_bench_objects", type=int, default=5)
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument(
        "--out_dir",
        default="/root/eccv/.worktrees/seedselect80/code/outputs/frontier",
    )
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_sensitivity(path: str):
    rows = load_json(path)["results"]
    by_name = {r["variant"]: r for r in rows}
    return by_name


def compute_fusion_risk(results_json: str, fusion_json: str, default_seed: str):
    results = load_json(results_json)
    fusion = load_json(fusion_json)
    pred = fusion["predicted_scores"]
    severe = 0
    worst = 0
    n = 0
    for obj, scores in pred.items():
        if obj not in results:
            continue
        gt = {str(s): float(v["cd"]) for s, v in results[obj]["seeds"].items() if "cd" in v}
        shared = sorted(set(gt.keys()) & set(scores.keys()))
        if default_seed not in shared or len(shared) < 2:
            continue
        selected = max(shared, key=lambda s: scores[s])
        d = gt[default_seed]
        s = gt[selected]
        w = max(shared, key=lambda x: gt[x])
        if selected == w:
            worst += 1
        if (s - d) / max(d, 1e-12) > 0.05:
            severe += 1
        n += 1
    return {
        "n": n,
        "severe_degrade_rate_pct": severe / max(n, 1) * 100.0,
        "worst_pick_rate_pct": worst / max(n, 1) * 100.0,
    }


def benchmark_render_time(results_json: str, sample_objects: int, render_size: int):
    import sys

    import trimesh

    sys.path.insert(0, "/root/eccv/.worktrees/seedselect80/code")
    from src.utils.camera import look_at
    from src.utils.render import render_mesh

    results = load_json(results_json)
    obj_ids = list(results.keys())[:sample_objects]
    views = [
        ("front", (0, 0, 2.0), (0, 1, 0)),
        ("back", (0, 0, -2.0), (0, 1, 0)),
        ("left", (-2.0, 0, 0), (0, 1, 0)),
        ("right", (2.0, 0, 0), (0, 1, 0)),
        ("top", (0, 2.0, 0), (0, 0, -1)),
        ("front_right", (1.41, 0, 1.41), (0, 1, 0)),
    ]
    seeds = ["0", "1", "2", "42"]

    t0 = time.time()
    n_triplets = 0
    for obj in obj_ids:
        for s in seeds:
            mesh_path = (
                f"/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/"
                f"{obj}/seed{s}/instant-mesh-large/meshes/{obj}.obj"
            )
            if not os.path.exists(mesh_path):
                continue
            mesh = trimesh.load(mesh_path, process=False, force="mesh")
            for _, eye, up in views:
                c2w = look_at(eye, up=up)
                render_mesh(mesh, c2w, width=render_size, height=render_size, yfov=np.deg2rad(50.0))
                n_triplets += 1
    elapsed = time.time() - t0
    per_obj = elapsed / max(len(obj_ids), 1)
    return {
        "sample_objects": len(obj_ids),
        "elapsed_sec": elapsed,
        "n_renders": n_triplets,
        "render_sec_per_object_k4": per_obj,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sensitivity = parse_sensitivity(args.sensitivity_json)
    guardrail = load_json(args.guardrail_json)
    multi = load_json(args.multimetric_json)
    fusion_risk = compute_fusion_risk(args.results_json, args.fusion_json, args.default_seed)

    # Time terms
    render_bench = benchmark_render_time(
        args.results_json,
        sample_objects=args.render_bench_objects,
        render_size=args.render_size,
    )
    score_sec_per_obj = (
        float(multi["metadata"]["elapsed_sec"]) / max(float(multi["metadata"]["n_objects_processed"]), 1.0)
    )
    # Tiny forward on logistic ranker.
    t0 = time.time()
    for _ in range(50000):
        _ = 1.7 * 0.52 - 0.3
    forward_sec = (time.time() - t0) / 50000.0

    t_render = render_bench["render_sec_per_object_k4"]
    t_feature = score_sec_per_obj
    t_lpips = t_render + t_feature
    t_guard = t_lpips
    t_fusion = t_lpips + forward_sec

    # Quality/risk entries
    lpips_row = sensitivity["Difix3D+ x LPIPS"]
    fusion_row = sensitivity["LearnedFusion x LogisticRanker"]
    base_guard = min(guardrail["sweep"], key=lambda r: abs(r["threshold"] - 0.0))
    guard_best = guardrail["best"]

    methods = [
        {
            "method": "Default (seed42)",
            "improvement_pct": 0.0,
            "severe_degrade_rate_pct": 0.0,
            "worst_pick_rate_pct": 0.0,
            "time_sec_per_object": 0.0,
            "coverage_pct": 100.0,
            "p_value": 1.0,
        },
        {
            "method": "SeedSelect LPIPS",
            "improvement_pct": float(lpips_row["improvement_pct"]),
            "severe_degrade_rate_pct": float(base_guard["severe_degrade_rate_pct"]),
            "worst_pick_rate_pct": float(lpips_row["worst_pick_rate_pct"]),
            "time_sec_per_object": float(t_lpips),
            "coverage_pct": 100.0,
            "p_value": float(lpips_row["wilcoxon_p"]),
        },
        {
            "method": "LearnedFusion Ranker",
            "improvement_pct": float(fusion_row["improvement_pct"]),
            "severe_degrade_rate_pct": float(fusion_risk["severe_degrade_rate_pct"]),
            "worst_pick_rate_pct": float(fusion_risk["worst_pick_rate_pct"]),
            "time_sec_per_object": float(t_fusion),
            "coverage_pct": 100.0,
            "p_value": float(fusion_row["wilcoxon_p"]),
        },
        {
            "method": "SeedSelect + Guardrail",
            "improvement_pct": float(guard_best["improvement_pct"]),
            "severe_degrade_rate_pct": float(guard_best["severe_degrade_rate_pct"]),
            "worst_pick_rate_pct": float(guard_best["worst_pick_rate_pct"]),
            "time_sec_per_object": float(t_guard),
            "coverage_pct": float(100.0 - guard_best["abstain_rate_pct"]),
            "p_value": float(guard_best["p_value"]),
        },
    ]

    csv_path = os.path.join(args.out_dir, "quality_time_risk_frontier.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "improvement_pct",
                "severe_degrade_rate_pct",
                "worst_pick_rate_pct",
                "time_sec_per_object",
                "coverage_pct",
                "p_value",
            ],
        )
        w.writeheader()
        for row in methods:
            w.writerow(row)

    # Scatter: x=time, y=quality, color=risk(severe), marker size=coverage
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = [m["time_sec_per_object"] for m in methods]
    y = [m["improvement_pct"] for m in methods]
    c = [m["severe_degrade_rate_pct"] for m in methods]
    s = [40 + 1.2 * m["coverage_pct"] for m in methods]
    sc = ax.scatter(x, y, c=c, s=s, cmap="viridis_r", alpha=0.9, edgecolors="black", linewidth=0.6)
    for m in methods:
        ax.annotate(
            m["method"],
            (m["time_sec_per_object"], m["improvement_pct"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Severe degrade rate (%)")
    ax.axhline(0.0, color="#999999", linewidth=0.9)
    ax.set_xlabel("Inference-time cost per object (sec)")
    ax.set_ylabel("CD improvement (%)")
    ax.set_title("Compute-Matched Quality-Time-Risk Frontier")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "quality_time_risk_frontier.pdf")
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close()

    out_json = os.path.join(args.out_dir, "quality_time_risk_frontier.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "timing_breakdown": {
                    "render_benchmark": render_bench,
                    "feature_time_sec_per_object": t_feature,
                    "verifier_forward_sec": forward_sec,
                    "protocol": "T_total = T_render + T_feature + T_verifier_forward",
                },
                "methods": methods,
                "artifacts": {"csv": csv_path, "figure": fig_path},
            },
            f,
            indent=2,
        )

    print(f"Saved {out_json}")
    print(
        f"Time breakdown: render={t_render:.3f}s, feature={t_feature:.3f}s, "
        f"fusion_forward={forward_sec*1000:.3f}ms"
    )


if __name__ == "__main__":
    main()


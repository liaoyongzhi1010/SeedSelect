#!/usr/bin/env python3
"""Per-category analysis for OmniObject3D SeedSelect results.

Extracts category from object IDs (e.g., 'glasses_004' -> 'glasses'),
computes per-category improvement statistics, and generates a horizontal
bar chart figure.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "multiseed")
FIGURE_DIR = os.path.join(PROJECT_DIR, "..", "eccv2026", "paper", "img")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def category_from_id(obj_id):
    """Extract category from OmniObject3D object ID."""
    if '_' in obj_id:
        parts = obj_id.rsplit('_', 1)
        # Check if last part is numeric (e.g., 'glasses_004')
        if parts[-1].isdigit():
            return parts[0]
    return obj_id


def main():
    omni_dir = os.path.join(OUTPUT_DIR, "omni_full")
    results = load_json(os.path.join(omni_dir, "results.json"))
    difix = load_json(os.path.join(omni_dir, "difix_multiview_scores.json"))

    # Get proxy scores
    proxy_scores = difix["scores"]["difix_mv_mean"]

    # Per-category analysis
    categories = defaultdict(list)

    for obj, data in results.items():
        if "seeds" not in data:
            continue

        cat = category_from_id(obj)
        seeds = data["seeds"]
        default_seed = "42"
        if default_seed not in seeds:
            default_seed = sorted(seeds.keys())[-1]

        default_cd = seeds[default_seed]["cd"]

        # Oracle CD (best possible)
        cd_values = {s: v["cd"] for s, v in seeds.items()}
        oracle_cd = min(cd_values.values())

        # SeedSelect CD (select by proxy)
        if obj in proxy_scores:
            px = proxy_scores[obj]
            # Higher (less negative) = better
            selected_seed = max(px, key=px.get)
            selected_cd = cd_values.get(selected_seed, default_cd)
        else:
            selected_cd = default_cd

        improvement_pct = (default_cd - selected_cd) / default_cd * 100 if default_cd > 0 else 0
        oracle_improvement = (default_cd - oracle_cd) / default_cd * 100 if default_cd > 0 else 0

        categories[cat].append({
            "obj": obj,
            "default_cd": default_cd,
            "selected_cd": selected_cd,
            "oracle_cd": oracle_cd,
            "improvement_pct": improvement_pct,
            "oracle_improvement_pct": oracle_improvement,
        })

    # Compute per-category statistics
    cat_stats = {}
    for cat, objs in categories.items():
        improvements = [o["improvement_pct"] for o in objs]
        oracle_improvements = [o["oracle_improvement_pct"] for o in objs]
        cat_stats[cat] = {
            "n_objects": len(objs),
            "mean_improvement_pct": float(np.mean(improvements)),
            "std_improvement_pct": float(np.std(improvements)),
            "mean_oracle_improvement_pct": float(np.mean(oracle_improvements)),
            "median_improvement_pct": float(np.median(improvements)),
            "min_improvement_pct": float(np.min(improvements)),
            "max_improvement_pct": float(np.max(improvements)),
        }

    # Sort categories by mean improvement
    sorted_cats = sorted(cat_stats.items(), key=lambda x: x[1]["mean_improvement_pct"])

    print("Per-Category Analysis (OmniObject3D)")
    print("=" * 70)
    print(f"{'Category':<20} {'N':>3} {'Improv%':>8} {'Oracle%':>8} {'Median%':>8}")
    print("-" * 70)
    for cat, stats in sorted_cats:
        print(f"{cat:<20} {stats['n_objects']:>3} "
              f"{stats['mean_improvement_pct']:>+7.1f} "
              f"{stats['mean_oracle_improvement_pct']:>+7.1f} "
              f"{stats['median_improvement_pct']:>+7.1f}")

    # Generate figure
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Professional color palette
    cats_sorted = [c[0] for c in sorted_cats]
    improvements = [c[1]["mean_improvement_pct"] for c in sorted_cats]
    oracle_imps = [c[1]["mean_oracle_improvement_pct"] for c in sorted_cats]
    n_objs = [c[1]["n_objects"] for c in sorted_cats]

    y_pos = np.arange(len(cats_sorted))

    # Oracle bars (lighter)
    bars_oracle = ax.barh(y_pos, oracle_imps, height=0.4, color='#B8D4E3',
                          label='Oracle', alpha=0.8, edgecolor='#7BA7C4', linewidth=0.5)
    # SeedSelect bars (darker, overlaid)
    colors = ['#2D6A4F' if v >= 0 else '#C44E52' for v in improvements]
    bars_ours = ax.barh(y_pos, improvements, height=0.4, color=colors,
                        label='SeedSelect', alpha=0.9, edgecolor='none', linewidth=0.5)

    # Labels
    ax.set_yticks(y_pos)
    labels = [f"{c} ({n})" for c, n in zip(cats_sorted, n_objs)]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('CD Improvement (%)', fontsize=9)
    ax.axvline(x=0, color='#333333', linewidth=0.5, linestyle='-')

    # Legend
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=7)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "category_breakdown.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    # Save results
    output = {
        "per_category": cat_stats,
        "n_categories": len(cat_stats),
        "n_total_objects": sum(s["n_objects"] for s in cat_stats.values()),
    }
    out_path = os.path.join(OUTPUT_DIR, "category_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Failure case analysis for SeedSelect.

Characterizes the ~18% worst-pick cases: distribution of CD degradation,
correlation with CD range, and generates a failure analysis figure.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "multiseed")
FIGURE_DIR = os.path.join(PROJECT_DIR, "..", "eccv2026", "paper", "img")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    analysis = load_json(os.path.join(OUTPUT_DIR, "gso_full", "comprehensive_analysis.json"))
    results = load_json(os.path.join(OUTPUT_DIR, "gso_full", "results.json"))

    per_object = analysis["per_object"]

    # Classify all objects
    improvements = []
    failures = []
    successes = []
    oracle_matches = []

    for obj_id, data in per_object.items():
        default_cd = data["default_cd"]
        selected_cd = data["selected_cd"]
        oracle_cd = data["oracle_cd"]
        cd_range = data["cd_range"]
        picked_worst = data["picked_worst"]

        improvement_pct = (default_cd - selected_cd) / default_cd * 100 if default_cd > 0 else 0
        rel_range = cd_range / default_cd * 100 if default_cd > 0 else 0

        entry = {
            "obj_id": obj_id,
            "default_cd": default_cd,
            "selected_cd": selected_cd,
            "oracle_cd": oracle_cd,
            "improvement_pct": improvement_pct,
            "cd_range": cd_range,
            "relative_range": rel_range,
            "picked_worst": picked_worst,
            "picked_oracle": data.get("picked_oracle", False),
        }
        improvements.append(entry)

        if improvement_pct < 0:
            failures.append(entry)
        else:
            successes.append(entry)

        if data.get("picked_oracle", False):
            oracle_matches.append(entry)

    # Sort failures by damage (most harmful first)
    failures_sorted = sorted(failures, key=lambda x: x["improvement_pct"])

    print("Failure Case Analysis (GSO-300, K=4)")
    print("=" * 70)
    print(f"Total objects: {len(improvements)}")
    print(f"Improvements (>0%): {len(successes)} ({100*len(successes)/len(improvements):.1f}%)")
    print(f"Degradations (<0%): {len(failures)} ({100*len(failures)/len(improvements):.1f}%)")
    print(f"Oracle matches: {len(oracle_matches)} ({100*len(oracle_matches)/len(improvements):.1f}%)")

    # Analyze failure severity
    if failures:
        fail_damages = [-f["improvement_pct"] for f in failures]
        print(f"\nFailure severity distribution:")
        print(f"  Mean degradation: {np.mean(fail_damages):.2f}%")
        print(f"  Median degradation: {np.median(fail_damages):.2f}%")
        print(f"  Max degradation: {np.max(fail_damages):.2f}%")
        print(f"  <1% degradation: {sum(1 for d in fail_damages if d < 1)}/{len(fail_damages)} ({100*sum(1 for d in fail_damages if d < 1)/len(fail_damages):.0f}%)")
        print(f"  <5% degradation: {sum(1 for d in fail_damages if d < 5)}/{len(fail_damages)} ({100*sum(1 for d in fail_damages if d < 5)/len(fail_damages):.0f}%)")

        print(f"\nTop 10 most harmful failures:")
        for i, f in enumerate(failures_sorted[:10]):
            print(f"  {i+1}. {f['obj_id'][:40]:<40} {f['improvement_pct']:>+6.2f}% (range={f['relative_range']:.1f}%)")

    # Correlation: CD range vs failure
    all_ranges = [e["relative_range"] for e in improvements]
    all_outcomes = [1 if e["improvement_pct"] >= 0 else 0 for e in improvements]

    # Success rate by range quartile
    ranges_arr = np.array(all_ranges)
    outcomes_arr = np.array(all_outcomes)
    quartiles = np.percentile(ranges_arr, [25, 50, 75])

    print(f"\nSuccess rate by CD range quartile:")
    bins = [0, quartiles[0], quartiles[1], quartiles[2], np.inf]
    labels_q = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    for i in range(4):
        mask = (ranges_arr >= bins[i]) & (ranges_arr < bins[i+1])
        if mask.sum() > 0:
            rate = outcomes_arr[mask].mean()
            print(f"  {labels_q[i]:>12}: {rate:.1%} success ({mask.sum()} objects)")

    # Generate figure
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Panel a: Distribution of improvement/degradation
    ax = axes[0]
    all_imps = [e["improvement_pct"] for e in improvements]
    colors_hist = ['#2D6A4F' if v >= 0 else '#C44E52' for v in np.linspace(-15, 40, 50)]

    bins_hist = np.linspace(min(all_imps) - 1, max(all_imps) + 1, 40)
    vals, bin_edges, patches = ax.hist(all_imps, bins=bins_hist, edgecolor='white', linewidth=0.3)
    for patch, left_edge in zip(patches, bin_edges):
        if left_edge < 0:
            patch.set_facecolor('#C44E52')
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor('#2D6A4F')
            patch.set_alpha(0.8)

    ax.axvline(x=0, color='#333333', linewidth=0.8, linestyle='--')
    ax.set_xlabel('CD Improvement (%)', fontsize=7)
    ax.set_ylabel('Number of Objects', fontsize=7)
    ax.set_title('(a) Improvement Distribution', fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate failure region
    n_fail = len(failures)
    n_total = len(improvements)
    ax.annotate(f'{n_fail}/{n_total}\ndegrade',
                xy=(-2, max(vals) * 0.5), fontsize=6, color='#C44E52',
                ha='center')

    # Panel b: Failure severity (sorted)
    ax = axes[1]
    if failures:
        fail_imps = sorted([f["improvement_pct"] for f in failures])
        x = np.arange(len(fail_imps))
        ax.bar(x, [-v for v in fail_imps], color='#C44E52', alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.set_xlabel('Failure Cases (sorted)', fontsize=7)
        ax.set_ylabel('CD Degradation (%)', fontsize=7)
        ax.set_title('(b) Failure Severity', fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Annotate: most are mild
        mild = sum(1 for v in fail_imps if -v < 5)
        ax.annotate(f'{mild}/{len(fail_imps)} < 5%',
                    xy=(len(fail_imps) * 0.3, max([-v for v in fail_imps]) * 0.7),
                    fontsize=6, color='#666666')

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "failures.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    # Save results
    output = {
        "n_total": len(improvements),
        "n_improvements": len(successes),
        "n_failures": len(failures),
        "n_oracle_matches": len(oracle_matches),
        "failure_rate": len(failures) / len(improvements),
        "mean_failure_degradation_pct": float(np.mean([-f["improvement_pct"] for f in failures])) if failures else 0,
        "median_failure_degradation_pct": float(np.median([-f["improvement_pct"] for f in failures])) if failures else 0,
        "max_failure_degradation_pct": float(np.max([-f["improvement_pct"] for f in failures])) if failures else 0,
        "pct_failures_under_5pct": sum(1 for f in failures if -f["improvement_pct"] < 5) / len(failures) * 100 if failures else 0,
        "pct_failures_under_1pct": sum(1 for f in failures if -f["improvement_pct"] < 1) / len(failures) * 100 if failures else 0,
        "top_10_failures": [
            {"obj_id": f["obj_id"], "degradation_pct": -f["improvement_pct"], "cd_range": f["cd_range"]}
            for f in failures_sorted[:10]
        ],
    }
    out_path = os.path.join(OUTPUT_DIR, "failure_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

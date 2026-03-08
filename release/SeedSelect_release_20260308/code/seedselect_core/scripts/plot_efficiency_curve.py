#!/usr/bin/env python3
"""Quality-vs-Compute efficiency curve for SeedSelect.

Plots CD improvement as a function of compute (number of candidates K),
showing the scaling behavior of SeedSelect vs optimization-based methods.
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
    ablation = load_json(os.path.join(OUTPUT_DIR, "ablation_results.json"))

    # SeedSelect data (GSO-50 subset with K=2,4,8 + K=16 from EXPERIMENT_STATUS)
    # Per-object generation time: InstantMesh ~60s per seed + 42s scoring
    # K=1: default, no improvement
    k_values = [1, 2, 4, 8, 16]
    improvements_ours = [0.0]  # K=1 = default
    improvements_oracle = [0.0]
    compute_times = []  # seconds per object

    gen_time_per_seed = 60  # seconds (InstantMesh)
    score_time = 42  # seconds (Difix3D+ 6 views)

    for k in k_values:
        compute_times.append(k * gen_time_per_seed + score_time)

    for k_label in ["K=2", "K=4", "K=8"]:
        if k_label in ablation["ablation_k"]:
            data = ablation["ablation_k"][k_label]
            improvements_ours.append(data["scoring_improvement_pct"])
            improvements_oracle.append(data["oracle_improvement_pct"])

    # K=16 from EXPERIMENT_STATUS
    improvements_ours.append(5.4)
    improvements_oracle.append(14.3)

    # Also add GSO-300 data points
    k_300 = [4, 8]
    imp_300_ours = [1.5, 2.2]
    imp_300_oracle = [6.6, 8.3]
    compute_300 = [k * gen_time_per_seed + score_time for k in k_300]

    # Optimization baselines (approximate times from literature)
    opt_baselines = {
        'DreamFusion': {'time': 30 * 60, 'improvement': None, 'marker': 'v'},
        'SJC': {'time': 20 * 60, 'improvement': None, 'marker': '^'},
    }

    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Plot SeedSelect curves (GSO-50)
    ax.plot(k_values, improvements_ours, 'o-', color='#2D6A4F', linewidth=2,
            markersize=6, label='SeedSelect (GSO-50)', zorder=5)
    ax.plot(k_values, improvements_oracle, 's--', color='#B8D4E3', linewidth=1.5,
            markersize=5, label='Oracle (GSO-50)', zorder=4)

    # Plot GSO-300 points
    ax.plot(k_300, imp_300_ours, 'D-', color='#1B4332', linewidth=2,
            markersize=6, label='SeedSelect (GSO-300)', zorder=5)
    ax.plot(k_300, imp_300_oracle, 's--', color='#7BA7C4', linewidth=1.5,
            markersize=5, label='Oracle (GSO-300)', zorder=4)

    # Annotate K values
    for k, imp in zip(k_values, improvements_ours):
        if k > 1:
            ax.annotate(f'K={k}', (k, imp), textcoords='offset points',
                        xytext=(8, -3), fontsize=6, color='#2D6A4F')

    # Add theoretical curve (order statistics prediction)
    k_theoretical = np.linspace(1, 20, 100)
    # sigma * Phi^{-1}(K/(K+1)) approximation
    from scipy.stats import norm
    sigma = improvements_oracle[-1] / norm.ppf(16/17)  # calibrate to K=16 oracle
    theoretical = sigma * norm.ppf(k_theoretical / (k_theoretical + 1))
    theoretical = np.maximum(theoretical, 0)
    ax.plot(k_theoretical, theoretical, ':', color='#999999', linewidth=1,
            label='Order statistics prediction', zorder=2)

    # Formatting
    ax.set_xlabel('Number of Candidates $K$', fontsize=9)
    ax.set_ylabel('CD Improvement (%)', fontsize=9)
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.tick_params(labelsize=7)
    ax.set_xlim(0.8, 20)
    ax.set_ylim(-0.5, 16)

    # Add "random" baseline
    ax.axhline(y=0, color='#CCCCCC', linewidth=0.8, linestyle='-', zorder=1)

    ax.legend(fontsize=6, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "efficiency_curve.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()

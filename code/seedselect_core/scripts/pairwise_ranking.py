#!/usr/bin/env python3
"""Pairwise ranking accuracy analysis for SeedSelect.

For each object, enumerates all C(K,2) pairs of seeds and checks whether
the proxy ranking agrees with the GT (CD) ranking. Reports pairwise
accuracy and Kendall's tau with bootstrap confidence intervals.
"""

import json
import os
import sys
from itertools import combinations
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "multiseed")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_pairwise_stats(gt_scores, proxy_scores, objects, lower_is_better_gt=True, higher_is_better_proxy=True):
    """Compute pairwise ranking accuracy and Kendall's tau.

    Args:
        gt_scores: dict[obj][seed] -> float (ground truth, e.g., CD)
        proxy_scores: dict[obj][seed] -> float (proxy score)
        objects: list of object IDs to evaluate
        lower_is_better_gt: if True, lower GT value = better quality
        higher_is_better_proxy: if True, higher proxy value = better quality
    """
    per_object_acc = []
    per_object_tau = []
    total_concordant = 0
    total_discordant = 0
    total_tied = 0
    total_pairs = 0

    for obj in objects:
        if obj not in gt_scores or obj not in proxy_scores:
            continue

        gt = gt_scores[obj]
        px = proxy_scores[obj]
        seeds = sorted(set(gt.keys()) & set(px.keys()))

        if len(seeds) < 2:
            continue

        concordant = 0
        discordant = 0
        tied = 0
        pairs = list(combinations(seeds, 2))

        for si, sj in pairs:
            # GT preference: which seed is better?
            gt_diff = gt[si] - gt[sj]
            if lower_is_better_gt:
                gt_diff = -gt_diff  # flip so positive = si is better

            # Proxy preference
            px_diff = px[si] - px[sj]
            if not higher_is_better_proxy:
                px_diff = -px_diff  # flip so positive = si is better

            if gt_diff == 0 or px_diff == 0:
                tied += 1
            elif (gt_diff > 0) == (px_diff > 0):
                concordant += 1
            else:
                discordant += 1

        n_effective = concordant + discordant
        if n_effective > 0:
            acc = concordant / n_effective
            tau = (concordant - discordant) / n_effective
        else:
            acc = 0.5
            tau = 0.0

        per_object_acc.append(acc)
        per_object_tau.append(tau)
        total_concordant += concordant
        total_discordant += discordant
        total_tied += tied
        total_pairs += len(pairs)

    per_object_acc = np.array(per_object_acc)
    per_object_tau = np.array(per_object_tau)

    n_effective_total = total_concordant + total_discordant
    if n_effective_total > 0:
        global_acc = total_concordant / n_effective_total
        global_tau = (total_concordant - total_discordant) / n_effective_total
    else:
        global_acc = 0.5
        global_tau = 0.0

    return {
        "mean_pairwise_acc": float(np.mean(per_object_acc)),
        "median_pairwise_acc": float(np.median(per_object_acc)),
        "std_pairwise_acc": float(np.std(per_object_acc)),
        "mean_kendall_tau": float(np.mean(per_object_tau)),
        "median_kendall_tau": float(np.median(per_object_tau)),
        "global_pairwise_acc": global_acc,
        "global_kendall_tau": global_tau,
        "total_concordant": total_concordant,
        "total_discordant": total_discordant,
        "total_tied": total_tied,
        "total_pairs": total_pairs,
        "n_objects": len(per_object_acc),
        "per_object_acc": per_object_acc,
        "per_object_tau": per_object_tau,
    }


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True))
                           for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lo, hi


def analyze_dataset(results_path, difix_scores_path, clip_scores_path=None, dataset_name=""):
    """Analyze pairwise ranking for one dataset."""
    results = load_json(results_path)
    difix_data = load_json(difix_scores_path)

    # Build GT CD dict: obj -> seed -> cd
    gt_cd = {}
    for obj, data in results.items():
        if "seeds" not in data:
            continue
        gt_cd[obj] = {s: v["cd"] for s, v in data["seeds"].items()}

    objects = list(gt_cd.keys())
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({len(objects)} objects)")
    print(f"{'='*60}")

    all_results = {}

    # 1. Difix MV Mean (SeedSelect proxy)
    # Handle two formats: nested (scores.difix_mv_mean.obj.seed) or flat (scores.obj.seed)
    difix_scores_section = None
    if "scores" in difix_data:
        if "difix_mv_mean" in difix_data["scores"]:
            difix_scores_section = difix_data["scores"]["difix_mv_mean"]
        elif isinstance(next(iter(difix_data["scores"].values()), None), dict):
            # Flat format: scores.obj.seed -> value
            difix_scores_section = difix_data["scores"]

    if difix_scores_section is not None:
        proxy_scores = {}
        for obj, seeds_dict in difix_scores_section.items():
            if isinstance(seeds_dict, dict):
                proxy_scores[obj] = {s: float(v) for s, v in seeds_dict.items()}

        stats = compute_pairwise_stats(
            gt_cd, proxy_scores, objects,
            lower_is_better_gt=True,
            higher_is_better_proxy=True  # Less negative = less refinement = better quality
        )
        ci_lo, ci_hi = bootstrap_ci(stats["per_object_acc"])
        tau_ci_lo, tau_ci_hi = bootstrap_ci(stats["per_object_tau"])

        print(f"\nSeedSelect (Difix MV Mean):")
        print(f"  Mean pairwise accuracy: {stats['mean_pairwise_acc']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  Mean Kendall tau:       {stats['mean_kendall_tau']:.3f} [{tau_ci_lo:.3f}, {tau_ci_hi:.3f}]")
        print(f"  Global accuracy:        {stats['global_pairwise_acc']:.3f}")
        print(f"  Concordant/Discordant:  {stats['total_concordant']}/{stats['total_discordant']}")

        all_results["seedselect_difix_mv_mean"] = {
            "mean_pairwise_acc": stats["mean_pairwise_acc"],
            "ci_95": [ci_lo, ci_hi],
            "mean_kendall_tau": stats["mean_kendall_tau"],
            "tau_ci_95": [tau_ci_lo, tau_ci_hi],
            "global_pairwise_acc": stats["global_pairwise_acc"],
            "global_kendall_tau": stats["global_kendall_tau"],
            "concordant": stats["total_concordant"],
            "discordant": stats["total_discordant"],
            "tied": stats["total_tied"],
            "n_objects": stats["n_objects"],
        }

    # 2. Difix Front only
    difix_front_section = None
    if "scores" in difix_data and "difix_front" in difix_data["scores"]:
        difix_front_section = difix_data["scores"]["difix_front"]

    if difix_front_section is not None:
        proxy_scores = {}
        for obj, seeds_dict in difix_front_section.items():
            if isinstance(seeds_dict, dict):
                proxy_scores[obj] = {s: float(v) for s, v in seeds_dict.items()}

        stats = compute_pairwise_stats(
            gt_cd, proxy_scores, objects,
            lower_is_better_gt=True,
            higher_is_better_proxy=True  # Less negative = better
        )
        ci_lo, ci_hi = bootstrap_ci(stats["per_object_acc"])

        print(f"\nDifix Front Only:")
        print(f"  Mean pairwise accuracy: {stats['mean_pairwise_acc']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  Mean Kendall tau:       {stats['mean_kendall_tau']:.3f}")

        all_results["difix_front"] = {
            "mean_pairwise_acc": stats["mean_pairwise_acc"],
            "ci_95": [ci_lo, ci_hi],
            "mean_kendall_tau": stats["mean_kendall_tau"],
            "global_pairwise_acc": stats["global_pairwise_acc"],
            "n_objects": stats["n_objects"],
        }

    # 3. PSNR (from results.json)
    psnr_scores = {}
    for obj, data in results.items():
        if "seeds" not in data:
            continue
        psnr_dict = {}
        for s, v in data["seeds"].items():
            psnr = v.get("psnr", -1)
            if psnr > 0:
                psnr_dict[s] = psnr
        if len(psnr_dict) >= 2:
            psnr_scores[obj] = psnr_dict

    if psnr_scores:
        stats = compute_pairwise_stats(
            gt_cd, psnr_scores, objects,
            lower_is_better_gt=True,
            higher_is_better_proxy=True  # Higher PSNR = better
        )
        ci_lo, ci_hi = bootstrap_ci(stats["per_object_acc"])

        print(f"\nPSNR:")
        print(f"  Mean pairwise accuracy: {stats['mean_pairwise_acc']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  Mean Kendall tau:       {stats['mean_kendall_tau']:.3f}")

        all_results["psnr"] = {
            "mean_pairwise_acc": stats["mean_pairwise_acc"],
            "ci_95": [ci_lo, ci_hi],
            "mean_kendall_tau": stats["mean_kendall_tau"],
            "global_pairwise_acc": stats["global_pairwise_acc"],
            "n_objects": stats["n_objects"],
        }

    # 4. CLIP and DINOv2 (if available)
    if clip_scores_path and os.path.exists(clip_scores_path):
        clip_data = load_json(clip_scores_path)

        for score_key, label in [("per_object_clip", "CLIP"), ("per_object_dino", "DINOv2")]:
            if score_key not in clip_data:
                continue
            proxy_scores = {}
            for obj, obj_data in clip_data[score_key].items():
                if "scores" in obj_data:
                    proxy_scores[obj] = {s: float(v) for s, v in obj_data["scores"].items()}

            if proxy_scores:
                stats = compute_pairwise_stats(
                    gt_cd, proxy_scores, objects,
                    lower_is_better_gt=True,
                    higher_is_better_proxy=True  # Higher similarity = better
                )
                ci_lo, ci_hi = bootstrap_ci(stats["per_object_acc"])

                print(f"\n{label}:")
                print(f"  Mean pairwise accuracy: {stats['mean_pairwise_acc']:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
                print(f"  Mean Kendall tau:       {stats['mean_kendall_tau']:.3f}")

                all_results[label.lower()] = {
                    "mean_pairwise_acc": stats["mean_pairwise_acc"],
                    "ci_95": [ci_lo, ci_hi],
                    "mean_kendall_tau": stats["mean_kendall_tau"],
                    "global_pairwise_acc": stats["global_pairwise_acc"],
                    "n_objects": stats["n_objects"],
                }

    # Random baseline (expected 50%)
    all_results["random"] = {
        "mean_pairwise_acc": 0.5,
        "ci_95": [0.5, 0.5],
        "mean_kendall_tau": 0.0,
        "global_pairwise_acc": 0.5,
        "n_objects": len(objects),
    }
    print(f"\nRandom baseline: 0.500")

    return all_results


def main():
    output = {}

    # GSO-300 K=4
    gso_dir = os.path.join(OUTPUT_DIR, "gso_full")
    if os.path.exists(gso_dir):
        output["gso_300_k4"] = analyze_dataset(
            os.path.join(gso_dir, "results.json"),
            os.path.join(gso_dir, "difix_multiview_scores.json"),
            os.path.join(gso_dir, "clip_scoring_results.json"),
            "GSO-300 (K=4)"
        )

    # GSO-300 K=8
    gso_k8_dir = os.path.join(OUTPUT_DIR, "gso_full_k8")
    if os.path.exists(gso_k8_dir):
        difix_k8 = os.path.join(gso_k8_dir, "difix_multiview_scores.json")
        if os.path.exists(difix_k8):
            output["gso_300_k8"] = analyze_dataset(
                os.path.join(gso_k8_dir, "results.json"),
                difix_k8,
                None,
                "GSO-300 (K=8)"
            )

    # OmniObject3D
    omni_dir = os.path.join(OUTPUT_DIR, "omni_full")
    if os.path.exists(omni_dir):
        output["omni_k4"] = analyze_dataset(
            os.path.join(omni_dir, "results.json"),
            os.path.join(omni_dir, "difix_multiview_scores.json"),
            None,
            "OmniObject3D (K=4+)"
        )

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'GSO-300 K=4':>12} {'GSO-300 K=8':>12} {'Omni':>12}")
    print("-" * 65)

    methods = ["random", "psnr", "clip", "dinov2", "difix_front", "seedselect_difix_mv_mean"]
    labels = ["Random", "PSNR", "CLIP", "DINOv2", "Difix (front)", "SeedSelect (Ours)"]

    for method, label in zip(methods, labels):
        vals = []
        for ds in ["gso_300_k4", "gso_300_k8", "omni_k4"]:
            if ds in output and method in output[ds]:
                vals.append(f"{output[ds][method]['mean_pairwise_acc']:.1%}")
            else:
                vals.append("---")
        print(f"{label:<25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Save results (convert numpy arrays for JSON serialization)
    out_path = os.path.join(OUTPUT_DIR, "pairwise_ranking_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

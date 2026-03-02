#!/usr/bin/env python3
"""Refiner x metric sensitivity analysis for SeedSelect.

This script evaluates multiple scoring variants against the same GT seed pool and
reports selection quality metrics used in the paper:
  - CD improvement (%)
  - Oracle-gap closed (%)
  - Pairwise ranking accuracy (%)
  - Wilcoxon p-value
  - Win/tie/loss and worst-pick rate

It is designed to be extensible: add new variants via --extra_scores_json where
each entry is an object->seed->score dictionary.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_RESULTS = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/results.json"
DEFAULT_DIFIX = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/difix_multiview_scores.json"
DEFAULT_CLIP = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/clip_scoring_results.json"
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "sensitivity")


@dataclass
class Variant:
    name: str
    refiner: str
    metric: str
    scores: Dict[str, Dict[str, float]]
    higher_is_better: bool = True


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def nested_get(data: dict, key_path: str):
    cur = data
    for key in key_path.split("."):
        if key not in cur:
            raise KeyError(f"Missing key '{key}' while reading '{key_path}'")
        cur = cur[key]
    return cur


def normalize_score_dict(raw_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    norm = {}
    for obj, seeds in raw_scores.items():
        if not isinstance(seeds, dict):
            continue
        obj_scores = {}
        for seed, value in seeds.items():
            try:
                obj_scores[str(seed)] = float(value)
            except (TypeError, ValueError):
                continue
        if len(obj_scores) >= 2:
            norm[obj] = obj_scores
    return norm


def extract_gt_cd(results_json: str) -> Dict[str, Dict[str, float]]:
    data = load_json(results_json)
    gt = {}
    for obj, payload in data.items():
        seeds = payload.get("seeds", {})
        obj_gt = {}
        for seed, seed_payload in seeds.items():
            if "cd" in seed_payload:
                obj_gt[str(seed)] = float(seed_payload["cd"])
        if len(obj_gt) >= 2:
            gt[obj] = obj_gt
    return gt


def pairwise_accuracy(
    gt_obj: Dict[str, float],
    score_obj: Dict[str, float],
    higher_is_better: bool,
) -> Tuple[float, int, int]:
    seeds = sorted(set(gt_obj.keys()) & set(score_obj.keys()))
    concordant = 0
    discordant = 0
    for si, sj in combinations(seeds, 2):
        gt_diff = gt_obj[si] - gt_obj[sj]
        # Lower CD is better, so gt_pref > 0 means si is better.
        gt_pref = -gt_diff

        sc_diff = score_obj[si] - score_obj[sj]
        sc_pref = sc_diff if higher_is_better else -sc_diff

        if gt_pref == 0 or sc_pref == 0:
            continue
        if (gt_pref > 0) == (sc_pref > 0):
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.5, 0, 0
    return concordant / total, concordant, discordant


def pick_seed(scores: Dict[str, float], higher_is_better: bool) -> str:
    key_fn = max if higher_is_better else min
    return key_fn(scores, key=scores.get)


def evaluate_variant(
    variant: Variant,
    gt_cd: Dict[str, Dict[str, float]],
    default_seed: str,
) -> Dict[str, float]:
    per_obj_pairwise = []
    selected_cds = []
    default_cds = []
    oracle_cds = []

    wins = 0
    ties = 0
    losses = 0
    oracle_match = 0
    worst_pick = 0
    total_concordant = 0
    total_discordant = 0
    used_objects = 0

    for obj, gt_obj in gt_cd.items():
        if obj not in variant.scores:
            continue
        sc_obj = variant.scores[obj]
        shared = sorted(set(gt_obj.keys()) & set(sc_obj.keys()))
        if len(shared) < 2:
            continue

        seed_default = default_seed if default_seed in shared else shared[0]
        selected = pick_seed({s: sc_obj[s] for s in shared}, variant.higher_is_better)
        oracle = min(shared, key=lambda s: gt_obj[s])
        worst = max(shared, key=lambda s: gt_obj[s])

        cd_d = gt_obj[seed_default]
        cd_s = gt_obj[selected]
        cd_o = gt_obj[oracle]

        default_cds.append(cd_d)
        selected_cds.append(cd_s)
        oracle_cds.append(cd_o)

        if cd_s < cd_d:
            wins += 1
        elif cd_s > cd_d:
            losses += 1
        else:
            ties += 1

        if selected == oracle:
            oracle_match += 1
        if selected == worst:
            worst_pick += 1

        acc, c, d = pairwise_accuracy(
            {s: gt_obj[s] for s in shared},
            {s: sc_obj[s] for s in shared},
            variant.higher_is_better,
        )
        per_obj_pairwise.append(acc)
        total_concordant += c
        total_discordant += d
        used_objects += 1

    if used_objects == 0:
        raise RuntimeError(f"No overlapping objects for variant: {variant.name}")

    default_cds = np.asarray(default_cds)
    selected_cds = np.asarray(selected_cds)
    oracle_cds = np.asarray(oracle_cds)
    per_obj_pairwise = np.asarray(per_obj_pairwise)

    default_mean = float(np.mean(default_cds))
    selected_mean = float(np.mean(selected_cds))
    oracle_mean = float(np.mean(oracle_cds))
    improvement = (default_mean - selected_mean) / max(default_mean, 1e-12) * 100.0
    oracle_gain = default_mean - oracle_mean
    gap_closed = (
        (default_mean - selected_mean) / max(oracle_gain, 1e-12) * 100.0
        if oracle_gain > 0
        else 0.0
    )

    # One-sided Wilcoxon: selected CD < default CD.
    try:
        w = wilcoxon(selected_cds, default_cds, alternative="less")
        p_value = float(w.pvalue)
    except ValueError:
        p_value = 1.0

    total_pairs = total_concordant + total_discordant
    global_pairwise = total_concordant / total_pairs if total_pairs > 0 else 0.5

    return {
        "variant": variant.name,
        "refiner": variant.refiner,
        "metric": variant.metric,
        "n_objects": int(used_objects),
        "default_cd": default_mean,
        "selected_cd": selected_mean,
        "oracle_cd": oracle_mean,
        "improvement_pct": float(improvement),
        "gap_closed_pct": float(gap_closed),
        "pairwise_acc_pct": float(np.mean(per_obj_pairwise) * 100.0),
        "pairwise_acc_global_pct": float(global_pairwise * 100.0),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "win_rate_pct": float(wins / used_objects * 100.0),
        "oracle_match_rate_pct": float(oracle_match / used_objects * 100.0),
        "worst_pick_rate_pct": float(worst_pick / used_objects * 100.0),
        "wilcoxon_p": p_value,
    }


def build_builtin_variants(args) -> List[Variant]:
    variants: List[Variant] = []

    difix = load_json(args.difix_scores_json)
    for key, metric_name in [
        ("scores.difix_mv_mean", "LPIPS"),
        ("scores.difix_front", "LPIPS-front"),
        ("scores.difix_mv_max", "LPIPS-max"),
        ("scores.difix_mv_front_weighted", "LPIPS-front-weighted"),
    ]:
        try:
            score_dict = normalize_score_dict(nested_get(difix, key))
            variants.append(
                Variant(
                    name=f"Difix3D+ x {metric_name}",
                    refiner="Difix3D+",
                    metric=metric_name,
                    scores=score_dict,
                    higher_is_better=True,
                )
            )
        except KeyError:
            continue

    clip_data = load_json(args.clip_scores_json)
    per_clip = clip_data.get("per_object_clip", {})
    if per_clip:
        variants.append(
            Variant(
                name="NoRefine x CLIP",
                refiner="NoRefine",
                metric="CLIP cosine",
                scores=normalize_score_dict({obj: v.get("scores", {}) for obj, v in per_clip.items()}),
                higher_is_better=True,
            )
        )
    per_dino = clip_data.get("per_object_dino", {})
    if per_dino:
        variants.append(
            Variant(
                name="NoRefine x DINOv2",
                refiner="NoRefine",
                metric="DINO cosine",
                scores=normalize_score_dict({obj: v.get("scores", {}) for obj, v in per_dino.items()}),
                higher_is_better=True,
            )
        )

    results = load_json(args.results_json)
    psnr_scores = {}
    for obj, payload in results.items():
        seeds = payload.get("seeds", {})
        cur = {}
        for seed, seed_payload in seeds.items():
            if "psnr" in seed_payload:
                cur[str(seed)] = float(seed_payload["psnr"])
        if len(cur) >= 2:
            psnr_scores[obj] = cur
    if psnr_scores:
        variants.append(
            Variant(
                name="NoRefine x PSNR",
                refiner="NoRefine",
                metric="PSNR",
                scores=psnr_scores,
                higher_is_better=True,
            )
        )

    return variants


def load_extra_variants(extra_scores_json: str) -> List[Variant]:
    if not extra_scores_json:
        return []
    payload = load_json(extra_scores_json)
    variants = []
    for entry in payload:
        variants.append(
            Variant(
                name=entry["name"],
                refiner=entry["refiner"],
                metric=entry["metric"],
                scores=normalize_score_dict(entry["scores"]),
                higher_is_better=bool(entry.get("higher_is_better", True)),
            )
        )
    return variants


def write_csv(rows: List[Dict[str, float]], path: str):
    if not rows:
        return
    fieldnames = [
        "variant",
        "refiner",
        "metric",
        "n_objects",
        "improvement_pct",
        "gap_closed_pct",
        "pairwise_acc_pct",
        "pairwise_acc_global_pct",
        "wilcoxon_p",
        "win_rate_pct",
        "oracle_match_rate_pct",
        "worst_pick_rate_pct",
        "wins",
        "ties",
        "losses",
        "default_cd",
        "selected_cd",
        "oracle_cd",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Refiner x metric sensitivity analysis")
    parser.add_argument("--results_json", default=DEFAULT_RESULTS)
    parser.add_argument("--difix_scores_json", default=DEFAULT_DIFIX)
    parser.add_argument("--clip_scores_json", default=DEFAULT_CLIP)
    parser.add_argument("--extra_scores_json", default="", help="Optional JSON list with extra variants")
    parser.add_argument("--default_seed", default="42")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt_cd = extract_gt_cd(args.results_json)

    variants = build_builtin_variants(args) + load_extra_variants(args.extra_scores_json)
    if not variants:
        raise RuntimeError("No variants available to evaluate.")

    rows = []
    for variant in variants:
        rows.append(evaluate_variant(variant, gt_cd, default_seed=args.default_seed))

    rows = sorted(rows, key=lambda x: x["improvement_pct"], reverse=True)

    json_path = os.path.join(args.out_dir, "refiner_metric_sensitivity.json")
    csv_path = os.path.join(args.out_dir, "refiner_metric_sensitivity.csv")
    with open(json_path, "w") as f:
        json.dump({"results": rows}, f, indent=2)
    write_csv(rows, csv_path)

    print(f"\nSaved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print("\nTop variants by CD improvement:")
    for row in rows[:6]:
        print(
            f"  {row['variant']:<34} "
            f"improv={row['improvement_pct']:+.2f}%  "
            f"gap={row['gap_closed_pct']:.1f}%  "
            f"pairwise={row['pairwise_acc_pct']:.1f}%  "
            f"p={row['wilcoxon_p']:.3g}"
        )


if __name__ == "__main__":
    main()

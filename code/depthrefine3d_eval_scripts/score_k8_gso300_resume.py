#!/usr/bin/env python3
"""Incremental SeedSelect scoring for GSO-300 K=8.

Compared with scripts/score_k8_gso300.py, this script supports resuming from an
existing score file and only computes missing (object, seed) pairs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "Difix3D" / "src"))

from pipeline_difix import DifixPipeline
import lpips as lp


VIEWS = ["front", "back", "left", "right", "top", "front_right"]
DEFAULT_SEED = "42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        default=str(REPO / "outputs/multiseed/gso_full_k8/results.json"),
    )
    parser.add_argument(
        "--renders-dir",
        default=str(REPO / "outputs/multiseed/gso_full_k8/renders_mv"),
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO / "outputs/multiseed/gso_full_k8/difix_multiview_scores.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing output-json and only score missing seeds.",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=50,
        help="Print progress every N newly computed seeds.",
    )
    return parser.parse_args()


def try_score_seed(
    pipe: DifixPipeline,
    lpips_fn: torch.nn.Module,
    render_dir: Path,
    seed: str,
) -> float | None:
    view_deltas = []
    for view_name in VIEWS:
        render_path = render_dir / f"seed{seed}_{view_name}.png"
        if not render_path.exists():
            return None
        try:
            rendered = Image.open(str(render_path)).convert("RGB")
            with torch.no_grad():
                output = pipe(
                    "remove degradation",
                    image=rendered,
                    num_inference_steps=1,
                    timesteps=[199],
                    guidance_scale=0.0,
                )
            fixed = output.images[0]
            size = 256
            r_arr = np.array(rendered.resize((size, size))).astype(np.float32) / 255.0
            f_arr = np.array(fixed.resize((size, size))).astype(np.float32) / 255.0
            r_t = torch.from_numpy(r_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            f_t = torch.from_numpy(f_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            r_t = r_t.cuda(non_blocking=True)
            f_t = f_t.cuda(non_blocking=True)
            with torch.no_grad():
                delta = lpips_fn(r_t, f_t).item()
            view_deltas.append(delta)
        except Exception:
            return None
    if len(view_deltas) != len(VIEWS):
        return None
    return float(-np.mean(view_deltas))  # higher is better


def summarize_scores(results: dict, all_scores: dict) -> dict:
    default_cds, selected_cds, oracle_cds = [], [], []
    wins = ties = losses = 0
    picks_oracle = picks_worst = 0

    for obj_id, obj_scores in all_scores.items():
        if obj_id not in results:
            continue
        seed_data = results[obj_id].get("seeds", {})
        if DEFAULT_SEED not in seed_data:
            continue
        # Keep only seeds that have both CD and score
        valid_seeds = [s for s in obj_scores.keys() if s in seed_data]
        if len(valid_seeds) < 2:
            continue

        best_seed = max(valid_seeds, key=lambda s: obj_scores[s])
        oracle_seed = min(valid_seeds, key=lambda s: seed_data[s]["cd"])
        worst_seed = max(valid_seeds, key=lambda s: seed_data[s]["cd"])

        sel_cd = float(seed_data[best_seed]["cd"])
        def_cd = float(seed_data[DEFAULT_SEED]["cd"])
        ora_cd = float(seed_data[oracle_seed]["cd"])

        default_cds.append(def_cd)
        selected_cds.append(sel_cd)
        oracle_cds.append(ora_cd)

        if best_seed == oracle_seed:
            picks_oracle += 1
        if best_seed == worst_seed:
            picks_worst += 1
        if sel_cd < def_cd - 1e-6:
            wins += 1
        elif sel_cd > def_cd + 1e-6:
            losses += 1
        else:
            ties += 1

    n = len(default_cds)
    if n == 0:
        return {
            "n": 0,
            "error": "No comparable objects with both default seed and scored candidates.",
        }

    default_cds = np.array(default_cds)
    selected_cds = np.array(selected_cds)
    oracle_cds = np.array(oracle_cds)

    improvement = (default_cds.mean() - selected_cds.mean()) / default_cds.mean() * 100.0
    oracle_improv = (default_cds.mean() - oracle_cds.mean()) / default_cds.mean() * 100.0
    gap_closed = (improvement / oracle_improv * 100.0) if oracle_improv > 0 else 0.0

    diffs = default_cds - selected_cds
    nonzero = diffs[np.abs(diffs) > 1e-12]
    _, w_pval = stats.wilcoxon(nonzero) if len(nonzero) > 10 else (0.0, 1.0)

    return {
        "n": int(n),
        "default_cd": float(default_cds.mean()),
        "selected_cd": float(selected_cds.mean()),
        "oracle_cd": float(oracle_cds.mean()),
        "improvement_pct": float(improvement),
        "oracle_improvement_pct": float(oracle_improv),
        "gap_closed_pct": float(gap_closed),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "oracle_match_rate": float(picks_oracle / n),
        "worst_pick_rate": float(picks_worst / n),
        "wilcoxon_pval": float(w_pval),
    }


def main() -> None:
    args = parse_args()

    with open(args.results_json, "r") as f:
        results = json.load(f)
    obj_ids = list(results.keys())

    existing_scores = {}
    if args.resume and os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            prev = json.load(f)
        existing_scores = prev.get("scores", {})
        print(f"Loaded existing scores for {len(existing_scores)} objects from {args.output_json}")

    total_seed_pairs = 0
    missing_seed_pairs = 0
    for obj_id in obj_ids:
        seeds = results[obj_id].get("seeds", {})
        for s in seeds.keys():
            total_seed_pairs += 1
            if s not in existing_scores.get(obj_id, {}):
                missing_seed_pairs += 1

    print("=" * 70)
    print("GSO-300 K=8 incremental Difix scoring")
    print(f"Objects: {len(obj_ids)}")
    print(f"Total seed pairs in results.json: {total_seed_pairs}")
    print(f"Missing seed pairs to compute: {missing_seed_pairs}")
    print("=" * 70)

    if missing_seed_pairs == 0:
        print("No missing scores. Recomputing summary only.")

    print("Loading Difix3D+ pipeline...")
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")
    print("Loading LPIPS...")
    lpips_fn = lp.LPIPS(net="alex").to("cuda").eval()

    all_scores = {obj_id: dict(v) for obj_id, v in existing_scores.items()}
    computed = 0
    t0 = time.time()
    renders_dir = Path(args.renders_dir)

    for obj_id in obj_ids:
        seed_data = results[obj_id].get("seeds", {})
        if not seed_data:
            continue
        obj_render_dir = renders_dir / obj_id
        if not obj_render_dir.exists():
            continue

        obj_scores = all_scores.setdefault(obj_id, {})
        for s in sorted(seed_data.keys(), key=int):
            if s in obj_scores:
                continue
            score = try_score_seed(pipe, lpips_fn, obj_render_dir, s)
            if score is None:
                continue
            obj_scores[s] = score
            computed += 1
            if computed % args.status_every == 0:
                elapsed = time.time() - t0
                rate = computed / elapsed if elapsed > 0 else 0.0
                print(f"[{computed}/{missing_seed_pairs}] elapsed={elapsed:.0f}s rate={rate:.2f} seeds/s")

    elapsed = time.time() - t0
    print(f"Computed new scores: {computed} in {elapsed:.0f}s")

    summary = summarize_scores(results, all_scores)
    output = {
        "summary": summary,
        "scores": all_scores,
        "metadata": {
            "results_json": args.results_json,
            "renders_dir": args.renders_dir,
            "resume": bool(args.resume),
            "newly_computed_seed_pairs": int(computed),
            "total_seed_pairs_in_results": int(total_seed_pairs),
            "missing_seed_pairs_before_run": int(missing_seed_pairs),
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {args.output_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute CD/FS metrics for GSO-300 K=8 from generated meshes.

This script is evaluation-only and does not run InstantMesh generation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.config import load_data_paths
from src.data.gso import GSOIndex
from src.utils.mesh import load_mesh, normalize_mesh
from src.utils.metrics import chamfer_fscore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-json",
        type=str,
        default=str(REPO / "configs/gso_eval.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO / "outputs/multiseed/gso_full_k8"),
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=str(REPO / "outputs/multiseed/gso_full_k8/results.json"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,42",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16000,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
    )
    return parser.parse_args()


def evaluate_mesh(mesh_path: str, gt_mesh_path: str, num_samples: int) -> tuple[float, float]:
    pred = normalize_mesh(load_mesh(mesh_path))
    gt = normalize_mesh(load_mesh(gt_mesh_path))
    cd, fs = chamfer_fscore(pred, gt, num_samples=num_samples, fscore_thresh=0.2)
    return float(cd), float(fs)


def mesh_path(out_dir: str, obj_id: str, seed: int) -> str:
    return os.path.join(
        out_dir, obj_id, f"seed{seed}", "instant-mesh-large", "meshes", f"{obj_id}.obj"
    )


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    with open(args.split_json, "r") as f:
        obj_ids = json.load(f)
    if os.path.exists(args.results_json):
        with open(args.results_json, "r") as f:
            results = json.load(f)
    else:
        results = {}

    paths = load_data_paths()
    gso_idx = GSOIndex(paths["gso_root"])

    updated = 0
    missing_mesh = 0
    missing_gt = 0
    failed_eval = 0

    for idx, obj_id in enumerate(obj_ids, start=1):
        gt = gso_idx.mesh_path(obj_id)
        if not os.path.exists(gt):
            missing_gt += 1
            continue
        obj_entry = results.setdefault(obj_id, {})
        seeds_entry = obj_entry.setdefault("seeds", {})

        for seed in seeds:
            mp = mesh_path(args.out_dir, obj_id, seed)
            if not os.path.exists(mp):
                missing_mesh += 1
                continue
            try:
                cd, fs = evaluate_mesh(mp, gt, args.num_samples)
                prev = seeds_entry.get(str(seed), {})
                seeds_entry[str(seed)] = {
                    "cd": cd,
                    "fs": fs,
                    "psnr": float(prev.get("psnr", -1.0)),
                }
                updated += 1
            except Exception:
                failed_eval += 1

        valid = seeds_entry
        if valid:
            best_seed = min(valid.keys(), key=lambda s: valid[s]["cd"])
            obj_entry["best_by_cd"] = {
                "seed": int(best_seed),
                "cd": float(valid[best_seed]["cd"]),
                "fs": float(valid[best_seed]["fs"]),
            }
            obj_entry["default"] = valid.get("42", {})

        if idx % args.save_every == 0:
            with open(args.results_json, "w") as f:
                json.dump(results, f, indent=2)
            print(
                f"[{idx}/{len(obj_ids)}] updated={updated} missing_mesh={missing_mesh} "
                f"failed_eval={failed_eval}"
            )

    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")
    print(
        f"updated={updated}, missing_mesh={missing_mesh}, missing_gt={missing_gt}, "
        f"failed_eval={failed_eval}"
    )


if __name__ == "__main__":
    main()

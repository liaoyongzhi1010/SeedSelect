#!/usr/bin/env python3
"""Generate missing meshes for GSO-300 K=8 with global parallel queue.

This script only handles mesh generation and never writes CD/FS metrics.
It is designed to keep GPU utilization high by scheduling missing (object, seed)
jobs globally instead of per-object batching.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-json",
        type=str,
        default=str(REPO / "configs/gso_eval.json"),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(REPO / "outputs/inputs/gso_eval"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO / "outputs/multiseed/gso_full_k8"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,42",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--instantmesh-python",
        type=str,
        default=os.environ.get("INSTANTMESH_PYTHON", "/root/miniconda3/envs/instantmesh/bin/python"),
    )
    parser.add_argument(
        "--instantmesh-dir",
        type=str,
        default=os.environ.get("INSTANTMESH_DIR", str(REPO / "third_party" / "InstantMesh")),
    )
    return parser.parse_args()


def _mesh_path(out_dir: str, obj_id: str, seed: int) -> str:
    return os.path.join(
        out_dir, obj_id, f"seed{seed}", "instant-mesh-large", "meshes", f"{obj_id}.obj"
    )


def run_one(job: tuple[str, int, str, str, str, str, int]) -> tuple[str, int, str]:
    obj_id, seed, input_img, out_dir, instantmesh_python, instantmesh_dir, timeout_sec = job
    mesh_path = _mesh_path(out_dir, obj_id, seed)
    if os.path.exists(mesh_path):
        return obj_id, seed, "skip"

    seed_dir = os.path.join(out_dir, obj_id, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    cfg = os.path.join(instantmesh_dir, "configs", "instant-mesh-large.yaml")
    cmd = [
        instantmesh_python,
        os.path.join(instantmesh_dir, "run.py"),
        cfg,
        os.path.abspath(input_img),
        "--output_path",
        os.path.abspath(seed_dir),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
    # Keep CUDA toolchain search path deterministic to avoid malformed inherited env.
    env["PATH"] = (
        f"{os.path.dirname(instantmesh_python)}:/usr/local/cuda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
    try:
        r = subprocess.run(
            cmd,
            cwd=instantmesh_dir,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            check=False,
        )
        if r.returncode == 0 and os.path.exists(mesh_path):
            return obj_id, seed, "ok"
        return obj_id, seed, "fail"
    except subprocess.TimeoutExpired:
        return obj_id, seed, "timeout"


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = args.out_dir
    input_dir = args.input_dir

    with open(args.split_json, "r") as f:
        obj_ids = json.load(f)

    jobs: list[tuple[str, int, str, str, str, str, int]] = []
    for obj_id in obj_ids:
        input_img = os.path.join(input_dir, f"{obj_id}.png")
        if not os.path.exists(input_img):
            continue
        for seed in seeds:
            if os.path.exists(_mesh_path(out_dir, obj_id, seed)):
                continue
            jobs.append(
                (
                    obj_id,
                    seed,
                    input_img,
                    out_dir,
                    args.instantmesh_python,
                    args.instantmesh_dir,
                    args.timeout_sec,
                )
            )

    print("=" * 70)
    print("Fill missing meshes for GSO-300 K=8")
    print(f"Objects in split: {len(obj_ids)}")
    print(f"Target seeds: {seeds}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Missing jobs: {len(jobs)}")
    print("=" * 70)

    if not jobs:
        print("No missing mesh jobs. Done.")
        return

    start = time.time()
    done = skip = fail = timeout = 0
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futs = [pool.submit(run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            _, _, status = fut.result()
            if status == "ok":
                done += 1
            elif status == "skip":
                skip += 1
            elif status == "timeout":
                timeout += 1
            else:
                fail += 1
            if i % args.status_every == 0 or i == len(jobs):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                remain = len(jobs) - i
                eta = remain / rate if rate > 0 else 0.0
                print(
                    f"[{i}/{len(jobs)}] ok={done} skip={skip} fail={fail} timeout={timeout} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )

    total = time.time() - start
    print("=" * 70)
    print(f"Finished in {total:.0f}s")
    print(f"ok={done}, skip={skip}, fail={fail}, timeout={timeout}")
    print("=" * 70)


if __name__ == "__main__":
    main()

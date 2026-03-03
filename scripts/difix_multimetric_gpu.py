#!/usr/bin/env python3
"""GPU-heavy multi-metric Difix scoring on pre-rendered multi-view images.

This script runs Difix3D+ on each (object, seed, view) render and computes
refinement deltas under multiple metrics:
  - LPIPS (perceptual)
  - L1 (pixel absolute error)
  - 1 - SSIM (structural dissimilarity)

Scores are stored as negative deltas so that higher score = better candidate
(less refinement needed).
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim


VIEWS = ["front", "back", "left", "right", "top", "front_right"]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-metric Difix scoring (GPU)")
    parser.add_argument(
        "--results_json",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/results.json",
    )
    parser.add_argument(
        "--render_dir",
        default="/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/renders_mv",
    )
    parser.add_argument(
        "--difix_src",
        default="/root/eccv/DepthRefine3D/third_party/Difix3D/src",
        help="Path containing pipeline_difix.py",
    )
    parser.add_argument(
        "--out_json",
        default="/root/eccv/.worktrees/seedselect80/code/outputs/multimetric/difix_multimetric_scores.json",
    )
    parser.add_argument("--seeds", default="0,1,2,42")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--batch_views", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_images(paths: List[str], image_size: int) -> List[Image.Image]:
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if image_size > 0:
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        images.append(img)
    return images


def pil_to_lpips_tensor(images: List[Image.Image], device: str) -> torch.Tensor:
    arr = np.stack([np.asarray(im).astype(np.float32) / 255.0 for im in images], axis=0)
    x = torch.from_numpy(arr).permute(0, 3, 1, 2) * 2 - 1
    return x.to(device)


def pil_to_np(images: List[Image.Image]) -> np.ndarray:
    return np.stack([np.asarray(im).astype(np.float32) / 255.0 for im in images], axis=0)


def compute_metrics_batch(
    orig_images: List[Image.Image],
    fixed_images: List[Image.Image],
    lpips_fn,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lpips_delta, l1_delta, ssim_dist) arrays of shape [N]."""
    orig_t = pil_to_lpips_tensor(orig_images, device)
    fixed_t = pil_to_lpips_tensor(fixed_images, device)
    with torch.no_grad():
        lp = lpips_fn(orig_t, fixed_t).detach().cpu().view(-1).numpy()

    orig_np = pil_to_np(orig_images)
    fixed_np = pil_to_np(fixed_images)
    l1 = np.mean(np.abs(orig_np - fixed_np), axis=(1, 2, 3))

    sdist = []
    for i in range(orig_np.shape[0]):
        # skimage returns similarity in [0, 1], convert to distance.
        s_val = ssim(orig_np[i], fixed_np[i], channel_axis=2, data_range=1.0)
        sdist.append(1.0 - float(s_val))
    sdist = np.asarray(sdist, dtype=np.float32)
    return lp, l1, sdist


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.results_json) as f:
        results = json.load(f)
    obj_ids = sorted(results.keys())
    if args.max_objects > 0:
        obj_ids = obj_ids[: args.max_objects]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    existing = {}
    if args.resume and os.path.exists(args.out_json):
        with open(args.out_json) as f:
            existing = json.load(f)

    # Existing schema support.
    scores = existing.get(
        "scores",
        {
            "difix_mv_mean": {},
            "difix_front": {},
            "difix_mv_l1": {},
            "difix_front_l1": {},
            "difix_mv_ssim": {},
            "difix_front_ssim": {},
        },
    )
    for key in [
        "difix_mv_mean",
        "difix_front",
        "difix_mv_l1",
        "difix_front_l1",
        "difix_mv_ssim",
        "difix_front_ssim",
    ]:
        scores.setdefault(key, {})

    import sys

    sys.path.insert(0, args.difix_src)
    from pipeline_difix import DifixPipeline
    import lpips as lp

    device = "cuda"
    print("Loading Difix3D+ pipeline...")
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("Loading LPIPS...")
    lpips_fn = lp.LPIPS(net="alex").to(device).eval()

    t0 = time.time()
    done_objects = 0
    done_triplets = 0
    skipped_triplets = 0

    for i, obj_id in enumerate(obj_ids):
        if i % args.save_every == 0 and i > 0:
            with open(args.out_json, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "results_json": args.results_json,
                            "render_dir": args.render_dir,
                            "num_inference_steps": args.num_inference_steps,
                            "timestep": args.timestep,
                            "image_size": args.image_size,
                            "n_objects_processed": done_objects,
                            "elapsed_sec": time.time() - t0,
                        },
                        "scores": scores,
                    },
                    f,
                    indent=2,
                )

        for seed in seeds:
            if obj_id in scores["difix_mv_mean"] and seed in scores["difix_mv_mean"][obj_id]:
                skipped_triplets += 1
                continue

            view_paths = []
            view_names = []
            for vname in VIEWS:
                p = os.path.join(args.render_dir, obj_id, f"seed{seed}_{vname}.png")
                if os.path.exists(p):
                    view_paths.append(p)
                    view_names.append(vname)

            if len(view_paths) < 2:
                continue

            try:
                orig_images = load_images(view_paths, args.image_size)
                prompts = ["remove degradation"] * len(orig_images)
                with torch.no_grad():
                    out = pipe(
                        prompts,
                        image=orig_images,
                        num_inference_steps=args.num_inference_steps,
                        timesteps=[args.timestep],
                        guidance_scale=args.guidance_scale,
                    )
                fixed_images = out.images
                lp_arr, l1_arr, ssim_arr = compute_metrics_batch(orig_images, fixed_images, lpips_fn, device)

                idx_front = view_names.index("front") if "front" in view_names else 0

                scores["difix_mv_mean"].setdefault(obj_id, {})[seed] = float(-np.mean(lp_arr))
                scores["difix_front"].setdefault(obj_id, {})[seed] = float(-lp_arr[idx_front])
                scores["difix_mv_l1"].setdefault(obj_id, {})[seed] = float(-np.mean(l1_arr))
                scores["difix_front_l1"].setdefault(obj_id, {})[seed] = float(-l1_arr[idx_front])
                scores["difix_mv_ssim"].setdefault(obj_id, {})[seed] = float(-np.mean(ssim_arr))
                scores["difix_front_ssim"].setdefault(obj_id, {})[seed] = float(-ssim_arr[idx_front])

                done_triplets += 1
            except Exception as e:
                print(f"[WARN] {obj_id} seed={seed}: {e}")

        done_objects += 1
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(obj_ids) - i - 1) / max(rate, 1e-6)
            print(
                f"[{i+1}/{len(obj_ids)}] processed_triplets={done_triplets}, "
                f"skipped={skipped_triplets}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
            )

    with open(args.out_json, "w") as f:
        json.dump(
            {
                "metadata": {
                    "results_json": args.results_json,
                    "render_dir": args.render_dir,
                    "num_inference_steps": args.num_inference_steps,
                    "timestep": args.timestep,
                    "image_size": args.image_size,
                    "n_objects_processed": done_objects,
                    "triplets_scored": done_triplets,
                    "triplets_skipped": skipped_triplets,
                    "elapsed_sec": time.time() - t0,
                },
                "scores": scores,
            },
            f,
            indent=2,
        )

    elapsed = time.time() - t0
    print(f"Saved {args.out_json}")
    print(f"Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

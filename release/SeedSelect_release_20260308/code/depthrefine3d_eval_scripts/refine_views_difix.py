#!/usr/bin/env python3
"""Stage A: Refine Zero123++ multi-view images with Difix3D+.

Takes the saved 6-view grids from InstantMesh's Zero123++ stage,
applies Difix3D+ to each view individually to remove artifacts,
then reassembles into the same grid format.

Usage:
    /root/miniconda3/envs/difix3d/bin/python scripts/refine_views_difix.py \
        --mesh_dir outputs/multiseed/gso_full \
        --results outputs/multiseed/gso_full/results.json \
        --max_objects 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'third_party', 'Difix3D', 'src'))


def split_grid(grid_image):
    """Split 640x960 grid (2 cols × 3 rows) into 6 individual 320x320 views."""
    w, h = grid_image.size  # 640, 960
    vw, vh = w // 2, h // 3  # 320, 320
    views = []
    for row in range(3):
        for col in range(2):
            box = (col * vw, row * vh, (col + 1) * vw, (row + 1) * vh)
            views.append(grid_image.crop(box))
    return views  # 6 views in order


def assemble_grid(views, vw=320, vh=320):
    """Assemble 6 views back into 640x960 grid."""
    grid = Image.new('RGB', (vw * 2, vh * 3))
    for i, view in enumerate(views):
        row, col = i // 2, i % 2
        view_resized = view.resize((vw, vh), Image.LANCZOS)
        grid.paste(view_resized, (col * vw, row * vh))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh_dir', required=True, help='Dir with multi-seed outputs')
    ap.add_argument('--results', required=True, help='results.json from multi-seed experiment')
    ap.add_argument('--out_dir', default=None, help='Output dir for refined views (default: {mesh_dir}/refined_views)')
    ap.add_argument('--max_objects', type=int, default=0)
    ap.add_argument('--seed', default='42', help='Which seed to refine (default: best by some criterion)')
    ap.add_argument('--all_seeds', action='store_true', help='Refine all seeds')
    ap.add_argument('--timestep', type=int, default=199, help='Difix3D+ timestep (lower=more change)')
    ap.add_argument('--save_comparison', action='store_true', help='Save side-by-side before/after')
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    obj_ids = list(results.keys())
    if args.max_objects > 0:
        obj_ids = obj_ids[:args.max_objects]

    out_dir = args.out_dir or os.path.join(args.mesh_dir, 'refined_views')
    os.makedirs(out_dir, exist_ok=True)

    # Determine which seeds to process
    if args.all_seeds:
        seeds = sorted(results[obj_ids[0]].get('seeds', {}).keys(), key=int)
    else:
        seeds = [args.seed]

    print(f"Objects: {len(obj_ids)}, Seeds: {seeds}, Timestep: {args.timestep}")

    # Load Difix3D+
    print("Loading Difix3D+ pipeline...")
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained('nvidia/difix', trust_remote_code=True)
    pipe.to('cuda')
    print("  Ready.")

    t0 = time.time()
    processed = 0
    skipped = 0

    for i, obj_id in enumerate(obj_ids):
        for s in seeds:
            # Input: saved 6-view grid from InstantMesh
            grid_path = os.path.join(args.mesh_dir, obj_id, f'seed{s}',
                                     'instant-mesh-large', 'images', f'{obj_id}.png')
            if not os.path.exists(grid_path):
                continue

            # Output: refined grid
            out_grid_dir = os.path.join(out_dir, obj_id, f'seed{s}')
            out_grid_path = os.path.join(out_grid_dir, f'{obj_id}.png')
            if os.path.exists(out_grid_path):
                skipped += 1
                continue

            os.makedirs(out_grid_dir, exist_ok=True)

            try:
                grid = Image.open(grid_path).convert('RGB')
                views = split_grid(grid)

                # Apply Difix3D+ to each view
                refined_views = []
                for view in views:
                    # Difix3D+ expects reasonable resolution
                    # Views are 320x320, upscale to 512x512 for better Difix3D+ performance
                    view_up = view.resize((512, 512), Image.LANCZOS)
                    with torch.no_grad():
                        output = pipe('remove degradation', image=view_up,
                                      num_inference_steps=1, timesteps=[args.timestep],
                                      guidance_scale=0.0)
                    refined = output.images[0]
                    # Resize back to 320x320
                    refined_views.append(refined)

                # Reassemble grid
                refined_grid = assemble_grid(refined_views)
                refined_grid.save(out_grid_path)

                if args.save_comparison:
                    # Save side-by-side for visual inspection
                    comp = Image.new('RGB', (grid.width * 2, grid.height))
                    comp.paste(grid, (0, 0))
                    comp.paste(refined_grid, (grid.width, 0))
                    comp.save(os.path.join(out_grid_dir, f'{obj_id}_comparison.png'))

                processed += 1

            except Exception as e:
                print(f'  [WARN] Failed {obj_id} seed={s}: {e}')

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(obj_ids) - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{len(obj_ids)}] {processed} refined, {skipped} skipped, '
                  f'{elapsed:.0f}s, ETA {eta:.0f}s')

    print(f"\nDone: {processed} grids refined, {skipped} skipped, {time.time()-t0:.0f}s")
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()

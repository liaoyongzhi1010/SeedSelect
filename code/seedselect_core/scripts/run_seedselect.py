#!/usr/bin/env python3
"""SeedSelect: Run multi-seed selection on a single input image.

Generates K candidate 3D reconstructions, scores each using multi-view
Difix3D+ quality proxy, and saves the best candidate.

Usage:
    PYOPENGL_PLATFORM=egl python scripts/run_seedselect.py \
        --image path/to/input.png \
        --backbone instantmesh \
        --k 4 \
        --output_dir results/

Requirements:
    - InstantMesh installed in third_party/InstantMesh/
    - Difix3D+ model (auto-downloaded from HuggingFace)
    - EGL rendering support (headless) or display server
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.utils.mesh import load_mesh, normalize_mesh
from src.utils.camera import look_at
from src.utils.render import render_mesh

# Default seeds
DEFAULT_SEEDS = [0, 1, 2, 42]

# 6 canonical viewpoints for scoring
VIEWS = [
    ('front',       (0, 0, 2.0),       (0, 1, 0)),
    ('back',        (0, 0, -2.0),      (0, 1, 0)),
    ('left',        (-2.0, 0, 0),      (0, 1, 0)),
    ('right',       (2.0, 0, 0),       (0, 1, 0)),
    ('top',         (0, 2.0, 0),       (0, 0, -1)),
    ('front_right', (1.41, 0, 1.41),   (0, 1, 0)),
]


def generate_candidates(image_path, output_dir, seeds, backbone='instantmesh'):
    """Generate K candidate meshes using different seeds."""
    mesh_paths = {}

    if backbone == 'instantmesh':
        instantmesh_dir = os.path.join(REPO, 'third_party', 'InstantMesh')
        instantmesh_cfg = os.path.join(instantmesh_dir, 'configs', 'instant-mesh-large.yaml')

        # Find python in current env or system
        python_bin = sys.executable

        for seed in seeds:
            print(f"  Generating candidate with seed={seed}...")
            seed_dir = os.path.join(output_dir, f'seed{seed}')
            os.makedirs(seed_dir, exist_ok=True)

            env = os.environ.copy()
            cmd = [
                python_bin, os.path.join(instantmesh_dir, 'run.py'),
                instantmesh_cfg, os.path.abspath(image_path),
                '--output_path', os.path.abspath(seed_dir),
                '--seed', str(seed),
            ]

            result = subprocess.run(cmd, cwd=instantmesh_dir, env=env,
                                    capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"    [WARN] seed={seed} failed: {result.stderr[-200:] if result.stderr else 'no error'}")
                continue

            # Find output mesh
            name = os.path.splitext(os.path.basename(image_path))[0]
            mesh_path = os.path.join(seed_dir, 'instant-mesh-large', 'meshes', f'{name}.obj')
            if os.path.exists(mesh_path):
                mesh_paths[seed] = mesh_path
                print(f"    [OK] seed={seed} -> {mesh_path}")
            else:
                print(f"    [WARN] seed={seed}: mesh not found at {mesh_path}")
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Use 'instantmesh'.")

    return mesh_paths


def render_views(mesh_path, render_size=512):
    """Render mesh from 6 canonical viewpoints."""
    mesh = load_mesh(mesh_path)
    mesh = normalize_mesh(mesh, mode='unit_cube')

    images = []
    for name, eye, up in VIEWS:
        c2w = look_at(eye, up=up)
        color, _ = render_mesh(mesh, c2w, width=render_size, height=render_size,
                               yfov=np.deg2rad(50.0))
        images.append((Image.fromarray(color), name))
    return images


def score_candidates(mesh_paths, render_size=512):
    """Score each candidate using multi-view Difix3D+ quality proxy."""
    import torch
    import lpips as lp

    # Lazy import Difix3D+ pipeline
    from diffusers import StableDiffusionPipeline
    try:
        sys.path.insert(0, os.path.join(REPO, 'third_party', 'Difix3D', 'src'))
        from pipeline_difix import DifixPipeline
        pipe = DifixPipeline.from_pretrained('nvidia/difix', trust_remote_code=True)
    except ImportError:
        # Fallback: try loading from HuggingFace directly
        from diffusers import AutoPipelineForImage2Image
        pipe = AutoPipelineForImage2Image.from_pretrained(
            'nvidia/difix3d-plus', trust_remote_code=True)
    pipe.to('cuda')

    lpips_fn = lp.LPIPS(net='alex').to('cuda').eval()

    scores = {}

    for seed, mesh_path in mesh_paths.items():
        print(f"  Scoring seed={seed}...")
        try:
            views = render_views(mesh_path, render_size)
        except Exception as e:
            print(f"    [WARN] Rendering failed: {e}")
            continue

        deltas = []
        for img, vname in views:
            with torch.no_grad():
                output = pipe('remove degradation', image=img,
                              num_inference_steps=1, timesteps=[199],
                              guidance_scale=0.0)
            fixed = output.images[0]

            # Compute LPIPS between original and refined
            size = 256
            r_arr = np.array(img.resize((size, size))).astype(np.float32) / 255.0
            f_arr = np.array(fixed.resize((size, size))).astype(np.float32) / 255.0
            r_t = torch.from_numpy(r_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            f_t = torch.from_numpy(f_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1

            with torch.no_grad():
                delta = lpips_fn(r_t.cuda(), f_t.cuda()).item()
            deltas.append(delta)

        if deltas:
            # Lower mean delta = less refinement needed = better quality
            scores[seed] = -np.mean(deltas)
            print(f"    Score: {scores[seed]:.6f} (mean delta: {np.mean(deltas):.4f})")

    return scores


def main():
    parser = argparse.ArgumentParser(description='SeedSelect: Multi-seed 3D reconstruction selection')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--backbone', default='instantmesh', choices=['instantmesh'],
                        help='3D reconstruction backbone')
    parser.add_argument('--k', type=int, default=4, help='Number of candidate seeds')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 0,1,2,42 for K=4)')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--render_size', type=int, default=512, help='Rendering resolution')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip generation, use existing meshes')
    args = parser.parse_args()

    # Parse seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = DEFAULT_SEEDS[:args.k]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"SeedSelect | backbone={args.backbone} | K={len(seeds)} | seeds={seeds}")
    print("=" * 60)

    # Step 1: Generate candidates
    t0 = time.time()
    if not args.skip_generation:
        print(f"\n[Step 1/3] Generating {len(seeds)} candidates...")
        mesh_paths = generate_candidates(args.image, args.output_dir, seeds, args.backbone)
    else:
        print(f"\n[Step 1/3] Loading existing meshes...")
        mesh_paths = {}
        name = os.path.splitext(os.path.basename(args.image))[0]
        for seed in seeds:
            mp = os.path.join(args.output_dir, f'seed{seed}',
                              'instant-mesh-large', 'meshes', f'{name}.obj')
            if os.path.exists(mp):
                mesh_paths[seed] = mp

    if len(mesh_paths) < 2:
        print(f"[ERROR] Need at least 2 candidates, got {len(mesh_paths)}")
        return

    print(f"  Generated {len(mesh_paths)} candidates in {time.time()-t0:.1f}s")

    # Step 2: Score candidates
    t1 = time.time()
    print(f"\n[Step 2/3] Scoring candidates with multi-view Difix3D+...")
    scores = score_candidates(mesh_paths, args.render_size)
    print(f"  Scored {len(scores)} candidates in {time.time()-t1:.1f}s")

    if not scores:
        print("[ERROR] No candidates scored successfully")
        return

    # Step 3: Select best
    best_seed = max(scores, key=scores.get)
    best_mesh = mesh_paths[best_seed]

    print(f"\n[Step 3/3] Selection results:")
    print(f"  {'Seed':<8} {'Score':<12} {'Selected'}")
    print(f"  {'-'*30}")
    for seed in sorted(scores.keys()):
        marker = " <-- BEST" if seed == best_seed else ""
        print(f"  {seed:<8} {scores[seed]:+.6f}{marker}")

    # Copy best mesh
    best_output = os.path.join(args.output_dir, 'best.obj')
    shutil.copy2(best_mesh, best_output)
    print(f"\n  Best mesh (seed={best_seed}) saved to: {best_output}")

    # Save results
    results = {
        'image': os.path.abspath(args.image),
        'backbone': args.backbone,
        'K': len(seeds),
        'seeds': seeds,
        'scores': {str(k): v for k, v in scores.items()},
        'best_seed': best_seed,
        'best_mesh': best_output,
    }
    results_path = os.path.join(args.output_dir, 'seedselect_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"\n  Total time: {total:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()

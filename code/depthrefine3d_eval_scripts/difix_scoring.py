#!/usr/bin/env python3
"""Score reconstructed meshes using Difix3D+ as a quality discriminator.

Hypothesis: Difix3D+ is trained to fix rendering artifacts from 3D reconstructions.
A high-quality mesh will have renderings that need LESS fixing → smaller Difix3D+ delta.

For each seed's mesh:
1. Render front view
2. Apply Difix3D+ to the rendered view
3. Compute delta = LPIPS(original_render, difix_render)
4. Score = -delta (less change = better quality)

Also tests: Difix3D+ refinement + re-reconstruction pipeline.

Usage:
    PYOPENGL_PLATFORM=egl /root/miniconda3/envs/difix3d/bin/python scripts/difix_scoring.py \
        --results outputs/multiseed/gso_full/results.json \
        --mesh_dir outputs/multiseed/gso_full \
        --input_dir outputs/inputs/gso_eval \
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
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'third_party', 'Difix3D', 'src'))

from src.utils.mesh import load_mesh, normalize_mesh
from src.utils.camera import look_at
from src.utils.render import render_mesh


def render_views(mesh_path, views='front', size=512):
    """Render mesh from specified viewpoints. Returns list of (PIL.Image, view_name)."""
    mesh = load_mesh(mesh_path)
    mesh = normalize_mesh(mesh, mode='unit_cube')

    results = []
    if views == 'front' or views == 'multi':
        c2w = look_at((0, 0, 2.0))
        color, _ = render_mesh(mesh, c2w, width=size, height=size, yfov=np.deg2rad(50.0))
        results.append((Image.fromarray(color), 'front'))

    if views == 'multi':
        for name, eye in [('left', (-2, 0, 0)), ('right', (2, 0, 0)),
                          ('top', (0, 2, 0)), ('back', (0, 0, -2))]:
            c2w = look_at(eye)
            color, _ = render_mesh(mesh, c2w, width=size, height=size, yfov=np.deg2rad(50.0))
            results.append((Image.fromarray(color), name))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--mesh_dir', required=True)
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--max_objects', type=int, default=0)
    ap.add_argument('--views', choices=['front', 'multi'], default='front')
    ap.add_argument('--render_size', type=int, default=512)
    ap.add_argument('--save_images', action='store_true')
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    obj_ids = list(results.keys())
    if args.max_objects > 0:
        obj_ids = obj_ids[:args.max_objects]

    seeds = sorted(results[obj_ids[0]].get('seeds', {}).keys(), key=int)
    print(f"Seeds: {seeds}, Objects: {len(obj_ids)}, Views: {args.views}")

    # Phase 1: Pre-render all views (CPU/EGL only, no CUDA)
    print("\n--- Phase 1: Rendering views ---")
    render_dir = os.path.join(os.path.dirname(args.results), 'renders')
    os.makedirs(render_dir, exist_ok=True)

    t0 = time.time()
    rendered_count = 0
    for i, obj_id in enumerate(obj_ids):
        seed_data = results[obj_id].get('seeds', {})
        if len(seed_data) < 2:
            continue

        for s in seeds:
            mp = os.path.join(args.mesh_dir, obj_id, f'seed{s}',
                              'instant-mesh-large', 'meshes', f'{obj_id}.obj')
            out_dir = os.path.join(render_dir, obj_id)
            os.makedirs(out_dir, exist_ok=True)

            front_path = os.path.join(out_dir, f'seed{s}_front.png')
            if os.path.exists(front_path):
                rendered_count += 1
                continue

            if not os.path.exists(mp):
                continue

            try:
                views = render_views(mp, views=args.views, size=args.render_size)
                for img, vname in views:
                    img.save(os.path.join(out_dir, f'seed{s}_{vname}.png'))
                rendered_count += 1
            except Exception as e:
                print(f'  [WARN] Render failed for {obj_id} seed={s}: {e}')

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(obj_ids)}] {rendered_count} renders, {elapsed:.0f}s')

    print(f"  Rendered {rendered_count} views in {time.time()-t0:.0f}s")
    print(f"  Saved to {render_dir}")

    # Phase 2: Apply Difix3D+ and compute deltas
    print("\n--- Phase 2: Difix3D+ scoring ---")
    from pipeline_difix import DifixPipeline
    import lpips as lp

    print("  Loading Difix3D+ pipeline...")
    pipe = DifixPipeline.from_pretrained('nvidia/difix', trust_remote_code=True)
    pipe.to('cuda')
    print("  Loading LPIPS...")
    lpips_fn = lp.LPIPS(net='alex').to('cuda').eval()

    difix_scores = {}  # obj_id -> {seed: score}
    difix_lpips_input = {}  # obj_id -> {seed: LPIPS(difix_render, input_image)}
    t1 = time.time()

    for i, obj_id in enumerate(obj_ids):
        seed_data = results[obj_id].get('seeds', {})
        if len(seed_data) < 2:
            continue

        input_img = os.path.join(args.input_dir, f'{obj_id}.png')
        input_pil = Image.open(input_img).convert('RGB') if os.path.exists(input_img) else None

        obj_scores = {}
        obj_lpips_input = {}

        for s in seeds:
            render_path = os.path.join(render_dir, obj_id, f'seed{s}_front.png')
            if not os.path.exists(render_path):
                continue

            try:
                rendered = Image.open(render_path).convert('RGB')

                # Apply Difix3D+
                with torch.no_grad():
                    output = pipe('remove degradation', image=rendered,
                                  num_inference_steps=1, timesteps=[199],
                                  guidance_scale=0.0)
                fixed = output.images[0]

                if args.save_images:
                    fixed.save(os.path.join(render_dir, obj_id, f'seed{s}_front_difix.png'))

                # Compute delta: LPIPS between original render and Difix3D+ output
                size = 256
                r_arr = np.array(rendered.resize((size, size))).astype(np.float32) / 255.0
                f_arr = np.array(fixed.resize((size, size))).astype(np.float32) / 255.0

                r_t = torch.from_numpy(r_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                f_t = torch.from_numpy(f_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                r_t, f_t = r_t.cuda(), f_t.cuda()

                with torch.no_grad():
                    delta = lpips_fn(r_t, f_t).item()

                # Score: less change = better reconstruction
                obj_scores[s] = -delta

                # Also: LPIPS between Difix3D+ output and input image
                if input_pil is not None:
                    i_arr = np.array(input_pil.resize((size, size))).astype(np.float32) / 255.0
                    i_t = torch.from_numpy(i_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    i_t = i_t.cuda()
                    with torch.no_grad():
                        lpips_to_input = lpips_fn(f_t, i_t).item()
                    obj_lpips_input[s] = -lpips_to_input  # Higher = more similar to input

            except Exception as e:
                print(f'  [WARN] Difix3D+ failed for {obj_id} seed={s}: {e}')

        if obj_scores:
            difix_scores[obj_id] = obj_scores
        if obj_lpips_input:
            difix_lpips_input[obj_id] = obj_lpips_input

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(obj_ids) - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{len(obj_ids)}] {elapsed:.0f}s, ETA {eta:.0f}s')

    # Phase 3: Analyze
    print(f"\n{'='*70}")
    print(f"DIFIX3D+ SCORING ANALYSIS — {len(obj_ids)} objects")
    print(f"{'='*70}")

    from scipy import stats

    for method_name, method_scores in [('difix_delta', difix_scores),
                                        ('difix_lpips_input', difix_lpips_input)]:
        if not method_scores:
            continue

        default_cds, selected_cds, oracle_cds = [], [], []
        wins, ties, losses = 0, 0, 0
        picks_oracle, picks_worst = 0, 0
        n = 0

        for obj_id, obj_scores in method_scores.items():
            res = results[obj_id]
            seed_data = res.get('seeds', {})
            default = seed_data.get('42', {})
            if not default:
                continue

            valid = {s: v for s, v in obj_scores.items() if v > -900}
            if not valid:
                continue

            best_seed = max(valid, key=valid.get)
            best_by_cd = min(seed_data.items(), key=lambda x: x[1]['cd'])
            worst_by_cd = max(seed_data.items(), key=lambda x: x[1]['cd'])

            sel_cd = seed_data[best_seed]['cd']
            def_cd = default['cd']
            ora_cd = best_by_cd[1]['cd']

            default_cds.append(def_cd)
            selected_cds.append(sel_cd)
            oracle_cds.append(ora_cd)

            if best_seed == best_by_cd[0]:
                picks_oracle += 1
            if best_seed == worst_by_cd[0]:
                picks_worst += 1
            if sel_cd < def_cd - 1e-6:
                wins += 1
            elif sel_cd > def_cd + 1e-6:
                losses += 1
            else:
                ties += 1
            n += 1

        if n == 0:
            continue

        default_cds = np.array(default_cds)
        selected_cds = np.array(selected_cds)
        oracle_cds = np.array(oracle_cds)

        improvement = (default_cds.mean() - selected_cds.mean()) / default_cds.mean() * 100
        oracle_improv = (default_cds.mean() - oracle_cds.mean()) / default_cds.mean() * 100
        gap_closed = improvement / oracle_improv * 100 if oracle_improv > 0 else 0

        t_stat, t_pval = stats.ttest_rel(default_cds, selected_cds)

        print(f"\n--- {method_name} ---")
        print(f"  n={n}")
        print(f"  Default CD:     {default_cds.mean():.6f}")
        print(f"  Selected CD:    {selected_cds.mean():.6f}")
        print(f"  Oracle CD:      {oracle_cds.mean():.6f}")
        print(f"  Improvement:    {improvement:+.2f}%")
        print(f"  Gap closed:     {gap_closed:.1f}%")
        print(f"  Wins/Ties/Loss: {wins}/{ties}/{losses} ({wins/n*100:.1f}%/{ties/n*100:.1f}%/{losses/n*100:.1f}%)")
        print(f"  Oracle match:   {picks_oracle}/{n} ({picks_oracle/n*100:.1f}%)")
        print(f"  Picks worst:    {picks_worst}/{n} ({picks_worst/n*100:.1f}%)")
        print(f"  t-test p-val:   {t_pval:.2e}")

    # Save
    out_path = os.path.join(os.path.dirname(args.results), 'difix_scores.json')
    with open(out_path, 'w') as f:
        json.dump({'difix_delta': difix_scores, 'difix_lpips_input': difix_lpips_input}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()

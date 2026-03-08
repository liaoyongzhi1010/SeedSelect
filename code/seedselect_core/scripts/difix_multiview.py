#!/usr/bin/env python3
"""Multi-view Difix3D+ scoring.

Hypothesis: Single front-view scoring is noisy. Averaging Difix3D+ delta
across 6 viewpoints gives a more reliable quality signal.

Phase 1: Render 6 views per mesh (EGL only, no CUDA)
Phase 2: Apply Difix3D+ to all views, average delta (CUDA)

Usage:
    # Phase 1 (can run while GPU is busy):
    PYOPENGL_PLATFORM=egl .venv/bin/python scripts/difix_multiview.py \
        --results outputs/multiseed/gso_full/results.json \
        --mesh_dir outputs/multiseed/gso_full \
        --input_dir outputs/inputs/gso_eval \
        --phase 1

    # Phase 2 (needs GPU):
    PYOPENGL_PLATFORM=egl /root/miniconda3/envs/difix3d/bin/python scripts/difix_multiview.py \
        --results outputs/multiseed/gso_full/results.json \
        --mesh_dir outputs/multiseed/gso_full \
        --input_dir outputs/inputs/gso_eval \
        --phase 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# 6 canonical viewpoints: (name, eye_position, up_vector)
# top view needs up=(0,0,-1) since default up=(0,1,0) is parallel to view direction
VIEWS = [
    ('front',       (0, 0, 2.0),       (0, 1, 0)),
    ('back',        (0, 0, -2.0),      (0, 1, 0)),
    ('left',        (-2.0, 0, 0),      (0, 1, 0)),
    ('right',       (2.0, 0, 0),       (0, 1, 0)),
    ('top',         (0, 2.0, 0),       (0, 0, -1)),
    ('front_right', (1.41, 0, 1.41),   (0, 1, 0)),
]


def render_all_views(mesh_path, size=512):
    """Render mesh from 6 viewpoints. Returns list of (PIL.Image, view_name)."""
    from src.utils.mesh import load_mesh, normalize_mesh
    from src.utils.camera import look_at
    from src.utils.render import render_mesh
    from PIL import Image

    mesh = load_mesh(mesh_path)
    mesh = normalize_mesh(mesh, mode='unit_cube')

    results = []
    for name, eye, up in VIEWS:
        c2w = look_at(eye, up=up)
        color, _ = render_mesh(mesh, c2w, width=size, height=size, yfov=np.deg2rad(50.0))
        results.append((Image.fromarray(color), name))
    return results


def phase1_render(args, results, obj_ids, seeds):
    """Phase 1: Pre-render all views (CPU/EGL only)."""
    print("\n--- Phase 1: Multi-view rendering ---")
    render_dir = os.path.join(os.path.dirname(args.results), 'renders_mv')
    os.makedirs(render_dir, exist_ok=True)

    t0 = time.time()
    rendered_count = 0
    skipped = 0

    for i, obj_id in enumerate(obj_ids):
        seed_data = results[obj_id].get('seeds', {})
        if len(seed_data) < 2:
            continue

        for s in seeds:
            mp = os.path.join(args.mesh_dir, obj_id, f'seed{s}',
                              'instant-mesh-large', 'meshes', f'{obj_id}.obj')
            out_dir = os.path.join(render_dir, obj_id)
            os.makedirs(out_dir, exist_ok=True)

            # Check if all views already rendered
            all_exist = all(
                os.path.exists(os.path.join(out_dir, f'seed{s}_{vname}.png'))
                for vname, _, _ in VIEWS
            )
            if all_exist:
                skipped += 1
                continue

            if not os.path.exists(mp):
                continue

            try:
                views = render_all_views(mp, size=args.render_size)
                for img, vname in views:
                    img.save(os.path.join(out_dir, f'seed{s}_{vname}.png'))
                rendered_count += 1
            except Exception as e:
                print(f'  [WARN] Render failed for {obj_id} seed={s}: {e}')

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(obj_ids) - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{len(obj_ids)}] {rendered_count} new renders, {skipped} skipped, {elapsed:.0f}s, ETA {eta:.0f}s')

    print(f"  Rendered {rendered_count} meshes ({len(VIEWS)} views each), skipped {skipped}, in {time.time()-t0:.0f}s")
    print(f"  Saved to {render_dir}")
    return render_dir


def phase2_score(args, results, obj_ids, seeds, render_dir):
    """Phase 2: Apply Difix3D+ and compute multi-view deltas."""
    import torch
    from PIL import Image

    sys.path.insert(0, os.path.join(REPO, 'third_party', 'Difix3D', 'src'))
    from pipeline_difix import DifixPipeline
    import lpips as lp

    print("\n--- Phase 2: Multi-view Difix3D+ scoring ---")
    print("  Loading Difix3D+ pipeline...")
    pipe = DifixPipeline.from_pretrained('nvidia/difix', trust_remote_code=True)
    pipe.to('cuda')
    print("  Loading LPIPS...")
    lpips_fn = lp.LPIPS(net='alex').to('cuda').eval()

    # Scoring methods:
    # 1. difix_mv_mean: mean delta across all views
    # 2. difix_mv_max: max delta (worst view)
    # 3. difix_mv_front_weighted: 2x weight on front view
    # 4. difix_front: single front view (baseline comparison)

    all_scores = {
        'difix_mv_mean': {},
        'difix_mv_max': {},
        'difix_mv_front_weighted': {},
        'difix_front': {},
        'difix_mv_lpips_input': {},  # multi-view LPIPS to input after Difix3D+
    }
    t1 = time.time()

    for i, obj_id in enumerate(obj_ids):
        seed_data = results[obj_id].get('seeds', {})
        if len(seed_data) < 2:
            continue

        input_img = os.path.join(args.input_dir, f'{obj_id}.png')
        input_pil = Image.open(input_img).convert('RGB') if os.path.exists(input_img) else None

        obj_mv_mean = {}
        obj_mv_max = {}
        obj_mv_fw = {}
        obj_front = {}
        obj_mv_lpips_input = {}

        for s in seeds:
            view_deltas = {}

            for vname, _, _ in VIEWS:
                render_path = os.path.join(render_dir, obj_id, f'seed{s}_{vname}.png')
                if not os.path.exists(render_path):
                    continue

                try:
                    rendered = Image.open(render_path).convert('RGB')

                    with torch.no_grad():
                        output = pipe('remove degradation', image=rendered,
                                      num_inference_steps=1, timesteps=[199],
                                      guidance_scale=0.0)
                    fixed = output.images[0]

                    if args.save_images:
                        fixed.save(os.path.join(render_dir, obj_id, f'seed{s}_{vname}_difix.png'))

                    size = 256
                    r_arr = np.array(rendered.resize((size, size))).astype(np.float32) / 255.0
                    f_arr = np.array(fixed.resize((size, size))).astype(np.float32) / 255.0
                    r_t = torch.from_numpy(r_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    f_t = torch.from_numpy(f_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    r_t, f_t = r_t.cuda(), f_t.cuda()

                    with torch.no_grad():
                        delta = lpips_fn(r_t, f_t).item()

                    view_deltas[vname] = delta

                    # For front view, also compute LPIPS to input
                    if vname == 'front' and input_pil is not None:
                        i_arr = np.array(input_pil.resize((size, size))).astype(np.float32) / 255.0
                        i_t = torch.from_numpy(i_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                        i_t = i_t.cuda()
                        with torch.no_grad():
                            lpips_to_input = lpips_fn(f_t, i_t).item()
                        obj_mv_lpips_input[s] = -lpips_to_input

                except Exception as e:
                    print(f'  [WARN] Difix3D+ failed for {obj_id} seed={s} {vname}: {e}')

            if not view_deltas:
                continue

            deltas = list(view_deltas.values())
            obj_mv_mean[s] = -np.mean(deltas)
            obj_mv_max[s] = -np.max(deltas)

            # Front-weighted: front gets 2x weight
            if 'front' in view_deltas:
                weighted = [view_deltas['front']] * 2 + [d for v, d in view_deltas.items() if v != 'front']
                obj_mv_fw[s] = -np.mean(weighted)

            if 'front' in view_deltas:
                obj_front[s] = -view_deltas['front']

        for method, obj_scores in [('difix_mv_mean', obj_mv_mean),
                                    ('difix_mv_max', obj_mv_max),
                                    ('difix_mv_front_weighted', obj_mv_fw),
                                    ('difix_front', obj_front),
                                    ('difix_mv_lpips_input', obj_mv_lpips_input)]:
            if obj_scores:
                all_scores[method][obj_id] = obj_scores

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(obj_ids) - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{len(obj_ids)}] {elapsed:.0f}s, ETA {eta:.0f}s')

    # Phase 3: Analyze
    print(f"\n{'='*70}")
    print(f"MULTI-VIEW DIFIX3D+ SCORING — {len(obj_ids)} objects")
    print(f"{'='*70}")

    from scipy import stats

    summary = {}
    print(f"\n{'Method':<25} {'CD↓':>8} {'Improv%':>8} {'Gap%':>6} {'Win%':>6} {'Oracle%':>8} {'Worst%':>8} {'p-val':>10}")
    print("-" * 85)

    for method_name in ['difix_front', 'difix_mv_mean', 'difix_mv_max',
                         'difix_mv_front_weighted', 'difix_mv_lpips_input']:
        method_scores = all_scores[method_name]
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
        w_stat, w_pval = stats.wilcoxon(default_cds, selected_cds)

        print(f"{method_name:<25} {selected_cds.mean():8.6f} {improvement:+7.2f}% {gap_closed:5.1f}% "
              f"{wins/n*100:5.1f}% {picks_oracle/n*100:7.1f}% {picks_worst/n*100:7.1f}% {t_pval:10.2e}")

        summary[method_name] = {
            'n': n,
            'default_cd': float(default_cds.mean()),
            'selected_cd': float(selected_cds.mean()),
            'oracle_cd': float(oracle_cds.mean()),
            'improvement_pct': float(improvement),
            'gap_closed_pct': float(gap_closed),
            'wins': wins, 'ties': ties, 'losses': losses,
            'win_rate': float(wins / n),
            'oracle_match_rate': float(picks_oracle / n),
            'worst_pick_rate': float(picks_worst / n),
            'ttest_pval': float(t_pval),
            'wilcoxon_pval': float(w_pval),
        }

    # Save
    out_path = os.path.join(os.path.dirname(args.results), 'difix_multiview_scores.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'scores': all_scores}, f, indent=2)
    print(f"\nSaved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--mesh_dir', required=True)
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--max_objects', type=int, default=0)
    ap.add_argument('--phase', type=int, choices=[1, 2], required=True)
    ap.add_argument('--render_size', type=int, default=512)
    ap.add_argument('--save_images', action='store_true')
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    obj_ids = list(results.keys())
    if args.max_objects > 0:
        obj_ids = obj_ids[:args.max_objects]

    seeds = sorted(results[obj_ids[0]].get('seeds', {}).keys(), key=int)
    print(f"Seeds: {seeds}, Objects: {len(obj_ids)}, Views: {len(VIEWS)}")

    render_dir = os.path.join(os.path.dirname(args.results), 'renders_mv')

    if args.phase == 1:
        phase1_render(args, results, obj_ids, seeds)
    elif args.phase == 2:
        phase2_score(args, results, obj_ids, seeds, render_dir)


if __name__ == '__main__':
    main()

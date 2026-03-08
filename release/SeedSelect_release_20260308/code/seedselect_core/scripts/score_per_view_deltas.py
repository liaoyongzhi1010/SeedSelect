#!/usr/bin/env python3
"""Extract per-view LPIPS deltas for learned proxy (Task B1).

For each object × seed, compute 6D feature vector of per-view LPIPS deltas
from Difix3D+. This data feeds the learned proxy MLP (Task B2).

Uses existing multi-view renders from renders_mv/.
Output: per_view_deltas.json with per-view LPIPS for each candidate.

Usage:
    PYOPENGL_PLATFORM=egl /root/miniconda3/envs/difix3d/bin/python \
        scripts/score_per_view_deltas.py
"""
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULTS_PATH = REPO / "outputs/multiseed/gso_full/results.json"
RENDER_DIR = REPO / "outputs/multiseed/gso_full/renders_mv"
OUTPUT_PATH = REPO / "outputs/multiseed/gso_full/per_view_deltas.json"

VIEWS = ['front', 'back', 'left', 'right', 'top', 'front_right']

print("=" * 70)
print("Per-View LPIPS Delta Extraction (Task B1)")
print("=" * 70)

# Load results
with open(RESULTS_PATH) as f:
    results = json.load(f)

obj_ids = list(results.keys())
print(f"Objects: {len(obj_ids)}, Views: {len(VIEWS)}")

# Check renders exist
rendered_count = 0
for obj_id in obj_ids:
    seeds = list(results[obj_id].get('seeds', {}).keys())
    obj_dir = RENDER_DIR / obj_id
    if obj_dir.exists():
        s = seeds[0]
        if all((obj_dir / f"seed{s}_{v}.png").exists() for v in VIEWS):
            rendered_count += 1

print(f"Objects with renders: {rendered_count}/{len(obj_ids)}")

# Load models
import torch
from PIL import Image

sys.path.insert(0, str(REPO / 'third_party' / 'Difix3D' / 'src'))
from pipeline_difix import DifixPipeline
import lpips as lp

print("Loading Difix3D+ pipeline...")
pipe = DifixPipeline.from_pretrained('nvidia/difix', trust_remote_code=True)
pipe.to('cuda')
print("Loading LPIPS...")
lpips_fn = lp.LPIPS(net='alex').to('cuda').eval()
print("Models loaded. Starting scoring...\n")

all_deltas = {}
t0 = time.time()
processed = 0

for i, obj_id in enumerate(obj_ids):
    seed_data = results[obj_id].get('seeds', {})
    seeds = sorted(seed_data.keys(), key=int)
    obj_dir = RENDER_DIR / obj_id

    if not obj_dir.exists():
        continue

    obj_deltas = {}

    for s in seeds:
        view_deltas = {}
        all_ok = True

        for vname in VIEWS:
            render_path = obj_dir / f"seed{s}_{vname}.png"
            if not render_path.exists():
                all_ok = False
                break

            try:
                rendered = Image.open(str(render_path)).convert('RGB')

                with torch.no_grad():
                    output = pipe('remove degradation', image=rendered,
                                  num_inference_steps=1, timesteps=[199],
                                  guidance_scale=0.0)
                fixed = output.images[0]

                size = 256
                r_arr = np.array(rendered.resize((size, size))).astype(np.float32) / 255.0
                f_arr = np.array(fixed.resize((size, size))).astype(np.float32) / 255.0
                r_t = torch.from_numpy(r_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                f_t = torch.from_numpy(f_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                r_t, f_t = r_t.cuda(), f_t.cuda()

                with torch.no_grad():
                    delta = lpips_fn(r_t, f_t).item()
                view_deltas[vname] = delta

            except Exception as e:
                print(f'  [WARN] Failed {obj_id} seed={s} {vname}: {e}')
                all_ok = False
                break

        if all_ok and len(view_deltas) == len(VIEWS):
            obj_deltas[s] = view_deltas

    if obj_deltas:
        all_deltas[obj_id] = obj_deltas
        processed += 1

    if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(obj_ids):
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(obj_ids) - i - 1) / rate if rate > 0 else 0
        print(f'  [{i+1}/{len(obj_ids)}] processed={processed} '
              f'{elapsed:.0f}s ETA={eta:.0f}s ({rate:.1f} obj/s)')

elapsed_total = time.time() - t0
print(f"\nDone! Processed {processed} objects in {elapsed_total:.0f}s")

# Verify data shape
sample_obj = list(all_deltas.keys())[0]
sample_seed = list(all_deltas[sample_obj].keys())[0]
print(f"Sample: {sample_obj} seed={sample_seed}")
print(f"  Views: {list(all_deltas[sample_obj][sample_seed].keys())}")
print(f"  Deltas: {[f'{v:.4f}' for v in all_deltas[sample_obj][sample_seed].values()]}")

# Save
with open(str(OUTPUT_PATH), 'w') as f:
    json.dump(all_deltas, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
print(f"Total entries: {sum(len(v) for v in all_deltas.values())} seed-candidates across {len(all_deltas)} objects")

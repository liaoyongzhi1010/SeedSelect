# SeedSelect Release Bundle (2026-03-08)

This folder packages code, finalized experiment artifacts, and runnable examples for external sharing.

## Structure
- `code/seedselect_core/`: current SeedSelect code snapshot (configs/scripts/src/requirements)
- `code/depthrefine3d_eval_scripts/`: K=8 completion and scoring scripts used in this cycle
- `code/instantmesh_patch/`: patched InstantMesh `run.py` and config used in production runs
- `results/main/`: main frozen JSON/MD results used by paper tables
- `results/hybrid/`: hybrid-training and experiment artifacts
- `results/wonder3d_canonical/`: canonical Wonder3D protocol + frozen eval snapshot
- `examples/gso_k8_examples/`: 3 curated object examples with input/seed meshes/renders/summary
- `docs/`: experiment status, go/no-go, reproducibility, number provenance
- `paper/`: latest compiled paper and experiments section tex

## Notes
- Example summaries are generated from:
  - `DepthRefine3D/outputs/multiseed/gso_full_k8/results.json`
  - `DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json`
- Selected seed is obtained by max Difix score per object.

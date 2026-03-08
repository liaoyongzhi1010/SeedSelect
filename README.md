# SeedSelect: Exploiting the Seed Lottery in Single-Image 3D Reconstruction

Official release bundle for SeedSelect.

SeedSelect is a training-free method that improves single-image 3D reconstruction by sampling multiple random seeds and selecting the best candidate using a multi-view Difix-based proxy score.

## Highlights

- Training-free improvement over default seed selection.
- Verified on GSO and OmniObject3D settings.
- Includes canonical Wonder3D protocol snapshot.
- Includes curated runnable examples with seed-level meshes and renders.
- Includes provenance and reproducibility docs used for submission hardening.

## Main Results (Current Frozen Snapshot)

From `results/main/gso_full_k8_difix_multiview_scores_full.json`:

- Dataset: GSO-300, K=8
- Default CD: 0.3648
- Selected CD: 0.3566
- Improvement: +2.25%
- Oracle gap closed: 27.16%
- Wilcoxon p-value: 9.62e-4

From `results/main/pairwise_ranking_results_full.json`:

- Pairwise win rate (SeedSelect vs PSNR): 55.10% vs 50.48%

## Repository Layout

- `code/seedselect_core/`: core SeedSelect code snapshot (`configs/`, `scripts/`, `src/`, `requirements.txt`)
- `code/depthrefine3d_eval_scripts/`: scripts used to complete/recompute/score K=8 experiments
- `code/instantmesh_patch/`: InstantMesh `run.py` patch and model config used in final runs
- `results/main/`: frozen JSON/MD artifacts for main tables and stats
- `results/hybrid/`: hybrid training/evaluation artifacts
- `results/wonder3d_canonical/`: canonical Wonder3D protocol + frozen mc sensitivity files
- `examples/gso_k8_examples/`: 3 object-level examples with `input`, per-seed `mesh/render`, and summaries
- `docs/`: experiment status, go/no-go, repro commands, and number provenance

## Quick Start

### 1) Environment

```bash
cd code/seedselect_core
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run core pipeline (example)

```bash
cd code/seedselect_core
python scripts/run_seedselect.py \
  --image /path/to/input.png \
  --backbone instantmesh \
  --k 4 \
  --output_dir /tmp/seedselect_demo
```

### 3) Inspect frozen results

```bash
cat results/main/gso_full_k8_difix_multiview_scores_full.json
cat results/main/pairwise_ranking_results_full.json
```

## Reproducibility

- Command references: `docs/REPRO_COMMANDS_ZH.md`
- Claim-to-evidence mapping: `docs/paper_number_alignment.md`
- Table provenance: `docs/table_number_provenance.md`
- Artifact checksums: `docs/artifacts_manifest.json`

## Examples

Three curated examples are included:

- `examples/gso_k8_examples/2_of_Jenga_Classic_Game/`
- `examples/gso_k8_examples/OXO_Soft_Works_Can_Opener_SnapLock/`
- `examples/gso_k8_examples/Mens_Billfish_Slip_On_in_Coffee_e8bPKE9Lfgo/`

Each example contains:

- `input.png`
- `seeds_mesh/seed*.obj`
- `seeds_render/seed*.png`
- `example_summary.json`

## Citation

```bibtex
@inproceedings{seedselect2026,
  title={SeedSelect: Exploiting the Seed Lottery in Diffusion-Based Single-Image 3D Reconstruction},
  author={Anonymous},
  year={2026}
}
```

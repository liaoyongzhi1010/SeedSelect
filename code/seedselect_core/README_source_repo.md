# SeedSelect: Exploiting the Seed Lottery in Diffusion-Based 3D Reconstruction

**Training-free quality improvement for single-image 3D reconstruction via multi-seed selection.**

> Given a single input image, SeedSelect generates K candidate 3D reconstructions using different random seeds, scores each via a multi-view quality proxy based on [Difix3D+](https://github.com/NVIDIA/Difix3D), and selects the best one. No training required.

## Key Results

| Backbone | Metric | Improvement | Gap Closed | p-value | n |
|----------|--------|-------------|------------|---------|---|
| InstantMesh | CD | +1.5% | 22.1% | 0.014 | 300 |
| LGM | LPIPS | +0.6% | 28.4% | 9e-5 | 300 |
| Wonder3D | Proxy | +3.3% | --- | <1e-6 | 50 |

## How It Works

1. **Generate** K candidate meshes from the same input image using different random seeds
2. **Render** each candidate from 6 canonical viewpoints
3. **Score** each rendering with Difix3D+ (1-step diffusion) and measure LPIPS between original and refined views
4. **Select** the candidate with the smallest average refinement gap (less refinement needed = better quality)

The insight: *a 3D reconstruction that requires less 2D image refinement across its rendered views is geometrically more accurate.*

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support (tested on RTX A6000 48GB)
- Conda or Mamba

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/SeedSelect.git
cd SeedSelect

# Create environment
conda create -n seedselect python=3.10 -y
conda activate seedselect

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install Difix3D+ (required for scoring)
pip install diffusers transformers accelerate

# Install backbone (InstantMesh)
git clone https://github.com/TencentARC/InstantMesh.git third_party/InstantMesh
cd third_party/InstantMesh && pip install -r requirements.txt && cd ../..
```

### EGL Rendering (headless servers)
For headless GPU rendering, ensure EGL drivers are available:
```bash
export PYOPENGL_PLATFORM=egl
python -c "import pyrender; r = pyrender.OffscreenRenderer(64, 64); r.delete(); print('EGL OK')"
```

## Quick Start

### Run SeedSelect on a single image

```bash
python scripts/run_seedselect.py \
    --image path/to/input.png \
    --backbone instantmesh \
    --k 4 \
    --output_dir results/
```

This will:
1. Generate 4 candidate meshes using seeds 0, 1, 2, 42
2. Score each candidate using multi-view Difix3D+
3. Save the best mesh to `results/best.obj`

### Run evaluation on GSO/OmniObject3D

```bash
# Step 1: Generate multi-seed candidates
python scripts/exp_full_multiseed.py \
    --dataset gso \
    --split configs/gso_eval.json \
    --seeds 0,1,2,42 \
    --out outputs/multiseed/gso

# Step 2: Render multi-view images (Phase 1, CPU/EGL)
PYOPENGL_PLATFORM=egl python scripts/difix_multiview.py \
    --results outputs/multiseed/gso/results.json \
    --mesh_dir outputs/multiseed/gso \
    --input_dir outputs/inputs/gso_eval \
    --phase 1

# Step 3: Score with Difix3D+ (Phase 2, GPU)
PYOPENGL_PLATFORM=egl python scripts/difix_multiview.py \
    --results outputs/multiseed/gso/results.json \
    --mesh_dir outputs/multiseed/gso \
    --input_dir outputs/inputs/gso_eval \
    --phase 2
```

## Project Structure

```
SeedSelect/
├── scripts/
│   ├── run_seedselect.py          # Single-image demo
│   ├── exp_full_multiseed.py      # Full evaluation pipeline
│   ├── difix_multiview.py         # Multi-view Difix3D+ scoring (core)
│   ├── analyze_multiseed.py       # Results analysis
│   ├── clip_scoring.py            # CLIP/DINOv2 baselines
│   ├── lgm_multiseed.py           # LGM backbone support
│   ├── score_lgm_seedselect.py    # LGM scoring
│   ├── wonder3d_multiseed.py      # Wonder3D backbone support
│   └── plot_*.py                  # Figure generation
├── src/
│   ├── config.py                  # Configuration loading
│   ├── utils/
│   │   ├── camera.py              # Camera utilities (look_at, orbit)
│   │   ├── mesh.py                # Mesh loading and normalization
│   │   ├── metrics.py             # PSNR, SSIM, LPIPS, Chamfer Distance
│   │   └── render.py              # Pyrender-based mesh rendering
│   └── data/
│       ├── gso.py                 # GSO dataset loader
│       └── omniobject3d.py        # OmniObject3D dataset loader
├── configs/
│   ├── gso_eval.json              # GSO-300 object IDs
│   └── omni_eval.json             # OmniObject3D-100 object IDs
├── third_party/                   # External repos (InstantMesh, Difix3D)
├── requirements.txt
└── README.md
```

## Supported Backbones

| Backbone | Architecture | 3D Representation | Stochastic | Status |
|----------|-------------|-------------------|------------|--------|
| [InstantMesh](https://github.com/TencentARC/InstantMesh) | Zero123++ + FlexiCubes | Mesh | Yes | Primary |
| [LGM](https://github.com/3DTopia/LGM) | ImageDream + 3DGS | Gaussian Splatting | Yes | Supported |
| [Wonder3D](https://github.com/xxlong0/Wonder3D) | Cross-domain MV Diffusion + NeuS | Implicit Surface | Yes | Supported |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | LRM Transformer | Mesh | No (deterministic) | Not applicable |

## Multi-View Scoring Details

The quality proxy uses 6 canonical viewpoints:

| View | Eye Position | Up Vector |
|------|-------------|-----------|
| Front | (0, 0, 2) | (0, 1, 0) |
| Back | (0, 0, -2) | (0, 1, 0) |
| Left | (-2, 0, 0) | (0, 1, 0) |
| Right | (2, 0, 0) | (0, 1, 0) |
| Top | (0, 2, 0) | (0, 0, -1) |
| Front-right | (1.41, 0, 1.41) | (0, 1, 0) |

Scoring uses Difix3D+ with 1 diffusion step (timestep=199, guidance_scale=0.0) to minimize computation while capturing the refinement signal.

## Citation

```bibtex
@inproceedings{seedselect2026,
  title={SeedSelect: Exploiting the Seed Lottery in Diffusion-Based Single-Image 3D Reconstruction},
  author={Anonymous},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Future Work

- **Learned scoring functions** trained on proxy signal or GT quality data for higher oracle-match rates
- Seed-conditional generation steering diffusion toward higher-quality regions
- Adaptive K scheduling based on estimated object difficulty
- Extension to text-to-3D and video-to-3D pipelines

## License

This project is released under the MIT License.

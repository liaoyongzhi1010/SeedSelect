# Table Number Provenance (Updated)

## 0. 说明
- 更新时间: 2026-03-07 16:52 UTC
- 目标: 建立“论文数字 -> 结果文件 -> 生成脚本 -> 一致性状态”的审计链路。
- 本版已纳入 `GSO-300 K=8` 补齐与 postchain 完成后的最终结果。

## 1. 一致性状态定义
- `CONSISTENT`: 论文数字与结果文件一致（允许四舍五入误差）。
- `PARTIAL`: 数字可追溯，但仍存在协议/工程口径未完全消歧。
- `CONFLICT`: 同一结论存在冲突结果版本，当前不可作为稳定证据。

## 2. 逐表溯源（投稿主线）

### 2.1 `tab:sensitivity`
- 论文位置: `/root/eccv/eccv2026/paper/sec/4_experiments.tex`
- 来源:
  - `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/comprehensive_analysis.json`
  - `/root/eccv/DepthRefine3D/outputs/multiseed/omni_full/difix_multiview_scores.json`
- 脚本:
  - `/root/eccv/DepthRefine3D/scripts/comprehensive_analysis.py`
  - `/root/eccv/DepthRefine3D/scripts/difix_multiview.py`
- 状态: `CONSISTENT`

### 2.2 `tab:main_gso`
- 来源:
  - `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/comprehensive_analysis.json`
  - `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/rescore_v2.json`
  - `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/clip_scoring_results.json`
- 脚本:
  - `/root/eccv/DepthRefine3D/scripts/comprehensive_analysis.py`
  - `/root/eccv/DepthRefine3D/scripts/rescore_v2.py`
  - `/root/eccv/DepthRefine3D/scripts/clip_scoring.py`
- 状态: `CONSISTENT`

### 2.3 `tab:cross_dataset`
- 来源:
  - `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full/comprehensive_analysis.json`
  - `/root/eccv/DepthRefine3D/outputs/multiseed/omni_full/difix_multiview_scores.json`
- 状态: `CONSISTENT`

### 2.4 `tab:pairwise`
- 来源:
  - `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json`
- 脚本:
  - `/root/eccv/DepthRefine3D/scripts/pairwise_ranking.py`
- 状态: `CONSISTENT`
- 备注: 已按最新 K=8 结果更新为 `SeedSelect 55.1`, `PSNR 50.5`。

### 2.5 `tab:hybrid_main`
- 来源:
  - `/root/eccv/eccv_rebuttal/results/hybrid/hybrid_experiment_results_full.json`
- 脚本:
  - `/root/eccv/eccv_rebuttal/scripts/10_run_hybrid_experiments.py`
  - `/root/eccv/eccv_rebuttal/scripts/11_export_hybrid_to_paper.py`
- 状态: `CONSISTENT`

### 2.6 `tab:hybrid_alpha`
- 来源:
  - `/root/eccv/eccv_rebuttal/results/hybrid/hybrid_experiment_results_full.json` (`hybrid_grid`)
- 状态: `CONSISTENT`

### 2.7 `tab:hybrid_backbone`
- 来源:
  - InstantMesh/LGM: `/root/eccv/eccv_rebuttal/results/hybrid/hybrid_cross_backbone_results.json`
  - Wonder3D canonical: `/root/eccv/eccv_rebuttal/results/hybrid/snapshots/wonder3d_20260307_084332Z/canonical_eval_v1/hybrid_wonder3d_pvalues_mc16_v1.json`
- 状态: `CONSISTENT`
- 备注: Wonder3D 已切换到 canonical 口径，不再引用历史冲突文件。

### 2.8 `tab:hybrid_efficiency`
- 来源:
  - `/root/eccv/eccv_rebuttal/results/hybrid/hybrid_experiment_results_full.json`
- wall-clock 补充:
  - `/root/eccv/eccv_rebuttal/results/hybrid/hybrid_runtime_wallclock.json`
  - `/root/eccv/eccv_rebuttal/results/hybrid/HYBRID_RUNTIME_WALLCLOCK.md`
- 状态: `PARTIAL`
- 风险: 主表仍使用归一化成本口径；wall-clock 目前为 selection-stage（不含候选生成）。

### 2.9 `tab:learned_naive`
- 来源:
  - `/root/eccv/eccv_rebuttal/results/learned_verifier/selection_model_comparison.json`
- 状态: `CONSISTENT`

### 2.10 `tab:wonder3d_gt`
- 来源:
  - `/root/eccv/eccv_rebuttal/results/hybrid/snapshots/wonder3d_20260307_084332Z/canonical_eval_v1/hybrid_wonder3d_results_mc16_v1.json`
  - `/root/eccv/eccv_rebuttal/results/hybrid/snapshots/wonder3d_20260307_084332Z/canonical_eval_v1/hybrid_wonder3d_pvalues_mc16_v1.json`
  - `/root/eccv/eccv_rebuttal/results/hybrid/wonder3d_canonical.json`
- 状态: `CONSISTENT`

### 2.11 `tab:scaling`
- 来源:
  - GSO-50 K=2/4/8: `/root/eccv/DepthRefine3D/outputs/multiseed/ablation_results.json`
  - GSO-50 K=16: `/root/eccv/DepthRefine3D/outputs/multiseed/gso_k8/results.json`
  - GSO-300 K=8: `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json`
  - Pairwise/辅助一致性: `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json`
- 状态: `CONSISTENT`
- 备注: GSO-300 K=8 已升级为真 K=8（`2400/2400`）。

### 2.12 `tab:ablation`
- 来源:
  - `/root/eccv/DepthRefine3D/outputs/multiseed/ablation_results.json`
- 状态: `PARTIAL`
- 风险: `N=2..5` 视角结果含插值推断，若主审稿意见卡在此，需补独立重算。

## 3. 当前结论
- `CONFLICT=0`（投稿主线）
- 已消除最高风险项: 伪 K=8 与 Wonder3D 口径冲突
- 仍建议投稿前补两点:
  1. hybrid efficiency 的 wall-clock 实测替换估算口径
  2. 对 `tab:ablation` 非独立重算项给出附录说明或补跑

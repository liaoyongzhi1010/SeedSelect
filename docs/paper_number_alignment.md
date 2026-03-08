# Paper Number Alignment Report (Updated)

- 生成时间: 2026-03-07 16:50 UTC
- 目的: 对齐论文关键数字与当前唯一结果文件（冻结后处理结果）。
- 说明: 本版重点刷新 `GSO-300 K=8` 与 postchain 完成后的数字。

## 总览

- 状态: `核心主张已对齐`
- 重点修复: 伪 K=8 -> 真 K=8（`2400/2400`, `{8:300}`）
- 当前唯一 K=8 结果来源:
  - `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/results.json`
  - `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json`
  - `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json`

## 关键数字对齐（投稿主线）

| Claim | Paper | Source | Status | Source File |
|---|---:|---:|---|---|
| GSO K8 Improvement (%) | 2.3 | 2.253 | MATCH (rounded) | `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json` |
| GSO K8 Gap Closed (%) | 27.2 | 27.156 | MATCH (rounded) | `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json` |
| GSO K8 Worst Pick (%) | 17.0 | 17.0 | MATCH | `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json` |
| GSO K8 Wilcoxon p | 9.6e-4 | 9.624e-4 | MATCH (rounded) | `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json` |
| Pairwise GSO K8 SeedSelect (%) | 55.1 | 55.1048 | MATCH (rounded) | `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json` |
| Pairwise GSO K8 PSNR (%) | 50.5 | 50.4828 | MATCH (rounded) | `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json` |
| Pairwise GSO K4 SeedSelect (%) | 55.3 | 55.2778 | MATCH (rounded) | `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json` |
| Pairwise Omni SeedSelect (%) | 60.2 | 60.2143 | MATCH (rounded) | `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json` |
| Wonder3D Hybrid Gap (%) | 2.01 | 2.006 | MATCH (rounded) | `/root/eccv/eccv_rebuttal/results/hybrid/snapshots/wonder3d_20260307_084332Z/canonical_eval_v1/hybrid_wonder3d_pvalues_mc16_v1.json` |
| Wonder3D Hybrid p | 0.568 | 0.568 | MATCH | `/root/eccv/eccv_rebuttal/results/hybrid/snapshots/wonder3d_20260307_084332Z/canonical_eval_v1/hybrid_wonder3d_pvalues_mc16_v1.json` |

## 本轮变更

1. 论文 `4_experiments.tex` 已更新以下字段：
   - pairwise(K=8): `55.2 -> 55.1`
   - pairwise PSNR(K=8): `50.7 -> 50.5`
   - scaling(GSO-300, K=8): `+2.2/26.9/15.3/9.3e-4 -> +2.3/27.2/17.0/9.6e-4`
2. K=8 postchain 已完成并写入最终结果，结束时间：`2026-03-07 16:36:24 UTC`。

## 投稿前剩余事项（最小集合）

1. `main.tex` 已编译通过（两次 `pdflatex`），当前仅有轻微 overfull/underfull 警告，无阻塞错误。
2. `artifacts_manifest.json` 已刷新到 `2026-03-07T17:06:55Z`，主线文件 checksum 已更新。
3. 若要进一步提高顶会抗审稿强度，建议补 wall-clock runtime 实测（替代 efficiency 估算口径）。

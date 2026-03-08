# 实验执行状态（2026-03-07，已完成主链路）

## 一、已完成（关键）

### 1) Wonder3D canonical 协议已冻结
- 快照：
  - `/root/eccv/eccv_rebuttal/results/hybrid/snapshots/wonder3d_20260307_084332Z`
- canonical 指针：
  - `/root/eccv/eccv_rebuttal/results/hybrid/wonder3d_canonical.json`
- canonical 结果（GSO-150, mc=16）：
  - `hybrid_wonder3d_results_mc16_v1.json`
  - `hybrid_wonder3d_pvalues_mc16_v1.json`

### 2) Wonder3D MC 敏感性补充已完成
- `mc=8/16/32` 全部已落盘（同快照目录下 `mc_sensitivity`）。
- 结论已在文稿披露：hybrid 在 Wonder3D 上对 MC 采样数敏感。

### 3) GSO-300 真 K=8 已补齐并重算
- 补齐链路：
  - 并发4主跑：`fill_k8_missing_meshes_fixed_p4_20260307_110937Z.log`
  - 并发1重试：`fill_k8_postchain_p4_then_p1_20260307_112217Z.log`
- 重试结果：`170/170` 全成功。
- 当前完整性：
  - `existing=2400, missing=0`
  - per-object seed 分布：`{8: 300}`
- 结论：此前“伪 K=8”硬伤已消除。

### 4) 后处理链路已跑完
- `recompute_gso_full_k8_results_from_meshes.py`：完成（updated=2400, failed_eval=0）。
- `score_k8_gso300_resume.py --resume`：完成并刷新 `difix_multiview_scores.json`。
- `pairwise_ranking.py`：完成并刷新 `pairwise_ranking_results.json`。
- 完成时间戳（UTC）：
  - postchain done: `2026-03-07T16:36:24Z`

### 5) Hybrid runtime wall-clock 补充已完成
- 文件：
  - `/root/eccv/eccv_rebuttal/results/hybrid/hybrid_runtime_wallclock.json`
  - `/root/eccv/eccv_rebuttal/results/hybrid/HYBRID_RUNTIME_WALLCLOCK.md`
- 口径：selection-stage（使用预计算候选描述，不含候选生成）
- 结果（median ms/object）：
  - GSO-300: zero-shot `0.0007`, learned `0.3233`, hybrid `2.4524`
  - Omni-100: zero-shot `0.0011`, learned `0.3333`, hybrid `2.4462`

## 二、最新核心数字（用于论文）

### GSO-300, K=8（`difix_multiview_scores.json`）
- `default_cd=0.3648`
- `selected_cd=0.3566`
- `oracle_cd=0.3346`
- `improvement=+2.253%`
- `gap_closed=27.156%`
- `worst_pick_rate=17.0%`
- `wilcoxon_p=9.624e-4`

### Pairwise（`pairwise_ranking_results.json`）
- GSO-300 K=8:
  - SeedSelect: `55.10%`
  - PSNR: `50.48%`

## 三、当前运行状态
- 无 GPU 训练/评测任务在跑（空闲）。
- 所有本轮计划链路已完成。

## 四、已同步动作
- 论文主文 `4_experiments.tex` 中 K=8 相关数字已按最新结果更新：
  - pairwise(K=8): `55.1`（PSNR `50.5`）
  - scaling(GSO-300, K=8): `+2.3`, `27.2`, `17.0`, `9.6e-4`

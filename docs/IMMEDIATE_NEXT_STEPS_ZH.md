# 立即执行清单（投稿冲刺版）

## 已完成（截至 2026-03-07 17:18 UTC）
1. Wonder3D canonical 与 MC 敏感性实验已完成并冻结。
2. GSO-300 真 K=8 已补齐到 `2400/2400`（`{8:300}`）。
3. K=8 后处理链路已跑完：`recompute -> score(resume) -> pairwise`。
4. 论文 `4_experiments.tex` 已同步最新 K=8 数字。
5. `paper_number_alignment.md` 与 `table_number_provenance.md` 已刷新为最新口径。
6. `artifacts_manifest.json` SHA256 已刷新。
7. 论文已完成一次编译自检（`pdflatex` 两次通过）。
8. 已生成投稿门禁结论：`/root/eccv/SUBMISSION_GO_NO_GO_ZH.md`。
9. 已生成复现实验命令清单：`/root/eccv/REPRO_COMMANDS_ZH.md`。
10. 已生成投稿交付包：
   - 目录：`/root/eccv/deliverables/submission_package_20260307_170757Z`
   - 压缩包：`/root/eccv/deliverables/submission_package_20260307_170757Z.tar.gz`
11. 已补 runtime wall-clock benchmark（selection-stage）：
   - `/root/eccv/eccv_rebuttal/results/hybrid/hybrid_runtime_wallclock.json`
   - `/root/eccv/eccv_rebuttal/results/hybrid/HYBRID_RUNTIME_WALLCLOCK.md`
12. 论文正文已加入 runtime 口径声明与实测数值补充。

## 立即要做（按优先级）
1. 若冲刺更强抗审稿，可补一组端到端 wall-clock（含候选生成）作为附录扩展。
2. 对 `tab:ablation` 中插值项补附录说明（或补独立重算）。
3. 对交付包做人工终审（导师/合作者）并冻结最终版本号。

## 当前策略
- 进入“收口模式”，不再开新大规模训练任务；
- 以“证据唯一、数字一致、可复现脚本可跑通”为第一目标。

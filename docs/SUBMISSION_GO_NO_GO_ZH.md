# 投稿门禁（Go/No-Go）

- 更新时间: 2026-03-07 17:18 UTC
- 目标会议级别: ECCV/CVPR/NeurIPS 主会

## 当前判断
- 结论: `Go (有条件)`
- 置信度: 中等偏高

## 已满足的硬条件
1. 主实验链路可闭环：`K=8 mesh补齐 -> 重算CD/FS -> Difix打分 -> pairwise`。
2. K=8 完整性满足：`2400/2400`，每对象 8 seeds。
3. Wonder3D 已锁定 canonical，避免旧口径冲突。
4. 论文关键数字已同步到最新结果，并成功编译。

## 仍存在的主要风险
1. 已补 selection-stage wall-clock，但尚未给出包含候选生成在内的端到端 wall-clock。
2. `ablation` 中部分视角点来自插值口径，可能被强审稿人追问。

## 投稿前最小补丁（建议）
1. 在附录补一段 runtime 口径声明（selection-stage 实测 + 端到端未覆盖范围）。
2. 在附录补一段 ablation 口径声明（哪些是独立重算，哪些是插值）。
3. 固化最终 artifacts 清单（SHA256 已刷新）并附复现命令。

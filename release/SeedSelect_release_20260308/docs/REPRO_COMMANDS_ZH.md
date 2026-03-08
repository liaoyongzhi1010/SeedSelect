# 复现实验命令清单（投稿包）

- 更新时间: 2026-03-07 17:08 UTC
- 目标: 从当前仓库复现论文主线关键数字（尤其 K=8 链路）。

## 1) GSO-300 K=8 补齐 mesh

```bash
/root/miniconda3/envs/difix3d/bin/python -u /root/eccv/DepthRefine3D/scripts/fill_gso_full_k8_missing_meshes.py \
  --split-json /root/eccv/DepthRefine3D/configs/gso_eval.json \
  --input-dir /root/eccv/DepthRefine3D/outputs/inputs/gso_eval \
  --out-dir /root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8 \
  --seeds 0,1,2,3,4,5,6,42 \
  --parallel 4 \
  --status-every 10 \
  --instantmesh-python /root/miniconda3/envs/instantmesh/bin/python
```

## 2) 从 mesh 重算 CD/FS（results.json）

```bash
/root/miniconda3/envs/difix3d/bin/python /root/eccv/DepthRefine3D/scripts/recompute_gso_full_k8_results_from_meshes.py
```

## 3) 增量 Difix 打分（支持 resume）

```bash
/root/miniconda3/envs/difix3d/bin/python /root/eccv/DepthRefine3D/scripts/score_k8_gso300_resume.py --resume
```

## 4) Pairwise 排序评估

```bash
/root/miniconda3/envs/difix3d/bin/python /root/eccv/DepthRefine3D/scripts/pairwise_ranking.py
```

## 5) 关键结果文件

- `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/results.json`
- `/root/eccv/DepthRefine3D/outputs/multiseed/gso_full_k8/difix_multiview_scores.json`
- `/root/eccv/DepthRefine3D/outputs/multiseed/pairwise_ranking_results.json`

## 6) 论文编译

```bash
cd /root/eccv/eccv2026/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

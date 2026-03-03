#!/usr/bin/env python3
"""Fit and validate a simple mechanism model: Delta_K = G_K * R_K.

Definitions (lower CD is better):
  Delta_K = E[CD_default - CD_selected(K)]
  G_K     = E[CD_default - CD_oracle(K)]
  R_K     = Delta_K / G_K

This script:
  1) computes observed curves across K
  2) fits G_K with a saturation law: G_inf * (1 - K^{-gamma})
  3) fits R(p) with two variants:
     - linear+clip: clip(a p + b, 0, 1)
     - logistic   : sigmoid(a p + b)
  4) predicts held-out K and reports fit quality
  5) exports JSON/CSV + a paper-ready validation figure
"""

import argparse
import csv
import json
import math
import os
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_k8/results.json"
DEFAULT_SCORES = "/root/eccv/DepthRefine3D/outputs/multiseed/gso_k8/difix_multiview_scores.json"
DEFAULT_SCORE_KEY = "scores.difix_mv_mean"
DEFAULT_OUT_DIR = "/root/eccv/.worktrees/seedselect80/code/outputs/mechanism"


def parse_args():
    parser = argparse.ArgumentParser(description="Fit/validate Delta_K = G_K * R_K")
    parser.add_argument("--results_json", default=DEFAULT_RESULTS)
    parser.add_argument("--scores_json", default=DEFAULT_SCORES)
    parser.add_argument("--score_key", default=DEFAULT_SCORE_KEY)
    parser.add_argument("--k_values", default="2,4,6,8,10,12,14,16")
    parser.add_argument("--fit_k", default="2,4,8")
    parser.add_argument("--heldout_k", default="6,10,12,14,16")
    parser.add_argument("--default_seed", default="42")
    parser.add_argument("--higher_is_better", action="store_true", default=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def nested_get(d: dict, key_path: str):
    cur = d
    for key in key_path.split("."):
        if key not in cur:
            raise KeyError(f"Missing key '{key}' in path '{key_path}'")
        cur = cur[key]
    return cur


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(np.mean(values[idx])))
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5])
    return float(lo), float(hi)


def pairwise_acc(gt_obj: Dict[str, float], sc_obj: Dict[str, float], seeds: Sequence[str], higher_is_better: bool) -> float:
    c = 0
    d = 0
    for si, sj in combinations(seeds, 2):
        # gt: lower CD is better
        gt_pref = gt_obj[si] < gt_obj[sj]
        if higher_is_better:
            pred_pref = sc_obj[si] > sc_obj[sj]
        else:
            pred_pref = sc_obj[si] < sc_obj[sj]
        if sc_obj[si] == sc_obj[sj] or gt_obj[si] == gt_obj[sj]:
            continue
        if pred_pref == gt_pref:
            c += 1
        else:
            d += 1
    if c + d == 0:
        return 0.5
    return c / (c + d)


def k_prefix(obj_seed_order: List[str], k: int, allowed: Sequence[str]) -> List[str]:
    allowed_set = set(allowed)
    out = []
    for s in obj_seed_order:
        if s in allowed_set:
            out.append(s)
        if len(out) >= k:
            break
    return out


def compute_observed(
    results_json: str,
    scores_json: str,
    score_key: str,
    k_values: Sequence[int],
    default_seed: str,
    higher_is_better: bool,
    bootstrap: int,
    seed: int,
) -> dict:
    results = load_json(results_json)
    scores_full = load_json(scores_json)
    scores = nested_get(scores_full, score_key)

    per_k = {k: {"delta": [], "gain": [], "ratio": [], "pairwise": [], "default_cd": [], "selected_cd": [], "oracle_cd": []} for k in k_values}
    n_objects_used = 0

    for obj, payload in results.items():
        if obj not in scores:
            continue
        gt_raw = payload.get("seeds", {})
        gt_obj = {str(s): float(v["cd"]) for s, v in gt_raw.items() if "cd" in v}
        sc_obj = {str(s): float(v) for s, v in scores[obj].items()}
        shared = [s for s in sc_obj.keys() if s in gt_obj]
        if len(shared) < max(k_values):
            # allow partial K: if object doesn't have max K, skip object entirely
            continue
        if default_seed not in gt_obj:
            continue

        seed_order = list(sc_obj.keys())
        n_objects_used += 1

        for k in k_values:
            if k == 1:
                cand = [default_seed]
            else:
                cand = k_prefix(seed_order, k, shared)
            if len(cand) < k:
                continue
            if higher_is_better:
                selected = max(cand, key=lambda s: sc_obj[s])
            else:
                selected = min(cand, key=lambda s: sc_obj[s])
            oracle = min(cand, key=lambda s: gt_obj[s])

            d_cd = gt_obj[default_seed]
            s_cd = gt_obj[selected]
            o_cd = gt_obj[oracle]
            delta = d_cd - s_cd
            gain = d_cd - o_cd
            ratio = delta / gain if gain > 1e-12 else 0.0
            ratio = float(np.clip(ratio, 0.0, 1.5))

            per_k[k]["delta"].append(delta)
            per_k[k]["gain"].append(gain)
            per_k[k]["ratio"].append(ratio)
            per_k[k]["pairwise"].append(pairwise_acc(gt_obj, sc_obj, cand, higher_is_better))
            per_k[k]["default_cd"].append(d_cd)
            per_k[k]["selected_cd"].append(s_cd)
            per_k[k]["oracle_cd"].append(o_cd)

    observed = {}
    for k in k_values:
        d = np.asarray(per_k[k]["delta"], dtype=np.float64)
        g = np.asarray(per_k[k]["gain"], dtype=np.float64)
        r = np.asarray(per_k[k]["ratio"], dtype=np.float64)
        p = np.asarray(per_k[k]["pairwise"], dtype=np.float64)
        d_cd = np.asarray(per_k[k]["default_cd"], dtype=np.float64)
        s_cd = np.asarray(per_k[k]["selected_cd"], dtype=np.float64)
        o_cd = np.asarray(per_k[k]["oracle_cd"], dtype=np.float64)

        delta_mean = float(np.mean(d))
        gain_mean = float(np.mean(g))
        ratio_mean = float(np.mean(r))
        pair_mean = float(np.mean(p))

        imp_pct = (float(np.mean(d_cd)) - float(np.mean(s_cd))) / max(float(np.mean(d_cd)), 1e-12) * 100.0
        gap_closed_pct = (float(np.mean(d_cd)) - float(np.mean(s_cd))) / max(float(np.mean(d_cd)) - float(np.mean(o_cd)), 1e-12) * 100.0

        observed[k] = {
            "n_objects": int(len(d)),
            "Delta_K": delta_mean,
            "G_K": gain_mean,
            "R_K": ratio_mean,
            "p_K": pair_mean,
            "improvement_pct": float(imp_pct),
            "gap_closed_pct": float(gap_closed_pct),
            "Delta_CI95": bootstrap_ci(d, bootstrap, seed + k),
            "G_CI95": bootstrap_ci(g, bootstrap, seed + 100 + k),
            "R_CI95": bootstrap_ci(r, bootstrap, seed + 200 + k),
            "p_CI95": bootstrap_ci(p, bootstrap, seed + 300 + k),
        }

    return {"observed": observed, "n_objects_used": n_objects_used}


def g_model(k: np.ndarray, g_inf: float, gamma: float) -> np.ndarray:
    return g_inf * (1.0 - np.power(k, -gamma))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_models(observed: dict, fit_k: Sequence[int]) -> dict:
    ks = np.asarray(fit_k, dtype=np.float64)
    g_vals = np.asarray([observed[k]["G_K"] for k in fit_k], dtype=np.float64)
    r_vals = np.asarray([observed[k]["R_K"] for k in fit_k], dtype=np.float64)
    p_vals = np.asarray([observed[k]["p_K"] for k in fit_k], dtype=np.float64)

    # 1) Fit G(K).
    g0 = max(float(np.max(g_vals)), 1e-6)
    try:
        popt_g, _ = curve_fit(
            g_model,
            ks,
            g_vals,
            p0=[g0 * 1.3, 0.8],
            bounds=([1e-9, 1e-4], [1.0, 5.0]),
            maxfev=20000,
        )
        g_inf, gamma = float(popt_g[0]), float(popt_g[1])
    except Exception:
        # Fallback
        g_inf, gamma = g0, 0.7

    # 2a) Fit R(p): linear + clip.
    X = np.stack([p_vals, np.ones_like(p_vals)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(X, r_vals, rcond=None)
    a_lin, b_lin = float(coeff[0]), float(coeff[1])

    # 2b) Fit R(p): logistic.
    def logistic_p(p, a, b):
        return 1.0 / (1.0 + np.exp(-(a * p + b)))

    try:
        popt_r, _ = curve_fit(logistic_p, p_vals, np.clip(r_vals, 1e-4, 1 - 1e-4), p0=[8.0, -4.0], maxfev=20000)
        a_log, b_log = float(popt_r[0]), float(popt_r[1])
    except Exception:
        a_log, b_log = 8.0, -4.0

    return {
        "G_model": {"name": "power_saturation", "g_inf": g_inf, "gamma": gamma},
        "R_model_linear_clip": {"name": "linear_clip", "a": a_lin, "b": b_lin},
        "R_model_logistic": {"name": "logistic", "a": a_log, "b": b_log},
    }


def predict_delta(k: int, p_k: float, params: dict, model_name: str) -> float:
    g_inf = params["G_model"]["g_inf"]
    gamma = params["G_model"]["gamma"]
    g = g_model(np.asarray([k], dtype=np.float64), g_inf, gamma)[0]

    if model_name == "linear_clip":
        a = params["R_model_linear_clip"]["a"]
        b = params["R_model_linear_clip"]["b"]
        r = float(np.clip(a * p_k + b, 0.0, 1.0))
    elif model_name == "logistic":
        a = params["R_model_logistic"]["a"]
        b = params["R_model_logistic"]["b"]
        r = float(sigmoid(np.asarray([a * p_k + b]))[0])
    else:
        raise ValueError(model_name)
    return float(g * r)


def eval_predictions(observed: dict, ks: Sequence[int], pred: Dict[int, float]) -> dict:
    y = np.asarray([observed[k]["Delta_K"] for k in ks], dtype=np.float64)
    yhat = np.asarray([pred[k] for k in ks], dtype=np.float64)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / max(sst, 1e-12)
    mape = float(np.mean(np.abs((yhat - y) / np.maximum(np.abs(y), 1e-9))) * 100.0)
    rel = [float((yhat_i - y_i) / max(abs(y_i), 1e-9) * 100.0) for y_i, yhat_i in zip(y, yhat)]
    return {"R2": r2, "MAPE_pct": mape, "relative_error_pct": rel}


def save_csv(path: str, rows: List[dict], fieldnames: Sequence[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_validation(path: str, observed: dict, all_k: Sequence[int], fit_k: Sequence[int], heldout_k: Sequence[int], pred_lin: Dict[int, float], pred_log: Dict[int, float], params: dict):
    ks = np.asarray(all_k, dtype=np.float64)
    g_obs = np.asarray([observed[k]["G_K"] for k in all_k], dtype=np.float64)
    d_obs = np.asarray([observed[k]["Delta_K"] for k in all_k], dtype=np.float64)
    p_obs = np.asarray([observed[k]["p_K"] for k in all_k], dtype=np.float64)

    k_dense = np.linspace(min(all_k), max(all_k), 200)
    g_fit = g_model(k_dense, params["G_model"]["g_inf"], params["G_model"]["gamma"])

    plt.figure(figsize=(10.5, 4.0))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(all_k, g_obs, "s-", color="#1f77b4", label="Observed $G_K$")
    ax1.plot(k_dense, g_fit, "--", color="#1f77b4", alpha=0.8, label="Fitted $G_K$")
    ax1.plot(all_k, d_obs, "o-", color="#2ca02c", label="Observed $\\Delta_K$")
    ax1.plot(all_k, [pred_lin[k] for k in all_k], "^-", color="#d62728", label="Pred $\\Delta_K$ (linear+clip)")
    ax1.plot(all_k, [pred_log[k] for k in all_k], "v--", color="#9467bd", label="Pred $\\Delta_K$ (logistic)")
    for k in heldout_k:
        ax1.axvline(k, color="#aaaaaa", linestyle=":", linewidth=0.9)
    ax1.set_xlabel("K")
    ax1.set_ylabel("Absolute CD gain")
    ax1.set_title("Mechanism Fit and Held-out Prediction")
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    y = np.asarray([observed[k]["Delta_K"] for k in heldout_k], dtype=np.float64)
    y_lin = np.asarray([pred_lin[k] for k in heldout_k], dtype=np.float64)
    y_log = np.asarray([pred_log[k] for k in heldout_k], dtype=np.float64)
    mn = float(min(np.min(y), np.min(y_lin), np.min(y_log)))
    mx = float(max(np.max(y), np.max(y_lin), np.max(y_log)))
    pad = 0.05 * max(mx - mn, 1e-6)
    ax2.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth=1)
    ax2.scatter(y, y_lin, marker="^", color="#d62728", label="linear+clip")
    ax2.scatter(y, y_log, marker="v", color="#9467bd", label="logistic")
    for i, k in enumerate(heldout_k):
        ax2.annotate(f"K={k}", (y[i], y_lin[i]), textcoords="offset points", xytext=(5, 3), fontsize=8, color="#555555")
    ax2.set_xlabel("Measured $\\Delta_K$")
    ax2.set_ylabel("Predicted $\\Delta_K$")
    ax2.set_title("Held-out K Prediction")
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    k_values = parse_int_list(args.k_values)
    fit_k = parse_int_list(args.fit_k)
    heldout_k = parse_int_list(args.heldout_k)

    observed_pack = compute_observed(
        results_json=args.results_json,
        scores_json=args.scores_json,
        score_key=args.score_key,
        k_values=k_values,
        default_seed=args.default_seed,
        higher_is_better=args.higher_is_better,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    observed = observed_pack["observed"]
    params = fit_models(observed, fit_k)

    pred_lin = {k: predict_delta(k, observed[k]["p_K"], params, "linear_clip") for k in k_values}
    pred_log = {k: predict_delta(k, observed[k]["p_K"], params, "logistic") for k in k_values}

    held_lin_eval = eval_predictions(observed, heldout_k, pred_lin)
    held_log_eval = eval_predictions(observed, heldout_k, pred_log)

    rows = []
    for k in k_values:
        row = {
            "K": k,
            "n_objects": observed[k]["n_objects"],
            "Delta_K": observed[k]["Delta_K"],
            "G_K": observed[k]["G_K"],
            "R_K": observed[k]["R_K"],
            "p_K": observed[k]["p_K"],
            "improvement_pct": observed[k]["improvement_pct"],
            "gap_closed_pct": observed[k]["gap_closed_pct"],
            "Delta_pred_linear": pred_lin[k],
            "Delta_pred_logistic": pred_log[k],
            "is_fit_k": int(k in fit_k),
            "is_heldout_k": int(k in heldout_k),
        }
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "k_model_observed_pred.csv")
    save_csv(
        csv_path,
        rows,
        fieldnames=[
            "K",
            "n_objects",
            "Delta_K",
            "G_K",
            "R_K",
            "p_K",
            "improvement_pct",
            "gap_closed_pct",
            "Delta_pred_linear",
            "Delta_pred_logistic",
            "is_fit_k",
            "is_heldout_k",
        ],
    )

    fig_path = os.path.join(args.out_dir, "k_model_validation.pdf")
    plot_validation(fig_path, observed, k_values, fit_k, heldout_k, pred_lin, pred_log, params)

    out_json = os.path.join(args.out_dir, "k_model_fit_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "inputs": {
                    "results_json": args.results_json,
                    "scores_json": args.scores_json,
                    "score_key": args.score_key,
                    "k_values": k_values,
                    "fit_k": fit_k,
                    "heldout_k": heldout_k,
                    "default_seed": args.default_seed,
                    "n_objects_used": observed_pack["n_objects_used"],
                },
                "observed": observed,
                "params": params,
                "heldout_eval": {
                    "linear_clip": held_lin_eval,
                    "logistic": held_log_eval,
                },
                "predictions": {
                    "linear_clip": pred_lin,
                    "logistic": pred_log,
                },
                "artifacts": {
                    "csv": csv_path,
                    "figure": fig_path,
                },
            },
            f,
            indent=2,
        )

    print(f"Saved {out_json}")
    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}")
    print("Held-out K metrics:")
    print(f"  linear+clip: R2={held_lin_eval['R2']:.3f}, MAPE={held_lin_eval['MAPE_pct']:.2f}%")
    print(f"  logistic   : R2={held_log_eval['R2']:.3f}, MAPE={held_log_eval['MAPE_pct']:.2f}%")


if __name__ == "__main__":
    main()

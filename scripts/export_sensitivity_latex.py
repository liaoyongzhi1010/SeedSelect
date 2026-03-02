#!/usr/bin/env python3
"""Export sensitivity-analysis JSON to a LaTeX table snippet."""

import argparse
import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_JSON = os.path.join(PROJECT_DIR, "outputs", "sensitivity", "refiner_metric_sensitivity.json")
DEFAULT_TEX = os.path.join(PROJECT_DIR, "outputs", "sensitivity", "refiner_metric_sensitivity_table.tex")


def fmt_p(p):
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Export sensitivity table as LaTeX")
    parser.add_argument("--input_json", default=DEFAULT_JSON)
    parser.add_argument("--output_tex", default=DEFAULT_TEX)
    parser.add_argument("--max_rows", type=int, default=7)
    args = parser.parse_args()

    with open(args.input_json) as f:
        rows = json.load(f)["results"]

    rows = rows[: args.max_rows]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Refiner--metric sensitivity on GSO-300 ($K{=}4$).}")
    lines.append("  \\label{tab:refiner_metric_sensitivity}")
    lines.append("  \\begin{tabular}{lcccc}")
    lines.append("    \\toprule")
    lines.append("    Variant & Improv.(\\%) & Gap(\\%) & Pairwise(\\%) & $p$-value \\\\")
    lines.append("    \\midrule")

    for r in rows:
        variant = r["variant"].replace(" x ", " $\\times$ ")
        lines.append(
            "    "
            + f"{variant} & "
            + f"{r['improvement_pct']:+.2f} & "
            + f"{r['gap_closed_pct']:.1f} & "
            + f"{r['pairwise_acc_pct']:.1f} & "
            + f"{fmt_p(r['wilcoxon_p'])} \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    os.makedirs(os.path.dirname(args.output_tex), exist_ok=True)
    with open(args.output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved {args.output_tex}")


if __name__ == "__main__":
    main()

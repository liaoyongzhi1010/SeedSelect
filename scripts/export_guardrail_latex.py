#!/usr/bin/env python3
"""Export guardrail sweep JSON to a compact LaTeX table."""

import argparse
import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_JSON = os.path.join(PROJECT_DIR, "outputs", "guardrail", "guardrail_abstain_results.json")
DEFAULT_TEX = os.path.join(PROJECT_DIR, "outputs", "guardrail", "guardrail_abstain_table.tex")


def fmt_p(p):
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Export guardrail table as LaTeX")
    parser.add_argument("--input_json", default=DEFAULT_JSON)
    parser.add_argument("--output_tex", default=DEFAULT_TEX)
    args = parser.parse_args()

    with open(args.input_json) as f:
        payload = json.load(f)

    sweep = payload["sweep"]
    baseline = min(sweep, key=lambda x: abs(x["threshold"]))
    best = payload["best"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Confidence-abstain guardrail on GSO-300 ($K{=}4$).}")
    lines.append("  \\label{tab:guardrail_abstain}")
    lines.append("  \\begin{tabular}{lccccc}")
    lines.append("    \\toprule")
    lines.append("    Policy & Abstain(\\%) & Improv.(\\%) & Worst(\\%) & Severe$>$5\\%(\\%) & $p$-value \\\\")
    lines.append("    \\midrule")
    lines.append(
        "    "
        + f"SeedSelect (no guardrail) & {baseline['abstain_rate_pct']:.1f} & "
        + f"{baseline['improvement_pct']:+.2f} & {baseline['worst_pick_rate_pct']:.1f} & "
        + f"{baseline['severe_degrade_rate_pct']:.1f} & "
        + f"{fmt_p(baseline['p_value'])} \\\\"
    )
    lines.append(
        "    "
        + f"SeedSelect + abstain & {best['abstain_rate_pct']:.1f} & "
        + f"{best['improvement_pct']:+.2f} & {best['worst_pick_rate_pct']:.1f} & "
        + f"{best['severe_degrade_rate_pct']:.1f} & "
        + f"{fmt_p(best['p_value'])} \\\\"
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

"""
H2 Regression: Racial Channel — Does the CLC effect on homeownership gap
differ between majority-minority and majority-white tracts?

Hypothesis: The effect of 2019 CLC on Δhomeownership gap is significantly
larger in majority-minority tracts, meaning algorithmic pricing
disproportionately burdens communities of color.

Primary interaction model:
    delta_gap_black ~ log_clc * minority_tract + gap_black_pre + controls + C(metro)
    delta_gap_hispanic ~ log_clc * minority_tract + gap_hispanic_pre + controls + C(metro)

β3 (log_clc:minority_tract) is the key H2 coefficient.

Also runs:
  - Split-sample regressions (minority vs. white tracts separately)
  - Race-specific breakdowns (majority_black, majority_hispanic)
  - Robustness: threshold sensitivity (40/50/60%), continuous minority share,
    exclude high-income minority tracts, metro-specific CLC interaction

Saved to: data/processed/h2_gap_results.txt

Usage:
    python src/run_h2_regression.py
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CONTROLS = (
    "median_hh_income + renter_pct + pct_black + pct_hispanic "
    "+ total_housing_units + C(metro)"
)


def fmt_coef(res, var):
    if var not in res.params:
        return f"  {var}: NOT IN MODEL"
    c  = res.params[var]
    se = res.bse[var]
    p  = res.pvalues[var]
    lo, hi = res.conf_int().loc[var]
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "   "))
    return (f"  {var:<40} {c:>+9.4f}  SE={se:.4f}  p={p:.3f}{sig}  "
            f"[{lo:.4f}, {hi:.4f}]")


def section(t):  return f"\n{'='*70}\n{t}\n{'='*70}"
def subsection(t): return f"\n{'─'*70}\n{t}\n{'─'*70}"


def run_interaction(df, outcome, baseline, clc_var, interact_var, label):
    formula = (
        f"{outcome} ~ {clc_var} + {interact_var} + {clc_var}:{interact_var} "
        f"+ {baseline} + {CONTROLS}"
    )
    res = smf.ols(formula, data=df.dropna(subset=[outcome, baseline, clc_var])).fit(cov_type="HC1")
    b1 = res.params.get(clc_var, float("nan"))
    b2 = res.params.get(interact_var, float("nan"))
    b3 = res.params.get(f"{clc_var}:{interact_var}", float("nan"))
    p3 = res.pvalues.get(f"{clc_var}:{interact_var}", float("nan"))
    lines = [
        subsection(label),
        f"  N={int(res.nobs)}  R²={res.rsquared:.3f}",
        fmt_coef(res, clc_var),
        fmt_coef(res, interact_var),
        fmt_coef(res, f"{clc_var}:{interact_var}"),
        f"\n  CLC effect in white tracts (β1):              {b1:+.4f}",
        f"  Additional CLC effect in minority tracts (β3): {b3:+.4f}",
        f"  Total CLC effect in minority tracts (β1+β3):   {b1+b3:+.4f}",
        f"  β3 {'✓ SIGNIFICANT' if p3<0.05 else '~ MARGINAL' if p3<0.10 else '✗ NOT SIGNIFICANT'} (p={p3:.3f})",
    ]
    return "\n".join(lines), res, b1, b2, b3, p3


def main():
    path = PROCESSED_DIR / "regression_master.csv"
    if not path.exists():
        raise SystemExit("regression_master.csv not found. Run build_regression_dataset.py first.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} tracts")

    num_cols = ["log_clc", "clc_winsorized", "gap_black_pre", "gap_black_post",
                "delta_gap_black", "gap_hispanic_pre", "gap_hispanic_post",
                "delta_gap_hispanic", "median_hh_income", "renter_pct",
                "pct_black", "pct_hispanic", "pct_white_nh",
                "total_housing_units", "ahbi_pre"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived variables
    df["minority_share"]    = df["pct_black"] + df["pct_hispanic"]
    df["minority_tract"]    = (df["minority_share"] > 0.50).astype(int)
    df["majority_black"]    = (df["pct_black"]    > 0.50).astype(int)
    df["majority_hispanic"] = (df["pct_hispanic"] > 0.50).astype(int)
    hhi_thresh = df["median_hh_income"].quantile(0.75)
    df["high_income"] = (df["median_hh_income"] > hhi_thresh).astype(int)

    print(f"Minority tract (>50%): {df['minority_tract'].sum()} ({df['minority_tract'].mean()*100:.1f}%)")
    print(f"Majority Black:        {df['majority_black'].sum()}")
    print(f"Majority Hispanic:     {df['majority_hispanic'].sum()}")
    print(f"High-income threshold: ${hhi_thresh:,.0f}\n")

    out = []

    # ═══════════════════════════════════════════════════════════════════════════
    # PRIMARY INTERACTION MODELS
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("H2 PRIMARY: Interaction Models (majority-minority > 50%)"))

    for outcome, baseline, outcome_label in [
        ("delta_gap_black",    "gap_black_pre",    "Black-white gap"),
        ("delta_gap_hispanic", "gap_hispanic_pre", "Hispanic-white gap"),
    ]:
        block, res_int, b1, b2, b3, p3 = run_interaction(
            df, outcome, baseline, "log_clc", "minority_tract",
            f"PRIMARY — {outcome_label}"
        )
        out.append(block)
        out.append(f"\n  Four-cell summary ({outcome_label}):")
        out.append(f"  {'White, low CLC':<35}  baseline")
        out.append(f"  {'White, high CLC (β1)':<35}  {b1:+.4f}")
        out.append(f"  {'Minority, low CLC (β2)':<35}  {b2:+.4f}")
        out.append(f"  {'Minority, high CLC (β1+β2+β3)':<35}  {b1+b2+b3:+.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SPLIT SAMPLE
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("SPLIT-SAMPLE REGRESSIONS"))

    for outcome, baseline, outcome_label in [
        ("delta_gap_black",    "gap_black_pre",    "Black gap"),
        ("delta_gap_hispanic", "gap_hispanic_pre", "Hispanic gap"),
    ]:
        out.append(subsection(f"Split sample — {outcome_label}"))
        for label, mask in [
            ("Majority-minority (>50%)", df["minority_tract"] == 1),
            ("Majority-white (≤50%)",    df["minority_tract"] == 0),
        ]:
            sub = df[mask].dropna(subset=[outcome, baseline, "log_clc"])
            # Drop metros with <3 tracts for FE stability
            mc = sub["metro"].value_counts()
            sub = sub[sub["metro"].isin(mc[mc >= 3].index)]
            formula = f"{outcome} ~ log_clc + {baseline} + {CONTROLS}"
            res = smf.ols(formula, data=sub).fit(cov_type="HC1")
            out.append(f"\n  {label}  [N={len(sub)}]")
            out.append(fmt_coef(res, "log_clc"))

    # ═══════════════════════════════════════════════════════════════════════════
    # RACE-SPECIFIC BREAKDOWNS
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("RACE-SPECIFIC BREAKDOWNS"))

    for race_var, race_label in [
        ("majority_black",    "Majority Black (>50%)"),
        ("majority_hispanic", "Majority Hispanic (>50%)"),
    ]:
        for outcome, baseline, outcome_label in [
            ("delta_gap_black",    "gap_black_pre",    "Black gap"),
            ("delta_gap_hispanic", "gap_hispanic_pre", "Hispanic gap"),
        ]:
            block, _, b1, _, b3, p3 = run_interaction(
                df, outcome, baseline, "log_clc", race_var,
                f"{race_label} × {outcome_label}"
            )
            out.append(block)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROBUSTNESS CHECKS
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("ROBUSTNESS CHECKS"))

    # R1: Threshold sensitivity
    out.append(subsection("R1: Minority Threshold Sensitivity — Black gap outcome"))
    out.append(f"  {'Threshold':>10}  {'β1 (white)':>12}  {'β3 (interaction)':>18}  {'p(β3)':>8}  {'N_min':>6}")
    out.append("  " + "─" * 60)
    for thresh in [0.40, 0.50, 0.60]:
        mvar = f"minority_{int(thresh*100)}"
        df[mvar] = (df["minority_share"] > thresh).astype(int)
        sub = df.dropna(subset=["delta_gap_black", "gap_black_pre", "log_clc"])
        formula = f"delta_gap_black ~ log_clc + {mvar} + log_clc:{mvar} + gap_black_pre + {CONTROLS}"
        res = smf.ols(formula, data=sub).fit(cov_type="HC1")
        b1_t = res.params.get("log_clc", float("nan"))
        b3_t = res.params.get(f"log_clc:{mvar}", float("nan"))
        p3_t = res.pvalues.get(f"log_clc:{mvar}", float("nan"))
        sig  = "***" if p3_t<0.01 else "**" if p3_t<0.05 else "*" if p3_t<0.10 else ""
        out.append(f"  {thresh*100:>9.0f}%  {b1_t:>+12.4f}  {b3_t:>+18.4f}  {p3_t:>7.3f}{sig:<3}  {df[mvar].sum():>6}")

    # R2: Continuous minority share
    out.append(subsection("R2: Continuous Minority Share"))
    for outcome, baseline, olabel in [
        ("delta_gap_black",    "gap_black_pre",    "Black gap"),
        ("delta_gap_hispanic", "gap_hispanic_pre", "Hispanic gap"),
    ]:
        sub = df.dropna(subset=[outcome, baseline, "log_clc"])
        formula = f"{outcome} ~ log_clc + minority_share + log_clc:minority_share + {baseline} + {CONTROLS}"
        res = smf.ols(formula, data=sub).fit(cov_type="HC1")
        out.append(f"\n  {olabel}  [N={len(sub)}]")
        out.append(fmt_coef(res, "log_clc"))
        out.append(fmt_coef(res, "log_clc:minority_share"))

    # R3: Exclude high-income minority tracts
    out.append(subsection(f"R3: Exclude High-Income Minority Tracts (income > ${hhi_thresh:,.0f})"))
    df_ng = df[~((df["minority_tract"] == 1) & (df["high_income"] == 1))]
    out.append(f"  Dropped {len(df)-len(df_ng)} tracts. Remaining: {len(df_ng)}")
    for outcome, baseline, olabel in [
        ("delta_gap_black",    "gap_black_pre",    "Black gap"),
        ("delta_gap_hispanic", "gap_hispanic_pre", "Hispanic gap"),
    ]:
        sub = df_ng.dropna(subset=[outcome, baseline, "log_clc"])
        formula = f"{outcome} ~ log_clc + minority_tract + log_clc:minority_tract + {baseline} + {CONTROLS}"
        res = smf.ols(formula, data=sub).fit(cov_type="HC1")
        out.append(f"\n  {olabel}  [N={len(sub)}]")
        out.append(fmt_coef(res, "log_clc"))
        out.append(fmt_coef(res, "log_clc:minority_tract"))

    # R4: Metro-specific CLC interaction
    out.append(subsection("R4: Metro-Specific CLC Interaction"))
    for outcome, baseline, olabel in [
        ("delta_gap_black",    "gap_black_pre",    "Black gap"),
        ("delta_gap_hispanic", "gap_hispanic_pre", "Hispanic gap"),
    ]:
        sub = df.dropna(subset=[outcome, baseline, "log_clc"])
        formula = (
            f"{outcome} ~ log_clc + minority_tract + log_clc:minority_tract "
            f"+ log_clc:C(metro) + {baseline} + {CONTROLS}"
        )
        res = smf.ols(formula, data=sub).fit(cov_type="HC1")
        out.append(f"\n  {olabel}  N={len(sub)}  R²={res.rsquared:.3f}")
        for v in [k for k in res.params.index if "log_clc" in k and "metro" not in k]:
            out.append(fmt_coef(res, v))

    # ── Write output ──────────────────────────────────────────────────────────
    output = "\n".join(out)
    print(output)

    result_path = PROCESSED_DIR / "h2_gap_results.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"\n\nSaved → {result_path.name}")


if __name__ == "__main__":
    main()

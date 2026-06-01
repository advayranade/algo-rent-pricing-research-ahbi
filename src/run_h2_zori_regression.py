"""
H2 ZORI Regression: Does the CLC effect on rent growth differ between
majority-minority and majority-white tracts?

Primary interaction model:
    rent_growth ~ log_clc * minority_tract + ahbi_pre + controls + C(metro)

β3 (log_clc:minority_tract) is the key H2 coefficient.

Also runs:
  - Split-sample regressions (minority vs. white tracts separately)
  - Race-specific breakdowns (majority_black, majority_hispanic)
  - R1: Threshold sensitivity (40/50/60%)
  - R2: Continuous minority share

Saved to: data/processed/h2_zori_results.txt
"""
from pathlib import Path
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


def section(t):    return f"\n{'='*70}\n{t}\n{'='*70}"
def subsection(t): return f"\n{'─'*70}\n{t}\n{'─'*70}"


def run_interaction(df, clc_var, interact_var, label):
    formula = (
        f"rent_growth ~ {clc_var} + {interact_var} + {clc_var}:{interact_var} "
        f"+ ahbi_pre + {CONTROLS}"
    )
    sub = df.dropna(subset=["rent_growth", "ahbi_pre", clc_var])
    res = smf.ols(formula, data=sub).fit(cov_type="HC1")
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
        fmt_coef(res, "ahbi_pre"),
        f"\n  CLC effect in white tracts (β1):                {b1:+.4f}",
        f"  Additional CLC effect in minority tracts (β3):  {b3:+.4f}",
        f"  Total CLC effect in minority tracts (β1+β3):    {b1+b3:+.4f}",
        f"  β3 {'✓ SIGNIFICANT' if p3<0.05 else '~ MARGINAL' if p3<0.10 else '✗ NOT SIGNIFICANT'} (p={p3:.3f})",
    ]
    return "\n".join(lines), res, b1, b2, b3, p3


def main():
    path = PROCESSED_DIR / "regression_master.csv"
    if not path.exists():
        raise SystemExit("regression_master.csv not found.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} tracts")

    if "rent_growth" not in df.columns:
        raise SystemExit("rent_growth column not found. Run add_zori_to_regression.py first.")

    num_cols = ["log_clc", "clc_winsorized", "ahbi_pre", "rent_growth",
                "median_hh_income", "renter_pct", "pct_black",
                "pct_hispanic", "pct_white_nh", "total_housing_units"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["minority_share"]    = df["pct_black"] + df["pct_hispanic"]
    df["minority_tract"]    = (df["minority_share"] > 0.50).astype(int)
    df["majority_black"]    = (df["pct_black"]    > 0.50).astype(int)
    df["majority_hispanic"] = (df["pct_hispanic"] > 0.50).astype(int)

    print(f"Tracts with rent_growth: {df['rent_growth'].notna().sum()}")
    print(f"Minority tract (>50%): {df['minority_tract'].sum()} ({df['minority_tract'].mean()*100:.1f}%)")
    print(f"Majority Black:        {df['majority_black'].sum()}")
    print(f"Majority Hispanic:     {df['majority_hispanic'].sum()}\n")

    out = []

    # ═══════════════════════════════════════════════════════════════════════════
    # PRIMARY INTERACTION
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("H2 PRIMARY ZORI: rent_growth ~ log_clc * minority_tract"))

    block, res_int, b1, b2, b3, p3 = run_interaction(
        df, "log_clc", "minority_tract", "PRIMARY INTERACTION (minority > 50%)"
    )
    out.append(block)
    pct_white = b1 * 100
    pct_min   = (b1 + b3) * 100
    out.append(
        f"\n  Interpretation:\n"
        f"  White tracts: +{pct_white:.1f} ppt rent growth per CLC doubling\n"
        f"  Minority tracts: +{pct_min:.1f} ppt rent growth per CLC doubling\n"
        f"  Differential (β3): {b3*100:+.1f} ppt  p={p3:.3f}\n"
        f"  H2 {'SUPPORTED' if b3 > 0 and p3 < 0.05 else 'NOT SUPPORTED at p<0.05' if p3 >= 0.05 else 'MARGINAL'}"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SPLIT SAMPLE
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("SPLIT-SAMPLE REGRESSIONS"))

    for label, mask in [
        ("Majority-minority (>50%)", df["minority_tract"] == 1),
        ("Majority-white (≤50%)",    df["minority_tract"] == 0),
    ]:
        sub = df[mask].dropna(subset=["rent_growth", "ahbi_pre", "log_clc"])
        mc  = sub["metro"].value_counts()
        sub = sub[sub["metro"].isin(mc[mc >= 3].index)]
        formula = f"rent_growth ~ log_clc + ahbi_pre + {CONTROLS}"
        res = smf.ols(formula, data=sub).fit(cov_type="HC1")
        out.append(f"\n  {label}  [N={len(sub)}]")
        out.append(fmt_coef(res, "log_clc"))
        out.append(fmt_coef(res, "ahbi_pre"))

    # ═══════════════════════════════════════════════════════════════════════════
    # RACE-SPECIFIC BREAKDOWNS
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("RACE-SPECIFIC BREAKDOWNS"))

    for race_var, race_label in [
        ("majority_black",    "Majority Black (>50%)"),
        ("majority_hispanic", "Majority Hispanic (>50%)"),
    ]:
        block, _, b1, _, b3, p3 = run_interaction(
            df, "log_clc", race_var, f"{race_label} interaction"
        )
        out.append(block)

    # ═══════════════════════════════════════════════════════════════════════════
    # ROBUSTNESS CHECKS
    # ═══════════════════════════════════════════════════════════════════════════
    out.append(section("ROBUSTNESS CHECKS"))

    # R1: Threshold sensitivity
    out.append(subsection("R1: Minority Threshold Sensitivity"))
    out.append(f"  {'Threshold':>10}  {'β1 (white)':>12}  {'β3 (interaction)':>18}  {'p(β3)':>8}  {'N_min':>6}")
    out.append("  " + "─" * 60)
    for thresh in [0.40, 0.50, 0.60]:
        mvar = f"minority_{int(thresh*100)}"
        df[mvar] = (df["minority_share"] > thresh).astype(int)
        sub = df.dropna(subset=["rent_growth", "ahbi_pre", "log_clc"])
        formula = f"rent_growth ~ log_clc + {mvar} + log_clc:{mvar} + ahbi_pre + {CONTROLS}"
        res = smf.ols(formula, data=sub).fit(cov_type="HC1")
        b1_t = res.params.get("log_clc", float("nan"))
        b3_t = res.params.get(f"log_clc:{mvar}", float("nan"))
        p3_t = res.pvalues.get(f"log_clc:{mvar}", float("nan"))
        sig  = "***" if p3_t < 0.01 else "**" if p3_t < 0.05 else "*" if p3_t < 0.10 else ""
        out.append(f"  {thresh*100:>9.0f}%  {b1_t:>+12.4f}  {b3_t:>+18.4f}  {p3_t:>7.3f}{sig:<3}  {df[mvar].sum():>6}")

    # R2: Continuous minority share
    out.append(subsection("R2: Continuous Minority Share"))
    sub = df.dropna(subset=["rent_growth", "ahbi_pre", "log_clc"])
    formula = "rent_growth ~ log_clc + minority_share + log_clc:minority_share + ahbi_pre + " + CONTROLS
    res = smf.ols(formula, data=sub).fit(cov_type="HC1")
    out.append(f"  N={len(sub)}")
    out.append(fmt_coef(res, "log_clc"))
    out.append(fmt_coef(res, "log_clc:minority_share"))
    out.append(fmt_coef(res, "ahbi_pre"))

    # ── Write output ──────────────────────────────────────────────────────────
    output = "\n".join(out)
    print(output)

    result_path = PROCESSED_DIR / "h2_zori_results.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"\n\nSaved → {result_path.name}")


if __name__ == "__main__":
    main()

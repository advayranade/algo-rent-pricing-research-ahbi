"""
H1 Regression: Does 2019 REIT concentration predict widening racial
homeownership gaps by 2022?

Primary outcomes:
  delta_gap_black    = homeownership_gap_black_2022 − homeownership_gap_black_2019
  delta_gap_hispanic = homeownership_gap_hispanic_2022 − homeownership_gap_hispanic_2019

A positive coefficient on log_clc means higher REIT concentration in 2019
predicts a wider racial homeownership gap by 2022 — supporting H1.

Models:
  A  (primary):   delta_gap_black    ~ log_clc + gap_black_pre    + controls + C(metro)
  B  (primary):   delta_gap_hispanic ~ log_clc + gap_hispanic_pre + controls + C(metro)
  Ab (robustness): Model A with raw winsorized CLC instead of log_clc
  Bb (robustness): Model B with raw winsorized CLC instead of log_clc

Saved to: data/processed/h1_gap_results.txt

Usage:
    python src/run_h1_regression.py
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
    return (f"  {var:<30} {c:>+9.4f}  SE={se:.4f}  p={p:.3f}{sig}  "
            f"95% CI [{lo:.4f}, {hi:.4f}]")


def run_model(formula, data, label):
    res = smf.ols(formula, data=data).fit(cov_type="HC1")
    return res


def model_block(res, clc_var, gap_pre_var, label):
    lines = []
    lines.append(f"\n{'─'*65}")
    lines.append(f"{label}")
    lines.append(f"N={int(res.nobs)}  R²={res.rsquared:.3f}  Adj.R²={res.rsquared_adj:.3f}")
    lines.append(f"{'─'*65}")
    lines.append(fmt_coef(res, clc_var))
    lines.append(fmt_coef(res, gap_pre_var))
    lines.append(fmt_coef(res, "median_hh_income"))
    lines.append(fmt_coef(res, "renter_pct"))
    lines.append(fmt_coef(res, "pct_black"))
    lines.append(fmt_coef(res, "pct_hispanic"))
    return "\n".join(lines)


def headline(res, clc_var, outcome_label, direction_label):
    c   = res.params.get(clc_var, float("nan"))
    p   = res.pvalues.get(clc_var, float("nan"))
    lo, hi = res.conf_int().loc[clc_var] if clc_var in res.params else (float("nan"), float("nan"))
    sig = "✓ SIGNIFICANT" if p < 0.05 else ("~ MARGINAL (p<0.10)" if p < 0.10 else "✗ NOT SIGNIFICANT")
    direction = "WIDENS" if c > 0 else "NARROWS"
    support = (
        "→ H1 SUPPORTED: High-REIT tracts show wider racial gap."
        if c > 0 and p < 0.05
        else "→ H1 NOT SUPPORTED at p < 0.05."
        if p >= 0.05
        else "→ H1 DIRECTIONALLY CONSISTENT but marginal."
    )
    return (
        f"\n  {outcome_label}\n"
        f"  {clc_var} coef: {c:+.4f}  95% CI [{lo:.4f}, {hi:.4f}]  p={p:.3f}  {sig}\n"
        f"  N={int(res.nobs)}  R²={res.rsquared:.3f}\n"
        f"  A doubling of REIT concentration {direction} the {direction_label} gap.\n"
        f"  {support}"
    )


def main():
    path = PROCESSED_DIR / "regression_master.csv"
    if not path.exists():
        raise SystemExit("regression_master.csv not found. Run build_regression_dataset.py first.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} tracts")

    # Numeric conversion
    num_cols = ["log_clc", "clc_winsorized", "gap_black_pre", "gap_black_post",
                "delta_gap_black", "gap_hispanic_pre", "gap_hispanic_post",
                "delta_gap_hispanic", "median_hh_income", "renter_pct",
                "pct_black", "pct_hispanic", "total_housing_units", "ahbi_pre"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Subsets per outcome (drop rows with missing outcome or key predictors)
    base_vars = ["log_clc", "clc_winsorized", "median_hh_income", "renter_pct",
                 "pct_black", "pct_hispanic", "total_housing_units", "metro"]

    df_black = df.dropna(subset=base_vars + ["delta_gap_black", "gap_black_pre"])
    df_hisp  = df.dropna(subset=base_vars + ["delta_gap_hispanic", "gap_hispanic_pre"])

    print(f"Tracts for Black gap model:    {len(df_black):,}")
    print(f"Tracts for Hispanic gap model: {len(df_hisp):,}")
    print(f"delta_gap_black mean:    {df_black['delta_gap_black'].mean():+.4f}")
    print(f"delta_gap_hispanic mean: {df_hisp['delta_gap_hispanic'].mean():+.4f}\n")

    # ── Model A: Black gap, log_clc (primary) ─────────────────────────────────
    fa = f"delta_gap_black ~ log_clc + gap_black_pre + {CONTROLS}"
    res_a = run_model(fa, df_black, "Model A")

    # ── Model Ab: Black gap, raw winsorized CLC (robustness) ──────────────────
    fab = f"delta_gap_black ~ clc_winsorized + gap_black_pre + {CONTROLS}"
    res_ab = run_model(fab, df_black, "Model Ab")

    # ── Model B: Hispanic gap, log_clc (primary) ──────────────────────────────
    fb = f"delta_gap_hispanic ~ log_clc + gap_hispanic_pre + {CONTROLS}"
    res_b = run_model(fb, df_hisp, "Model B")

    # ── Model Bb: Hispanic gap, raw winsorized CLC (robustness) ───────────────
    fbb = f"delta_gap_hispanic ~ clc_winsorized + gap_hispanic_pre + {CONTROLS}"
    res_bb = run_model(fbb, df_hisp, "Model Bb")

    # ── Build output ──────────────────────────────────────────────────────────
    out = []
    out.append("=" * 70)
    out.append("H1 GAP REGRESSION RESULTS")
    out.append("Treatment: log_clc = log(1 + winsorized CLC)  [99th pct winsorized]")
    out.append("Outcomes: Δ racial homeownership gap (2019 → 2023)")
    out.append("SEs: HC1 heteroskedasticity-robust throughout")
    out.append("=" * 70)

    out.append("\n\n══ HEADLINE RESULTS ══")
    out.append(headline(res_a,  "log_clc", "delta_gap_black",    "Black-white"))
    out.append(headline(res_b,  "log_clc", "delta_gap_hispanic", "Hispanic-white"))

    out.append("\n\n══ MODEL A: delta_gap_black ~ log_clc (PRIMARY) ══")
    out.append(model_block(res_a, "log_clc", "gap_black_pre", "Model A — Black gap, log CLC"))
    out.append(res_a.summary().as_text())

    out.append("\n\n══ MODEL Ab: delta_gap_black ~ clc_winsorized (ROBUSTNESS) ══")
    out.append(model_block(res_ab, "clc_winsorized", "gap_black_pre", "Model Ab — Black gap, raw CLC"))

    out.append("\n\n══ MODEL B: delta_gap_hispanic ~ log_clc (PRIMARY) ══")
    out.append(model_block(res_b, "log_clc", "gap_hispanic_pre", "Model B — Hispanic gap, log CLC"))
    out.append(res_b.summary().as_text())

    out.append("\n\n══ MODEL Bb: delta_gap_hispanic ~ clc_winsorized (ROBUSTNESS) ══")
    out.append(model_block(res_bb, "clc_winsorized", "gap_hispanic_pre", "Model Bb — Hispanic gap, raw CLC"))

    out.append("\n\n══ METRO-CLUSTERED SE ROBUSTNESS (10 clusters — interpret cautiously) ══")
    for res, label, clc_var in [
        (res_a, "Model A Black gap", "log_clc"),
        (res_b, "Model B Hispanic gap", "log_clc"),
    ]:
        mod = smf.ols(res.model.formula, data=res.model.data.frame)
        rc  = mod.fit(cov_type="cluster",
                      cov_kwds={"groups": res.model.data.frame["metro"]})
        out.append(f"\n  {label} (clustered SEs):")
        out.append(fmt_coef(rc, clc_var))

    output = "\n".join(out)
    print(output)

    result_path = PROCESSED_DIR / "h1_gap_results.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"\nSaved → {result_path.name}")


if __name__ == "__main__":
    main()

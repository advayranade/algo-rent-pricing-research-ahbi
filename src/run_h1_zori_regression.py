"""
H1 ZORI Regression: Does 2019 REIT concentration predict rent growth (2019→2023)?
Outcome: rent_growth = (ZORI_2023 - ZORI_2019) / ZORI_2019

Metro fixed effects are intentionally omitted from primary models. Metro FEs absorb
all between-metro variation, which is where the CLC signal lives (high-CLC metros like
Dallas/Charlotte had ~30% rent growth; low-CLC metros like SF/San Jose had ~5%). FEs
eliminate this comparison entirely. Metro-clustered SEs are used instead — a lighter
touch that corrects for within-metro correlation without removing between-metro information.

Models:
  1  (primary):    rent_growth ~ log_clc + ahbi_pre + controls  [metro-clustered SEs]
  1b (robustness): same with clc_winsorized
  2  (robustness): Model 1 with HC1 robust SEs
  3  (diagnostic): Model 1 with metro FEs added — shows what FEs suppress

Saved to: data/processed/h1_zori_results.txt
"""
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CONTROLS = (
    "median_hh_income + renter_pct + pct_black + pct_hispanic + total_housing_units"
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


def model_block(res, clc_var, label):
    lines = [
        f"\n{'─'*65}",
        label,
        f"N={int(res.nobs)}  R²={res.rsquared:.3f}  Adj.R²={res.rsquared_adj:.3f}",
        f"{'─'*65}",
        fmt_coef(res, clc_var),
        fmt_coef(res, "ahbi_pre"),
        fmt_coef(res, "median_hh_income"),
        fmt_coef(res, "renter_pct"),
        fmt_coef(res, "pct_black"),
        fmt_coef(res, "pct_hispanic"),
    ]
    return "\n".join(lines)


def headline(res, clc_var):
    c   = res.params.get(clc_var, float("nan"))
    p   = res.pvalues.get(clc_var, float("nan"))
    lo, hi = res.conf_int().loc[clc_var] if clc_var in res.params else (float("nan"), float("nan"))
    sig = "✓ SIGNIFICANT (p<0.05)" if p < 0.05 else ("~ MARGINAL (p<0.10)" if p < 0.10 else "✗ NOT SIGNIFICANT")
    pct_pts = c * 100
    support = (
        "→ H1 SUPPORTED: REIT concentration predicts higher rent growth."
        if c > 0 and p < 0.05
        else "→ H1 NOT SUPPORTED at p < 0.05."
        if p >= 0.05
        else "→ H1 DIRECTIONALLY CONSISTENT but marginal."
    )
    return (
        f"\n  rent_growth coefficient on {clc_var}: {c:+.4f}  "
        f"95% CI [{lo:.4f}, {hi:.4f}]  p={p:.3f}  {sig}\n"
        f"  Interpretation: A doubling of REIT concentration is associated with\n"
        f"  {pct_pts:+.1f} percentage point {'higher' if c > 0 else 'lower'} rent growth (2019→2023).\n"
        f"  N={int(res.nobs)}  R²={res.rsquared:.3f}\n"
        f"  {support}"
    )


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
                "pct_hispanic", "total_housing_units"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    model_vars = ["log_clc", "clc_winsorized", "ahbi_pre", "rent_growth",
                  "median_hh_income", "renter_pct", "pct_black",
                  "pct_hispanic", "total_housing_units", "metro"]

    df_m = df.dropna(subset=model_vars)
    print(f"Tracts for H1 ZORI model: {len(df_m):,}")
    print(f"rent_growth: mean={df_m['rent_growth'].mean():.4f}  "
          f"std={df_m['rent_growth'].std():.4f}  "
          f"range=[{df_m['rent_growth'].min():.4f}, {df_m['rent_growth'].max():.4f}]")
    print(f"\nDescriptives by metro:")
    print(df_m.groupby("metro")[["rent_growth", "log_clc"]].mean().round(3).to_string())
    print()

    # ── Model 1: log_clc, metro-clustered SEs (primary) ──────────────────────
    f1 = f"rent_growth ~ log_clc + ahbi_pre + {CONTROLS}"
    res1 = smf.ols(f1, data=df_m).fit(
        cov_type="cluster", cov_kwds={"groups": df_m["metro"]}
    )

    # ── Model 1b: clc_winsorized, metro-clustered SEs (robustness) ───────────
    f1b = f"rent_growth ~ clc_winsorized + ahbi_pre + {CONTROLS}"
    res1b = smf.ols(f1b, data=df_m).fit(
        cov_type="cluster", cov_kwds={"groups": df_m["metro"]}
    )

    # ── Model 2: log_clc, HC1 robust SEs (robustness) ────────────────────────
    res2 = smf.ols(f1, data=df_m).fit(cov_type="HC1")

    # ── Model 3: log_clc + metro FEs (diagnostic — shows what FEs suppress) ──
    f3 = f"rent_growth ~ log_clc + ahbi_pre + {CONTROLS} + C(metro)"
    res3 = smf.ols(f3, data=df_m).fit(
        cov_type="cluster", cov_kwds={"groups": df_m["metro"]}
    )

    # ── Build output ─────────────────────────────────────────────────────────
    out = []
    out.append("=" * 70)
    out.append("H1 ZORI REGRESSION RESULTS")
    out.append("Outcome: rent_growth = (ZORI_2023 - ZORI_2019) / ZORI_2019")
    out.append("Treatment: log_clc = log(1 + winsorized CLC)  [99th pct winsorized]")
    out.append("")
    out.append("NOTE: Metro fixed effects are excluded from primary models.")
    out.append("Between-metro variation (Dallas/Charlotte high CLC + high rent growth")
    out.append("vs. SF/San Jose low CLC + low rent growth) is central to the hypothesis.")
    out.append("Metro-clustered SEs correct for within-metro error correlation.")
    out.append("Model 3 (diagnostic) shows what metro FEs do to the estimate.")
    out.append("=" * 70)

    out.append("\n\n══ HEADLINE RESULT (Model 1: metro-clustered SEs, no metro FE) ══")
    out.append(headline(res1, "log_clc"))

    out.append("\n\n══ MODEL 1: PRIMARY — log_clc, metro-clustered SEs ══")
    out.append(model_block(res1, "log_clc", "Model 1 — log CLC, metro-clustered SEs (no metro FE)"))
    out.append(res1.summary().as_text())

    out.append("\n\n══ MODEL 1b: ROBUSTNESS — clc_winsorized, metro-clustered SEs ══")
    out.append(model_block(res1b, "clc_winsorized", "Model 1b — raw CLC, metro-clustered SEs (no metro FE)"))
    out.append(headline(res1b, "clc_winsorized"))

    out.append("\n\n══ MODEL 2: ROBUSTNESS — log_clc, HC1 robust SEs ══")
    out.append(model_block(res2, "log_clc", "Model 2 — log CLC, HC1 robust SEs (no metro FE)"))
    out.append(headline(res2, "log_clc"))

    out.append("\n\n══ MODEL 3: DIAGNOSTIC — log_clc + metro FEs (what FEs suppress) ══")
    out.append(model_block(res3, "log_clc", "Model 3 — log CLC, metro-clustered SEs + metro FE"))
    out.append(headline(res3, "log_clc"))
    c1 = res1.params.get("log_clc", float("nan"))
    c3 = res3.params.get("log_clc", float("nan"))
    out.append(
        f"\n  Attenuation from metro FEs: {c1:.4f} → {c3:.4f} "
        f"({(c3-c1)/abs(c1)*100:+.1f}% change)\n"
        f"  This shows how much between-metro signal metro FEs absorb."
    )

    out.append("\n\n══ COLLINEARITY CHECK ══")
    corr = df_m[["log_clc", "ahbi_pre"]].corr().iloc[0, 1]
    out.append(f"  corr(log_clc, ahbi_pre) = {corr:.3f}")
    if abs(corr) > 0.7:
        out.append("  WARNING: High correlation — ahbi_pre may be absorbing CLC variance.")
        out.append("  Consider running Model 1 without ahbi_pre as a sensitivity check.")
    else:
        out.append("  OK: Correlation below 0.7 — collinearity unlikely to be a problem.")

    # Sensitivity: drop ahbi_pre if collinear
    out.append("\n\n══ SENSITIVITY: Model 1 without ahbi_pre ══")
    f_noahbi = f"rent_growth ~ log_clc + {CONTROLS}"
    res_noahbi = smf.ols(f_noahbi, data=df_m).fit(
        cov_type="cluster", cov_kwds={"groups": df_m["metro"]}
    )
    out.append(model_block(res_noahbi, "log_clc", "No ahbi_pre — metro-clustered SEs"))
    out.append(headline(res_noahbi, "log_clc"))

    output = "\n".join(out)
    print(output)

    result_path = PROCESSED_DIR / "h1_zori_results.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"\nSaved → {result_path.name}")


if __name__ == "__main__":
    main()

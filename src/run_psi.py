"""
Population Stability Index (PSI) — Appendix E

Two checks per variable:
  Temporal PSI  : 2019 baseline → 2023 comparison
                  Answers: did the distribution shift structurally over the study window?
  Subgroup PSI  : white-majority tracts (baseline) → minority-majority tracts (comparison)
                  Answers: are these genuinely different populations in CLC/PRB/HMT exposure,
                  motivating the H2 interaction term?

Variables: CLC (Corporate Landlord Concentration), PRB (Rent Burden Rate), HMT (Market Tightness)

PSI formula: Σ (actual% - expected%) × ln(actual% / expected%)
Thresholds:  < 0.10  → Stable
             0.10–0.20 → Monitor
             ≥ 0.20  → Shift

Outputs:
  data/processed/psi_results.csv
  data/processed/psi_results.txt  (formatted for Appendix E)
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

N_BINS  = 10
EPSILON = 1e-4  # prevents log(0) when a bin is empty


def psi(baseline: np.ndarray, comparison: np.ndarray, bins: np.ndarray) -> tuple[float, pd.DataFrame]:
    """
    Compute PSI given pre-defined bin edges.
    Returns (psi_value, bin_detail_dataframe).
    """
    base_counts = np.histogram(baseline,  bins=bins)[0]
    comp_counts = np.histogram(comparison, bins=bins)[0]

    base_pct = base_counts / base_counts.sum()
    comp_pct = comp_counts / comp_counts.sum()

    # Replace zeros with epsilon to avoid log(0)
    base_pct = np.where(base_pct == 0, EPSILON, base_pct)
    comp_pct = np.where(comp_pct == 0, EPSILON, comp_pct)

    psi_bins = (comp_pct - base_pct) * np.log(comp_pct / base_pct)
    psi_val  = float(psi_bins.sum())

    detail = pd.DataFrame({
        "bin":        range(1, N_BINS + 1),
        "base_pct":   (base_pct * 100).round(1),
        "comp_pct":   (comp_pct * 100).round(1),
        "psi_bin":    psi_bins.round(4),
    })
    return psi_val, detail


def stability_label(val: float) -> str:
    if val < 0.10:
        return "Stable"
    elif val < 0.20:
        return "Monitor"
    return "Shift"


def make_bins(series: pd.Series) -> np.ndarray:
    """Decile bin edges from baseline series, with forced open outer bounds."""
    quantiles = np.linspace(0, 100, N_BINS + 1)
    edges = np.percentile(series.dropna(), quantiles)
    edges[0]  = -np.inf
    edges[-1] = np.inf
    # Deduplicate edges (can occur with many zeros)
    edges = np.unique(edges)
    return edges


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    master = pd.read_csv(
        PROCESSED_DIR / "regression_master.csv",
        dtype={"census_tract_geoid": str},
    )
    master["census_tract_geoid"] = master["census_tract_geoid"].str.zfill(11)

    # 2023 CLC from c1_clc_2023.csv
    clc23 = pd.read_csv(
        PROCESSED_DIR / "c1_clc_2023.csv",
        dtype={"census_tract_geoid": str},
    )
    clc23["census_tract_geoid"] = clc23["census_tract_geoid"].str.zfill(11)
    clc23 = clc23[["census_tract_geoid", "clc"]].rename(columns={"clc": "clc_2023"})

    df = master.merge(clc23, on="census_tract_geoid", how="left")

    # Numeric coercion
    for col in ["clc_2019", "clc_2023", "prb_norm_2019", "prb_norm_2023",
                "tightness_norm_2019", "tightness_norm_2023", "pct_black", "pct_hispanic"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Minority flag
    df["minority_tract"] = ((df["pct_black"] + df["pct_hispanic"]) > 0.50).astype(int)

    n_minority = df["minority_tract"].sum()
    n_white    = (df["minority_tract"] == 0).sum()
    n_clc23    = df["clc_2023"].notna().sum()

    print(f"Sample: {len(df):,} tracts")
    print(f"  Minority tracts (>50%): {n_minority} | White tracts: {n_white}")
    print(f"  Tracts with 2023 CLC:   {n_clc23}\n")

    # ── Variable specs ────────────────────────────────────────────────────────
    # Each entry: (label, col_base, col_compare_temporal, col_base_subgroup, col_compare_subgroup)
    # For temporal: baseline=2019, comparison=2023
    # For subgroup: baseline=white tracts (2019 values), comparison=minority tracts (2019 values)

    variables = [
        {
            "name":    "CLC",
            "base_t":  "clc_2019",
            "comp_t":  "clc_2023",
            "base_s":  "clc_2019",
            "comp_s":  "clc_2019",
        },
        {
            "name":    "PRB",
            "base_t":  "prb_norm_2019",
            "comp_t":  "prb_norm_2023",
            "base_s":  "prb_norm_2019",
            "comp_s":  "prb_norm_2019",
        },
        {
            "name":    "HMT",
            "base_t":  "tightness_norm_2019",
            "comp_t":  "tightness_norm_2023",
            "base_s":  "tightness_norm_2019",
            "comp_s":  "tightness_norm_2019",
        },
    ]

    rows = []
    detail_blocks = []

    for var in variables:
        name   = var["name"]
        base_t = df[var["base_t"]].dropna()
        comp_t = df[var["comp_t"]].dropna()
        base_s = df.loc[df["minority_tract"] == 0, var["base_s"]].dropna()
        comp_s = df.loc[df["minority_tract"] == 1, var["comp_s"]].dropna()

        # Skip temporal CLC if 2023 data is missing
        if name == "CLC" and comp_t.empty:
            temporal_psi = float("nan")
            temporal_lbl = "N/A"
            print(f"  {name}: 2023 CLC not found — temporal PSI skipped")
        else:
            bins_t       = make_bins(base_t)
            temporal_psi, detail_t = psi(base_t.values, comp_t.values, bins_t)
            temporal_lbl = stability_label(temporal_psi)
            detail_blocks.append((f"{name} — Temporal (2019 baseline → 2023)", detail_t))

        bins_s       = make_bins(base_s)
        subgroup_psi, detail_s = psi(base_s.values, comp_s.values, bins_s)
        subgroup_lbl = stability_label(subgroup_psi)
        detail_blocks.append((f"{name} — Subgroup (white baseline → minority)", detail_s))

        print(f"{name}:")
        print(f"  Temporal PSI  = {temporal_psi:.4f}  [{temporal_lbl}]  "
              f"(N base={len(base_t)}, compare={len(comp_t)})")
        print(f"  Subgroup PSI  = {subgroup_psi:.4f}  [{subgroup_lbl}]  "
              f"(N white={len(base_s)}, minority={len(comp_s)})\n")

        rows.append({
            "Variable":        name,
            "Temporal PSI":    round(temporal_psi, 4) if not np.isnan(temporal_psi) else "N/A",
            "Temporal Label":  temporal_lbl,
            "Subgroup PSI":    round(subgroup_psi, 4),
            "Subgroup Label":  subgroup_lbl,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    results = pd.DataFrame(rows)
    csv_path = PROCESSED_DIR / "psi_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path.name}\n")

    # ── Formatted appendix text ───────────────────────────────────────────────
    out = []
    out.append("=" * 70)
    out.append("APPENDIX E — POPULATION STABILITY INDEX (PSI)")
    out.append("=" * 70)
    out.append("")
    out.append("PSI thresholds:  < 0.10 = Stable  |  0.10–0.20 = Monitor  |  ≥ 0.20 = Shift")
    out.append(f"Bins: {N_BINS} deciles based on baseline distribution")
    out.append("")
    out.append(f"  {'Variable':<10}  {'Temporal PSI':>14}  {'Label':<8}  "
               f"{'Subgroup PSI':>13}  {'Label':<8}")
    out.append("  " + "─" * 62)
    for _, row in results.iterrows():
        t_psi = f"{row['Temporal PSI']:.4f}" if row["Temporal PSI"] != "N/A" else "   N/A "
        s_psi = f"{row['Subgroup PSI']:.4f}"
        out.append(f"  {row['Variable']:<10}  {t_psi:>14}  {row['Temporal Label']:<8}  "
                   f"{s_psi:>13}  {row['Subgroup Label']:<8}")

    out.append("")
    out.append("Notes:")
    out.append("  Temporal: 2019 distribution as baseline, 2023 as comparison.")
    out.append("  Subgroup: white-majority tracts (≤50% Black+Hispanic) as baseline,")
    out.append("            minority-majority tracts (>50%) as comparison, both in 2019.")

    out.append("")
    out.append("─" * 70)
    out.append("BIN-LEVEL DETAIL")
    out.append("─" * 70)
    for label, detail in detail_blocks:
        out.append(f"\n  {label}")
        out.append(f"  {'Bin':>4}  {'Base %':>8}  {'Compare %':>10}  {'PSI contrib':>12}")
        out.append("  " + "─" * 38)
        for _, r in detail.iterrows():
            out.append(f"  {int(r['bin']):>4}  {r['base_pct']:>8.1f}  {r['comp_pct']:>10.1f}  {r['psi_bin']:>12.4f}")

    out.append("")
    out.append("─" * 70)
    out.append("INTERPRETATION")
    out.append("─" * 70)
    out.append("")
    out.append("Temporal PSI answers whether the predictor distributions shifted")
    out.append("structurally between the pre- and post-period. A Stable result")
    out.append("confirms that the 2019 snapshot used in the regression is representative")
    out.append("of the study window and not distorted by mid-period composition changes.")
    out.append("")
    out.append("Subgroup PSI answers whether minority and white tracts represent")
    out.append("meaningfully different populations in terms of exposure to each variable.")
    out.append("A Monitor or Shift result for CLC supports the H2 interaction design:")
    out.append("the two groups face structurally different CLC distributions, so a")
    out.append("pooled OLS with an interaction term is the appropriate specification.")

    txt_path = PROCESSED_DIR / "psi_results.txt"
    txt_path.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved → {txt_path.name}")


if __name__ == "__main__":
    main()

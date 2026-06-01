"""
Build the master regression dataset.

One row per census tract, joining:
  - CLC pre-year (treatment variable)
  - AHBI pre and post (secondary outcome, retained for descriptives)
  - Homeownership gap pre and post (primary outcome)
  - Pre-period ACS demographic controls

Output: data/processed/regression_master.csv

Usage:
    python src/build_regression_dataset.py                          # 2019 → 2022
    python src/build_regression_dataset.py --pre-year 2019 --post-year 2022
"""

import argparse
import csv
from pathlib import Path

import numpy as np

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pre-year",  type=int, default=2019,
                   help="Pre-period ACS year (default: 2019)")
    p.add_argument("--post-year", type=int, default=2022,
                   help="Post-period ACS year (default: 2022)")
    return p.parse_args()


def output_fields(pre_year: int, post_year: int) -> list:
    return [
        # Identifiers
        "metro", "census_tract_geoid", "county_fips", "state_fips",
        # Treatment
        f"clc_{pre_year}", "clc_winsorized", "log_clc",
        "reit_units_total", "num_reit_properties",
        # Primary outcomes — homeownership gap
        "gap_black_pre", "gap_black_post", "delta_gap_black",
        "gap_hispanic_pre", "gap_hispanic_post", "delta_gap_hispanic",
        # Secondary outcome — AHBI (retained for robustness/descriptives)
        f"ahbi_{pre_year}", f"ahbi_{post_year}", "delta_ahbi",
        # AHBI components
        f"prb_norm_{pre_year}", f"tightness_norm_{pre_year}",
        f"prb_norm_{post_year}", f"tightness_norm_{post_year}",
        # Demographic controls (pre-period ACS)
        "median_hh_income", "renter_pct", "total_housing_units",
        "pct_black", "pct_hispanic", "pct_white_nh",
        "homeownership_rate", "homeownership_rate_black",
        "homeownership_rate_hispanic", "homeownership_rate_white_nh",
        # Baseline gap control
        "ahbi_pre",
    ]


def load_csv(path: Path) -> dict:
    """Load CSV → {census_tract_geoid: row_dict}"""
    if not path.exists():
        raise SystemExit(f"Missing: {path.name}")
    return {
        r["census_tract_geoid"]: r
        for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
        if r.get("census_tract_geoid")
    }


def load_acs_demo(path: Path) -> dict:
    """Dedup ACS file to one row per tract (ACS values are tract-level)."""
    if not path.exists():
        raise SystemExit(f"Missing: {path.name}")
    result = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        g = r.get("census_tract_geoid", "")
        if g and g not in result:
            result[g] = r
    return result


def to_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_pct(num_col, den_col, row):
    n = to_float(row.get(num_col, ""))
    d = to_float(row.get(den_col, ""))
    if n is None or d is None or d == 0:
        return ""
    return f"{n / d:.6f}"


def fmt(val):
    return f"{val:.6f}" if val is not None else ""


def main():
    args      = parse_args()
    pre_year  = args.pre_year
    post_year = args.post_year
    print(f"Time window: {pre_year} → {post_year}")

    # ── Load source files ─────────────────────────────────────────────────────
    clc_pre   = load_csv(PROCESSED_DIR / f"c1_clc_{pre_year}.csv")
    ahbi_pre  = load_csv(PROCESSED_DIR / f"ahbi_{pre_year}.csv")
    ahbi_post = load_csv(PROCESSED_DIR / f"ahbi_{post_year}.csv")
    acs_demo  = load_acs_demo(PROCESSED_DIR / f"communities_with_acs_{pre_year}.csv")
    acs_post  = load_acs_demo(PROCESSED_DIR / f"communities_with_acs_{post_year}.csv")

    print(f"CLC {pre_year}:         {len(clc_pre):,} tracts")
    print(f"AHBI {pre_year}:        {len(ahbi_pre):,} tracts")
    print(f"AHBI {post_year}:       {len(ahbi_post):,} tracts")
    print(f"ACS demo (pre):         {len(acs_demo):,} tracts")
    print(f"ACS demo (post):        {len(acs_post):,} tracts")

    # ── Inner join on census_tract_geoid ──────────────────────────────────────
    geoids = sorted(set(clc_pre) & set(ahbi_pre) & set(ahbi_post) & set(acs_demo))
    print(f"\nInner join → {len(geoids):,} tracts")

    rows_out = []
    for g in geoids:
        c      = clc_pre[g]
        a_pre  = ahbi_pre[g]
        a_post = ahbi_post[g]
        d_pre  = acs_demo.get(g, {})
        d_post = acs_post.get(g, {})

        ahbi_pre_val  = to_float(a_pre.get("ahbi", ""))
        ahbi_post_val = to_float(a_post.get("ahbi", ""))
        if ahbi_pre_val is None or ahbi_post_val is None:
            continue
        delta_ahbi = ahbi_post_val - ahbi_pre_val

        # Homeownership gap outcomes
        gap_black_pre  = to_float(d_pre.get("homeownership_gap_black", ""))
        gap_black_post = to_float(d_post.get("homeownership_gap_black", ""))
        gap_hisp_pre   = to_float(d_pre.get("homeownership_gap_hispanic", ""))
        gap_hisp_post  = to_float(d_post.get("homeownership_gap_hispanic", ""))

        delta_gap_black    = (gap_black_post - gap_black_pre) \
                             if (gap_black_post is not None and gap_black_pre is not None) else None
        delta_gap_hispanic = (gap_hisp_post - gap_hisp_pre) \
                             if (gap_hisp_post is not None and gap_hisp_pre is not None) else None

        rows_out.append({
            "metro":               c.get("metro", ""),
            "census_tract_geoid":  g,
            "county_fips":         c.get("county_fips", ""),
            "state_fips":          c.get("state_fips", ""),
            # Treatment (CLC winsorized and log-transformed below)
            f"clc_{pre_year}":     c.get("clc", ""),
            "clc_winsorized":      "",   # filled below
            "log_clc":             "",   # filled below
            "reit_units_total":    c.get("reit_units_total", ""),
            "num_reit_properties": c.get("num_reit_properties", ""),
            # Primary outcomes
            "gap_black_pre":       fmt(gap_black_pre),
            "gap_black_post":      fmt(gap_black_post),
            "delta_gap_black":     fmt(delta_gap_black),
            "gap_hispanic_pre":    fmt(gap_hisp_pre),
            "gap_hispanic_post":   fmt(gap_hisp_post),
            "delta_gap_hispanic":  fmt(delta_gap_hispanic),
            # AHBI (secondary)
            f"ahbi_{pre_year}":             f"{ahbi_pre_val:.6f}",
            f"ahbi_{post_year}":            f"{ahbi_post_val:.6f}",
            "delta_ahbi":                   f"{delta_ahbi:.6f}",
            f"prb_norm_{pre_year}":         a_pre.get("prb_norm", ""),
            f"tightness_norm_{pre_year}":   a_pre.get("tightness_norm", ""),
            f"prb_norm_{post_year}":        a_post.get("prb_norm", ""),
            f"tightness_norm_{post_year}":  a_post.get("tightness_norm", ""),
            # Demographics (pre-period)
            "median_hh_income":            d_pre.get("median_hh_income", ""),
            "renter_pct":                  d_pre.get("renter_pct", ""),
            "total_housing_units":         d_pre.get("total_housing_units", ""),
            "pct_black":                   safe_pct("tenure_black_total",   "tenure_total", d_pre),
            "pct_hispanic":                safe_pct("tenure_hispanic_total", "tenure_total", d_pre),
            "pct_white_nh":                safe_pct("tenure_white_nh_total", "tenure_total", d_pre),
            "homeownership_rate":          d_pre.get("homeownership_rate", ""),
            "homeownership_rate_black":    d_pre.get("homeownership_rate_black", ""),
            "homeownership_rate_hispanic": d_pre.get("homeownership_rate_hispanic", ""),
            "homeownership_rate_white_nh": d_pre.get("homeownership_rate_white_nh", ""),
            "ahbi_pre":                    f"{ahbi_pre_val:.6f}",
        })

    # ── Winsorize CLC at 99th percentile (within analysis sample) ────────────
    clc_key    = f"clc_{pre_year}"
    clc_values = [to_float(r[clc_key]) for r in rows_out if to_float(r.get(clc_key)) is not None]
    p99 = float(np.percentile(clc_values, 99))
    p95 = float(np.percentile(clc_values, 95))
    print(f"\nCLC 99th percentile: {p99:.4f}  (raw max: {max(clc_values):.4f})")
    print(f"CLC 95th percentile: {p95:.4f}")

    for r in rows_out:
        raw = to_float(r.get(clc_key))
        if raw is not None:
            winsorized = min(raw, p99)
            r["clc_winsorized"] = f"{winsorized:.6f}"
            r["log_clc"]        = f"{np.log1p(winsorized):.6f}"

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "regression_master.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields(pre_year, post_year))
        writer.writeheader()
        writer.writerows(rows_out)

    # ── Summary ───────────────────────────────────────────────────────────────
    clcs    = [float(r[clc_key]) for r in rows_out if r.get(clc_key)]
    deltas  = [float(r["delta_ahbi"]) for r in rows_out]
    gap_b   = [float(r["delta_gap_black"])    for r in rows_out if r["delta_gap_black"]]
    gap_h   = [float(r["delta_gap_hispanic"]) for r in rows_out if r["delta_gap_hispanic"]]

    print(f"\n{'─'*50}")
    print(f"regression_master.csv  ({pre_year} → {post_year})")
    print(f"{'─'*50}")
    print(f"Rows:                    {len(rows_out)}")
    print(f"{clc_key} mean:          {sum(clcs)/len(clcs):.4f}")
    print(f"delta_ahbi mean:         {sum(deltas)/len(deltas):.4f}")
    print(f"delta_gap_black mean:    {sum(gap_b)/len(gap_b):.4f}  (n={len(gap_b)})")
    print(f"delta_gap_hispanic mean: {sum(gap_h)/len(gap_h):.4f}  (n={len(gap_h)})")

    from collections import Counter
    metro_counts = Counter(r["metro"] for r in rows_out)
    print(f"\n{'Metro':<22}  {'Tracts':>6}")
    print("─" * 32)
    for metro, n in sorted(metro_counts.items()):
        print(f"  {metro:<20}  {n:>6}")

    print(f"\nWrote {len(rows_out)} rows → {out_path.name}")


if __name__ == "__main__":
    main()

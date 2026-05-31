"""
Compute AHBI (Affordable Housing Burden Index) composite score.

AHBI measures pre-existing housing market stress at the census tract level.
It is a two-component index:

    Component 2 — PRB  (Pre-existing Rent Burden):  rent_burden_30plus_pct
    Component 3 — HMT  (Housing Market Tightness):  1 - vacancy_rate

CLC (Corporate Landlord Concentration) is intentionally excluded from AHBI.
It serves as the independent variable in the analysis:

    Hypothesis: tracts with higher CLC in 2018 experienced greater increases
    in AHBI between 2018 and 2023, controlling for pre-existing conditions.

Method:
    1. Z-score each component (fit mean/SD on 2022 reference data so scores
       are comparable across years)
    2. Average the two z-scores → raw AHBI
    3. Re-normalize to [0, 1] using the 2022 reference distribution

Usage:
    python src/compute_ahbi.py [--year 2022]
    python src/compute_ahbi.py --year 2018
    python src/compute_ahbi.py --year 2023
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
NORM_YEAR     = 2022   # reference year for z-score fit

OUTPUT_FIELDS = [
    "metro", "census_tract_geoid", "county_fips", "state_fips",
    "prb_norm", "tightness_norm", "ahbi", "acs_year",
]


def parse_args():
    p = argparse.ArgumentParser(description="Compute AHBI composite index (PRB + Tightness)")
    p.add_argument("--year", type=int, default=2022,
                   help="ACS vintage year to score (default: 2022)")
    return p.parse_args()


def load_component(name: str, year: int) -> dict:
    path = PROCESSED_DIR / f"{name}_{year}.csv"
    if not path.exists():
        raise SystemExit(f"Missing: {path.name}  —  run compute_{name}.py --year {year}")
    return {r["census_tract_geoid"]: r
            for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r.get("census_tract_geoid")}


def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_matrix(prb: dict, tightness: dict) -> tuple[list, np.ndarray]:
    """Inner join PRB and Tightness, return (geoids, matrix)."""
    geoids = sorted(set(prb) & set(tightness))
    mat = np.array([
        [to_float(prb[g].get("rent_burden_30plus_pct", "")),
         to_float(tightness[g].get("tightness", ""))]
        for g in geoids
    ], dtype=float)
    # Drop rows with any missing value
    valid = ~np.isnan(mat).any(axis=1)
    return [g for g, ok in zip(geoids, valid) if ok], mat[valid]


def main():
    args = parse_args()
    year = args.year

    # ── 1. Load components ────────────────────────────────────────────────────
    prb       = load_component("c2_prb",       year)
    tightness = load_component("c3_tightness", year)
    geoids, mat = build_matrix(prb, tightness)
    print(f"Tracts (PRB ∩ Tightness, {year}): {len(geoids)}")

    # ── 2. Fit z-score parameters on 2022 reference data ─────────────────────
    if year != NORM_YEAR:
        prb22  = load_component("c2_prb",       NORM_YEAR)
        tight22= load_component("c3_tightness", NORM_YEAR)
        _, ref_mat = build_matrix(prb22, tight22)
    else:
        ref_mat = mat

    ref_mean = ref_mat.mean(axis=0)
    ref_std  = ref_mat.std(axis=0)
    ref_std[ref_std == 0] = 1   # guard against zero std

    # ── 3. Z-score and compute AHBI ───────────────────────────────────────────
    z         = (mat - ref_mean) / ref_std          # z-score each component
    raw_ahbi  = z.mean(axis=1)                      # equal-weight average

    # Re-normalize to [0, 1] using 2022 reference distribution
    ref_z     = (ref_mat - ref_mean) / ref_std
    ref_ahbi  = ref_z.mean(axis=1)
    ahbi_min, ahbi_max = ref_ahbi.min(), ref_ahbi.max()
    ahbi      = (raw_ahbi - ahbi_min) / (ahbi_max - ahbi_min)
    ahbi      = np.clip(ahbi, 0, None)              # floor at 0; allow >1 for other years

    # ── 4. Write output ───────────────────────────────────────────────────────
    out_file = PROCESSED_DIR / f"ahbi_{year}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for i, g in enumerate(geoids):
            writer.writerow({
                "metro":              prb[g].get("metro", ""),
                "census_tract_geoid": g,
                "county_fips":        prb[g].get("county_fips", ""),
                "state_fips":         prb[g].get("state_fips", ""),
                "prb_norm":           f"{z[i, 0]:.6f}",
                "tightness_norm":     f"{z[i, 1]:.6f}",
                "ahbi":               f"{ahbi[i]:.6f}",
                "acs_year":           year,
            })

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"AHBI  (ACS year {year})  —  PRB + Tightness, equal-weighted z-score")
    print(f"{'─'*50}")
    print(f"Range:   {ahbi.min():.4f} – {ahbi.max():.4f}")
    print(f"Median:  {np.median(ahbi):.4f}")
    print(f"Mean:    {ahbi.mean():.4f}")
    if year != NORM_YEAR:
        above_1 = (ahbi > 1.0).sum()
        print(f"Tracts above 2022 ceiling (AHBI > 1): {above_1}")

    print(f"\n{'Metro':<22}  {'Tracts':>6}  {'AvgAHBI':>8}  {'MedianAHBI':>10}")
    print("─" * 52)
    metro_scores = defaultdict(list)
    for i, g in enumerate(geoids):
        metro_scores[prb[g].get("metro", "")].append(ahbi[i])
    for metro in sorted(metro_scores):
        vals = sorted(metro_scores[metro])
        print(f"  {metro:<20}  {len(vals):>6}  "
              f"{sum(vals)/len(vals):>8.4f}  {vals[len(vals)//2]:>10.4f}")

    print(f"\nWrote {len(geoids)} rows → {out_file.name}")
    print(f"\nNext step: join ahbi_{year}.csv with c1_clc_{year}.csv on")
    print(f"census_tract_geoid to get CLC + AHBI in one table for analysis.")


if __name__ == "__main__":
    main()

"""
Compute CLC (Corporate Landlord Concentration) — AHBI Component 1.

CLC_tract = Σ(REIT-owned units in tract) / ACS B25003 renter_occupied

Steps
─────
1. Load communities_with_acs.csv
2. Filter to properties whose county_fips falls within the 10 target metros
3. Use the most recent filing year per property to avoid double-counting
4. Aggregate REIT unit counts to census-tract level
5. Divide by ACS renter_occupied (B25003_003E) → CLC ratio
6. Output two files:
     clc_by_tract.csv   — one row per tract (primary analytic unit)
     clc_by_property.csv — original property rows with metro + CLC attached

Usage:
    python src/compute_clc.py
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Metro → county FIPS mapping
# ---------------------------------------------------------------------------

METRO_COUNTIES = {
    "Los Angeles":       ["06037", "06059"],
    "New York":          ["36061", "36047", "36081", "36005", "36085",
                          "36059", "36103", "36119",
                          "34003", "34013", "34017", "34019", "34023",
                          "34025", "34029", "34035", "34039"],
    "Seattle":           ["53033", "53053", "53061"],
    "San Francisco":     ["06001", "06013", "06041", "06075", "06081"],
    "Washington DC":     ["11001", "51013", "51510", "51059", "51153",
                          "51179", "24031", "24033"],
    "Dallas-Fort Worth": ["48113", "48085", "48121", "48139", "48251", "48439"],
    "San Jose":          ["06085"],
    "San Diego":         ["06073"],
    "Atlanta":           ["13121", "13135", "13089", "13067", "13063",
                          "13057", "13151"],
    "Charlotte":         ["37119", "37025", "37179", "37071", "37097"],
}

# Reverse lookup: county_fips → metro_name
COUNTY_TO_METRO = {
    county: metro
    for metro, counties in METRO_COUNTIES.items()
    for county in counties
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "communities_with_acs.csv"
OUT_TRACT     = PROCESSED_DIR / "clc_by_tract.csv"
OUT_PROPERTY  = PROCESSED_DIR / "clc_by_property.csv"

TRACT_FIELDS = [
    "metro",
    "census_tract_geoid",
    "census_tract_name",
    "county_fips",
    "county_name",
    "state_fips",
    # REIT unit totals
    "reit_units_total",
    "num_reit_properties",
    "tickers_present",
    # ACS denominator
    "acs_renter_occupied",
    # CLC
    "clc",
    # Supporting ACS context
    "acs_tenure_total",
    "acs_renter_pct",
    "acs_median_hh_income",
    "acs_vacancy_rate",
    "acs_rent_burden_30plus_pct",
    "acs_year",
]

PROPERTY_FIELDS = [
    "metro",
    "ticker", "filing_year", "community_name", "city", "state",
    "unit_count", "formatted_address", "lat", "lng",
    "census_tract_geoid", "census_tract_name",
    "county_fips", "county_name",
    # tract-level CLC for easy joining
    "clc",
    "reit_units_total",
    "acs_renter_occupied",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_float(val) -> Optional[float]:
    try:
        f = float(val)
        return None if f < 0 else f   # Census sentinel values are negative
    except (TypeError, ValueError):
        return None


def safe_div(num, den) -> str:
    if num is None or den is None or den == 0:
        return ""
    return f"{num / den:.6f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"{INPUT_FILE.name} not found. Run get_acs_data.py first.")

    all_rows = list(csv.DictReader(open(INPUT_FILE, newline="", encoding="utf-8")))
    print(f"Loaded {len(all_rows):,} rows from {INPUT_FILE.name}")

    # ── 1. Filter to metro counties ──────────────────────────────────────────
    # county_fips in the CSV is 3-digit; METRO_COUNTIES uses 5-digit (state+county)
    def full_fips(row) -> str:
        return row.get("state_fips", "") + row.get("county_fips", "")

    metro_rows = [
        r for r in all_rows
        if COUNTY_TO_METRO.get(full_fips(r))
    ]
    print(f"In target metros: {len(metro_rows):,} rows "
          f"({len(all_rows) - len(metro_rows):,} outside metros dropped)")

    # Attach metro name
    for r in metro_rows:
        r["metro"] = COUNTY_TO_METRO[full_fips(r)]

    # ── 2. Deduplicate: keep most recent filing year per property ─────────────
    # Key = (community_name, city, state) — same physical building across years
    latest: dict = {}
    for r in metro_rows:
        key = (r["community_name"].lower().strip(),
               r["city"].lower().strip(),
               r["state"].lower().strip())
        if key not in latest or r["filing_year"] > latest[key]["filing_year"]:
            latest[key] = r

    deduped = list(latest.values())
    print(f"After dedup (most recent year per property): {len(deduped):,} unique properties")

    # Warn about missing unit counts
    missing_units = [r for r in deduped if not r.get("unit_count")]
    if missing_units:
        print(f"  Warning: {len(missing_units)} properties have no unit_count "
              f"— excluded from CLC numerator")

    # ── 3. Aggregate to tract level ───────────────────────────────────────────
    tract_units:   dict[str, float]       = defaultdict(float)
    tract_props:   dict[str, int]         = defaultdict(int)
    tract_tickers: dict[str, set]         = defaultdict(set)
    tract_meta:    dict[str, dict]        = {}

    for r in deduped:
        geoid = r.get("census_tract_geoid", "")
        if not geoid:
            continue

        units = to_float(r.get("unit_count"))
        if units:
            tract_units[geoid] += units

        tract_props[geoid] += 1
        tract_tickers[geoid].add(r.get("ticker", ""))

        if geoid not in tract_meta:
            tract_meta[geoid] = {
                "metro":              r.get("metro", ""),
                "census_tract_geoid": geoid,
                "census_tract_name":  r.get("census_tract_name", ""),
                "county_fips":        r.get("county_fips", ""),
                "county_name":        r.get("county_name", ""),
                "state_fips":         r.get("state_fips", ""),
                "acs_renter_occupied":       r.get("renter_occupied", ""),
                "acs_tenure_total":          r.get("tenure_total", ""),
                "acs_renter_pct":            r.get("renter_pct", ""),
                "acs_median_hh_income":      r.get("median_hh_income", ""),
                "acs_vacancy_rate":          r.get("vacancy_rate", ""),
                "acs_rent_burden_30plus_pct": r.get("rent_burden_30plus_pct", ""),
                "acs_year":                  r.get("acs_year", ""),
            }

    # ── 4. Compute CLC ────────────────────────────────────────────────────────
    tract_clc: dict[str, str] = {}
    for geoid, meta in tract_meta.items():
        reit_units     = tract_units.get(geoid, 0.0)
        renter_occupied = to_float(meta["acs_renter_occupied"])
        clc = safe_div(reit_units, renter_occupied)
        tract_clc[geoid] = clc

    # ── 5. Write tract-level output ───────────────────────────────────────────
    tract_rows_out = []
    for geoid, meta in sorted(tract_meta.items()):
        tract_rows_out.append({
            **meta,
            "reit_units_total":    tract_units.get(geoid, ""),
            "num_reit_properties": tract_props.get(geoid, ""),
            "tickers_present":     "|".join(sorted(tract_tickers.get(geoid, set()))),
            "clc":                 tract_clc.get(geoid, ""),
        })

    with open(OUT_TRACT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACT_FIELDS)
        writer.writeheader()
        writer.writerows(tract_rows_out)

    # ── 6. Write property-level output with CLC attached ─────────────────────
    with open(OUT_PROPERTY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROPERTY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(deduped, key=lambda x: (x.get("metro",""), x.get("census_tract_geoid",""))):
            geoid = r.get("census_tract_geoid", "")
            writer.writerow({
                **r,
                "clc":              tract_clc.get(geoid, ""),
                "reit_units_total": tract_units.get(geoid, ""),
                "acs_renter_occupied": tract_meta.get(geoid, {}).get("acs_renter_occupied", ""),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_clc = [r for r in tract_rows_out if r["clc"]]
    clc_vals  = [float(r["clc"]) for r in valid_clc]

    print(f"\n{'─'*50}")
    print(f"CLC results")
    print(f"{'─'*50}")
    print(f"Tracts with ≥1 REIT property:  {len(tract_rows_out)}")
    print(f"Tracts with valid CLC:         {len(valid_clc)}")
    if clc_vals:
        print(f"CLC range:   {min(clc_vals):.4f} – {max(clc_vals):.4f}")
        print(f"CLC median:  {sorted(clc_vals)[len(clc_vals)//2]:.4f}")
        print(f"CLC mean:    {sum(clc_vals)/len(clc_vals):.4f}")
        high_clc = [(r['census_tract_geoid'], r['metro'], float(r['clc']))
                    for r in valid_clc]
        high_clc.sort(key=lambda x: -x[2])
        print(f"\nTop 10 tracts by CLC:")
        for geoid, metro, clc in high_clc[:10]:
            print(f"  {geoid}  {metro:<22}  CLC={clc:.4f}")

    print(f"\nWrote {len(tract_rows_out)} rows → {OUT_TRACT.name}")
    print(f"Wrote {len(deduped)} rows    → {OUT_PROPERTY.name}")

    # Per-metro summary
    print(f"\n{'Metro':<22}  {'Tracts':>6}  {'Properties':>10}  {'REIT Units':>10}  {'Avg CLC':>8}")
    print("─" * 65)
    metro_summary: dict[str, dict] = defaultdict(lambda: {"tracts":0,"props":0,"units":0,"clcs":[]})
    for r in tract_rows_out:
        m = r["metro"]
        metro_summary[m]["tracts"] += 1
        metro_summary[m]["props"]  += int(r["num_reit_properties"] or 0)
        metro_summary[m]["units"]  += float(r["reit_units_total"] or 0)
        if r["clc"]:
            metro_summary[m]["clcs"].append(float(r["clc"]))

    for metro in sorted(metro_summary):
        ms   = metro_summary[metro]
        avg  = sum(ms["clcs"]) / len(ms["clcs"]) if ms["clcs"] else float("nan")
        print(f"  {metro:<20}  {ms['tracts']:>6}  {ms['props']:>10}  "
              f"{int(ms['units']):>10,}  {avg:>8.4f}")


if __name__ == "__main__":
    main()

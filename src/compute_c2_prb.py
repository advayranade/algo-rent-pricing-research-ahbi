"""
Compute Component 2 — PRB (Pre-existing Rent Burden).

PRB_tract = rent_burden_30plus_pct (ACS B25070, already computed in input file)

Steps
─────
1. Load communities_with_acs_{year}.csv
2. Filter to properties in the 10 target metro counties
3. Dedup to one row per census_tract_geoid (ACS values are tract-level and
   identical across all property rows sharing a geoid)
4. Write data/processed/c2_prb_{year}.csv

Usage:
    python src/compute_c2_prb.py [--year 2022]
"""

import argparse
import csv
from pathlib import Path

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

OUTPUT_FIELDS = [
    "metro",
    "census_tract_geoid",
    "county_fips",
    "state_fips",
    "rent_burden_30plus_pct",
    "acs_year",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute AHBI Component 2: PRB")
    p.add_argument("--year", type=int, default=2022,
                   help="ACS vintage year (default: 2022)")
    return p.parse_args()


def full_fips(row) -> str:
    return row.get("state_fips", "") + row.get("county_fips", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    year       = args.year
    input_file = PROCESSED_DIR / f"communities_with_acs_{year}.csv"
    out_file   = PROCESSED_DIR / f"c2_prb_{year}.csv"

    if not input_file.exists():
        raise SystemExit(
            f"{input_file.name} not found.\n"
            f"Run: python src/get_acs_data.py --year {year}"
        )

    all_rows = list(csv.DictReader(open(input_file, newline="", encoding="utf-8")))
    print(f"Loaded {len(all_rows):,} rows from {input_file.name}")

    # ── 1. Filter to metro counties ──────────────────────────────────────────
    metro_rows = [r for r in all_rows if COUNTY_TO_METRO.get(full_fips(r))]
    for r in metro_rows:
        r["metro"] = COUNTY_TO_METRO[full_fips(r)]
    print(f"In target metros: {len(metro_rows):,} rows "
          f"({len(all_rows) - len(metro_rows):,} outside metros dropped)")

    # ── 2. Dedup to one row per tract ─────────────────────────────────────────
    seen: dict[str, dict] = {}
    for r in metro_rows:
        geoid = r.get("census_tract_geoid", "")
        if geoid and geoid not in seen:
            seen[geoid] = r

    rows_out = []
    for geoid, r in sorted(seen.items()):
        rows_out.append({
            "metro":                 r["metro"],
            "census_tract_geoid":    geoid,
            "county_fips":           r.get("county_fips", ""),
            "state_fips":            r.get("state_fips", ""),
            "rent_burden_30plus_pct": r.get("rent_burden_30plus_pct", ""),
            "acs_year":              r.get("acs_year", ""),
        })

    # ── 3. Write output ───────────────────────────────────────────────────────
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    # ── Summary ───────────────────────────────────────────────────────────────
    values = [float(r["rent_burden_30plus_pct"])
              for r in rows_out if r["rent_burden_30plus_pct"]]
    missing = sum(1 for r in rows_out if not r["rent_burden_30plus_pct"])

    print(f"\n{'─'*50}")
    print(f"C2 PRB  (ACS year {year})")
    print(f"{'─'*50}")
    print(f"Tracts:              {len(rows_out)}")
    print(f"Missing (suppressed): {missing}")
    if values:
        sv = sorted(values)
        print(f"PRB range:           {sv[0]:.4f} – {sv[-1]:.4f}")
        print(f"PRB median:          {sv[len(sv)//2]:.4f}")
        print(f"PRB mean:            {sum(sv)/len(sv):.4f}")

    print(f"\n{'Metro':<22}  {'Tracts':>6}  {'AvgPRB':>8}")
    print("─" * 40)
    from collections import defaultdict
    metro_stats: dict = defaultdict(lambda: {"tracts": 0, "vals": []})
    for r in rows_out:
        m = r["metro"]
        metro_stats[m]["tracts"] += 1
        if r["rent_burden_30plus_pct"]:
            metro_stats[m]["vals"].append(float(r["rent_burden_30plus_pct"]))
    for metro in sorted(metro_stats):
        ms  = metro_stats[metro]
        avg = sum(ms["vals"]) / len(ms["vals"]) if ms["vals"] else float("nan")
        print(f"  {metro:<20}  {ms['tracts']:>6}  {avg:>8.4f}")

    print(f"\nWrote {len(rows_out)} rows → {out_file.name}")


if __name__ == "__main__":
    main()

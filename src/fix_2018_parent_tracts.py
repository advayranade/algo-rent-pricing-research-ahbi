"""
Second-pass fix for ACS nulls: parent-tract fallback.

After fix_2018_crosswalk.py runs, some metro-county GEOIDs remain unresolved.
These are inter-decennial ACS tract splits — tracts created between the 2010 and
2020 census that appear in the 2022/2023 ACS but have no 2010 predecessor in the
2020→2010 relationship file.  They were geocoded as e.g. 53033024703 but the
target-year ACS only has 53033024702 (the pre-split parent).

Fix: for each missing GEOID, find the nearest lower-numbered sibling in the same
county that IS in the cache and use its ACS data as a proxy.

Usage:
    python src/fix_2018_parent_tracts.py            # defaults to --year 2018
    python src/fix_2018_parent_tracts.py --year 2019
"""

import argparse
import json
import csv
import subprocess
import sys
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CACHE_FILE    = PROCESSED_DIR / "acs_cache.json"
TRACTS_FILE   = PROCESSED_DIR / "communities_with_tracts.csv"

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
ALL_METRO_FIPS = {fips for fips_list in METRO_COUNTIES.values() for fips in fips_list}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2018,
                   help="ACS vintage year to fix (default: 2018)")
    return p.parse_args()


def main():
    year = parse_args().year
    print(f"Fixing parent-tract fallback for year {year}...\n")

    cache  = json.load(open(CACHE_FILE, encoding="utf-8"))
    tracts = list(csv.DictReader(open(TRACTS_FILE, newline="", encoding="utf-8")))

    all_geoids = {r["census_tract_geoid"] for r in tracts if r.get("census_tract_geoid")}

    flat_year = {
        g: rec
        for key, recs in cache.items()
        if key.startswith(f"{year}|") and isinstance(recs, dict)
        for g, rec in recs.items()
    }

    still_missing = all_geoids - set(flat_year.keys())
    metro_missing  = {g for g in still_missing if (g[:2] + g[2:5]) in ALL_METRO_FIPS}
    print(f"  Metro GEOIDs still missing from {year} cache: {len(metro_missing)}")

    # ── Find parent-tract proxies ─────────────────────────────────────────────
    parent_map: dict[str, str] = {}
    for g in metro_missing:
        county_cache = cache.get(f"{year}|{g[:2]}|{g[2:5]}", {})
        for suffix in range(int(g[9:]) - 1, -1, -1):
            candidate = g[:9] + f"{suffix:02d}"
            if candidate in county_cache:
                parent_map[g] = candidate
                break

    resolved   = len(parent_map)
    unresolved = len(metro_missing) - resolved
    print(f"  Resolved via parent-tract fallback: {resolved}")
    print(f"  Truly unresolvable (remain null):   {unresolved}")

    # ── Inject parent records under the child GEOID ───────────────────────────
    print("\nInjecting parent-tract ACS records into cache...")
    for g_child, g_parent in parent_map.items():
        state, county = g_child[:2], g_child[2:5]
        ck = f"{year}|{state}|{county}"
        if ck not in cache:
            cache[ck] = {}
        cache[ck][g_child] = dict(flat_year[g_parent])

    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), indent=2)

    new_flat = {
        g: rec
        for key, recs in cache.items()
        if key.startswith(f"{year}|") and isinstance(recs, dict)
        for g, rec in recs.items()
    }
    print(f"  Cache records (before): {len(flat_year)}")
    print(f"  Cache records (after):  {len(new_flat)}")

    # ── Regenerate communities_with_acs_{year}.csv ────────────────────────────
    print(f"\nRegenerating communities_with_acs_{year}.csv...")
    result = subprocess.run(
        [sys.executable, "get_acs_data.py", "--year", str(year)],
        capture_output=True, text=True
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise SystemExit("get_acs_data.py failed")

    # ── Verify ────────────────────────────────────────────────────────────────
    print("\nVerification:")
    out_csv = PROCESSED_DIR / f"communities_with_acs_{year}.csv"
    rows_out = list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    nulls_ro = sum(1 for r in rows_out if not r.get("renter_occupied", ""))
    nulls_rb = sum(1 for r in rows_out if not r.get("rent_burden_30plus_pct", ""))
    nulls_vr = sum(1 for r in rows_out if not r.get("vacancy_rate", ""))
    print(f"  Null renter_occupied:        {nulls_ro:4d} / {len(rows_out)}")
    print(f"  Null rent_burden_30plus_pct: {nulls_rb:4d} / {len(rows_out)}")
    print(f"  Null vacancy_rate:           {nulls_vr:4d} / {len(rows_out)}")

    metro_rows = [r for r in rows_out
                  if (r.get("state_fips","") + r.get("county_fips","")) in ALL_METRO_FIPS]
    m_nulls_ro = sum(1 for r in metro_rows if not r.get("renter_occupied", ""))
    m_nulls_rb = sum(1 for r in metro_rows if not r.get("rent_burden_30plus_pct", ""))
    print(f"\n  Metro-only ({len(metro_rows)} rows):")
    print(f"  Null renter_occupied:        {m_nulls_ro}")
    print(f"  Null rent_burden_30plus_pct: {m_nulls_rb}")


if __name__ == "__main__":
    main()

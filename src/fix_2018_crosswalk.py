"""
Fix ACS null data caused by 2020→2010 census tract boundary mismatch.

The communities were geocoded against 2020 tract definitions, but pre-2020 ACS
vintages only publish data under 2010 tract GEOIDs. Any 2020 tract that was
created or renumbered in the 2020 redistricting will be missing from the cache.

This script:
  1. Identifies the missing 2020-definition GEOIDs in the target year's cache
  2. Uses the Census 2020→2010 tract relationship file to find 2010 predecessors
  3. Looks up 2010 tract records already in the ACS cache (all counties were
     fetched by get_acs_data.py, so 2010 tracts are already there — no new API
     calls needed in most cases)
  4. Interpolates raw counts using area-overlap weights, then recomputes derived
     rates (never average rates directly)
  5. Injects the synthetic 2020-keyed records back into the cache under the
     correct "{year}|state|county" key
  6. Writes the updated cache and re-runs get_acs_data.py --year {year}

Usage:
    python src/fix_2018_crosswalk.py            # defaults to --year 2018
    python src/fix_2018_crosswalk.py --year 2019
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2018,
                   help="ACS vintage year to fix (default: 2018)")
    return p.parse_args()

# YEAR is set in main() from args; module-level references below use the global
YEAR = 2018  # overridden in main()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR       = Path(__file__).parent.parent
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
RAW_DIR        = BASE_DIR / "data" / "raw"

CACHE_FILE        = PROCESSED_DIR / "acs_cache.json"
TRACTS_FILE       = PROCESSED_DIR / "communities_with_tracts.csv"
RELATIONSHIP_FILE = RAW_DIR / "tab20_tract20_tract10_natl.txt"

REQUEST_DELAY = 0.25

# ---------------------------------------------------------------------------
# ACS variables — must stay in sync with get_acs_data.py
# ---------------------------------------------------------------------------

VARIABLES = {
    "B25003_001E": "tenure_total",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
    "B25003A_001E": "tenure_white_total",
    "B25003A_002E": "owner_occupied_white",
    "B25003A_003E": "renter_occupied_white",
    "B25003B_001E": "tenure_black_total",
    "B25003B_002E": "owner_occupied_black",
    "B25003B_003E": "renter_occupied_black",
    "B25003H_001E": "tenure_white_nh_total",
    "B25003H_002E": "owner_occupied_white_nh",
    "B25003H_003E": "renter_occupied_white_nh",
    "B25003I_001E": "tenure_hispanic_total",
    "B25003I_002E": "owner_occupied_hispanic",
    "B25003I_003E": "renter_occupied_hispanic",
    "B25070_001E": "rent_burden_universe",
    "B25070_002E": "rent_lt_10pct",
    "B25070_003E": "rent_10_14pct",
    "B25070_004E": "rent_15_19pct",
    "B25070_005E": "rent_20_24pct",
    "B25070_006E": "rent_25_29pct",
    "B25070_007E": "rent_30_34pct",
    "B25070_008E": "rent_35_39pct",
    "B25070_009E": "rent_40_49pct",
    "B25070_010E": "rent_50plus_pct",
    "B25070_011E": "rent_burden_not_computed",
    "B25002_001E": "occupancy_total",
    "B25002_002E": "occupied_units",
    "B25002_003E": "vacant_units",
    "B19013_001E": "median_hh_income",
    "B25001_001E": "total_housing_units",
}

# Columns that are true counts and can be summed with area weights.
# median_hh_income is a median — handle separately as weighted average.
COUNT_COLS = [c for c in VARIABLES.values() if c != "median_hh_income"]

SENTINEL_VALUES = {"-666666666", "-999999999", "-888888888"}


# ---------------------------------------------------------------------------
# Derived metrics — must stay in sync with get_acs_data.py
# ---------------------------------------------------------------------------

def safe_rate(numerator, denominator) -> str:
    try:
        n, d = float(numerator), float(denominator)
        if d <= 0:
            return ""
        return f"{n / d:.6f}"
    except (TypeError, ValueError):
        return ""


def compute_derived(r: dict) -> dict:
    renter_pct   = safe_rate(r.get("renter_occupied"),   r.get("tenure_total"))
    vacancy_rate = safe_rate(r.get("vacant_units"),       r.get("occupancy_total"))

    burden_30 = sum(
        float(r[c]) for c in
        ["rent_30_34pct", "rent_35_39pct", "rent_40_49pct", "rent_50plus_pct"]
        if r.get(c) not in ("", None)
    )
    rb_30 = safe_rate(burden_30, r.get("rent_burden_universe"))

    burden_50 = float(r["rent_50plus_pct"]) if r.get("rent_50plus_pct") not in ("", None) else 0.0
    rb_50 = safe_rate(burden_50, r.get("rent_burden_universe"))

    ho_overall  = safe_rate(r.get("owner_occupied"),          r.get("tenure_total"))
    ho_white_nh = safe_rate(r.get("owner_occupied_white_nh"), r.get("tenure_white_nh_total"))
    ho_black    = safe_rate(r.get("owner_occupied_black"),    r.get("tenure_black_total"))
    ho_hispanic = safe_rate(r.get("owner_occupied_hispanic"), r.get("tenure_hispanic_total"))

    def gap(a, b):
        try:
            return f"{float(a) - float(b):.6f}"
        except (ValueError, TypeError):
            return ""

    return {
        "renter_pct":                  renter_pct,
        "vacancy_rate":                vacancy_rate,
        "rent_burden_30plus_pct":      rb_30,
        "rent_burden_50plus_pct":      rb_50,
        "homeownership_rate":          ho_overall,
        "homeownership_rate_white_nh": ho_white_nh,
        "homeownership_rate_black":    ho_black,
        "homeownership_rate_hispanic": ho_hispanic,
        "homeownership_gap_black":     gap(ho_white_nh, ho_black),
        "homeownership_gap_hispanic":  gap(ho_white_nh, ho_hispanic),
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def county_cache_key(state: str, county: str) -> str:
    return f"{YEAR}|{state}|{county}"


def geoid_to_state_county(geoid: str):
    """Split 11-char GEOID into (state_2, county_3)."""
    return geoid[:2], geoid[2:5]


def load_flat_year(cache: dict) -> dict:
    """Merge all target-year county records into one flat {geoid: record} dict."""
    flat = {}
    for key, records in cache.items():
        if key.startswith(f"{YEAR}|") and isinstance(records, dict):
            flat.update(records)
    return flat


# ---------------------------------------------------------------------------
# Fetch a single county from the Census API (fallback for 2010 tracts not cached)
# ---------------------------------------------------------------------------

def fetch_county_year(state: str, county: str, api_key: str) -> dict:
    acs_url = f"https://api.census.gov/data/{YEAR}/acs/acs5"
    var_str = ",".join(VARIABLES.keys())
    params  = {
        "get": f"NAME,{var_str}",
        "for": "tract:*",
        "in":  f"state:{state} county:{county}",
    }
    if api_key:
        params["key"] = api_key

    resp = requests.get(acs_url, params=params, timeout=20)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()

    rows   = resp.json()
    header = rows[0]
    data   = rows[1:]
    idx    = {col: i for i, col in enumerate(header)}

    result = {}
    for row in data:
        sc    = row[idx["state"]]
        cc    = row[idx["county"]]
        tc    = row[idx["tract"]]
        geoid = sc + cc + tc
        record = {
            col_name: ("" if row[idx[api_var]] in SENTINEL_VALUES else row[idx[api_var]])
            for api_var, col_name in VARIABLES.items()
        }
        result[geoid] = record
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global YEAR
    YEAR = parse_args().year
    print(f"Fixing ACS crosswalk for year {YEAR}...\n")

    if not RELATIONSHIP_FILE.exists():
        raise SystemExit(
            f"Relationship file not found: {RELATIONSHIP_FILE}\n"
            "Download with:\n"
            "  curl -o data/raw/tab20_tract20_tract10_natl.txt \\\n"
            '  "https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/tab20_tract20_tract10_natl.txt"'
        )

    api_key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("CENSUS_API_KEY not set in .env")

    # ── 1. Find missing 2020 GEOIDs ──────────────────────────────────────────
    print(f"Step 1: Finding missing 2020 GEOIDs in {YEAR} ACS cache...")
    tracts_rows = list(csv.DictReader(open(TRACTS_FILE, newline="", encoding="utf-8")))
    all_2020_geoids = {r["census_tract_geoid"] for r in tracts_rows if r.get("census_tract_geoid")}
    print(f"  Unique 2020 GEOIDs in communities_with_tracts: {len(all_2020_geoids)}")

    cache = json.load(open(CACHE_FILE, encoding="utf-8"))
    flat_2018 = load_flat_year(cache)
    print(f"  {YEAR} tract records in cache:                  {len(flat_2018)}")

    missing_2020 = all_2020_geoids - set(flat_2018.keys())
    print(f"  Missing 2020 GEOIDs (not in {YEAR} cache):     {len(missing_2020)}")

    if not missing_2020:
        print(f"Nothing to fix — all GEOIDs already resolved in {YEAR} cache.")
        return

    # ── 2. Build crosswalk: 2020 GEOID → [(2010 GEOID, weight)] ──────────────
    print("\nStep 2: Loading Census 2020→2010 tract relationship file...")
    crosswalk: dict[str, list] = defaultdict(list)
    rel_count = 0

    with open(RELATIONSHIP_FILE, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            g20 = row["GEOID_TRACT_20"].strip()
            if g20 not in missing_2020:
                continue
            g10        = row["GEOID_TRACT_10"].strip()
            area_part  = float(row["AREALAND_PART"] or 0)
            area_20    = float(row["AREALAND_TRACT_20"] or 0)
            weight     = area_part / area_20 if area_20 > 0 else 0.0
            crosswalk[g20].append((g10, weight))
            rel_count += 1

    print(f"  Relationship rows matched to missing GEOIDs: {rel_count}")
    print(f"  Missing GEOIDs with at least one crosswalk:  {len(crosswalk)}")
    no_crosswalk = missing_2020 - set(crosswalk.keys())
    if no_crosswalk:
        print(f"  WARNING: {len(no_crosswalk)} GEOIDs have no crosswalk entry — will remain null")

    # ── 3. Fetch any 2010 tracts missing from cache (rare edge case) ──────────
    print("\nStep 3: Checking for uncached 2010 predecessor tracts...")
    needed_2010 = {g10 for preds in crosswalk.values() for g10, _ in preds}
    uncached_counties: set[tuple] = set()
    for g10 in needed_2010:
        if g10 not in flat_2018:
            state, county = geoid_to_state_county(g10)
            uncached_counties.add((state, county))

    if uncached_counties:
        print(f"  Need to fetch {len(uncached_counties)} additional county/year combos from Census API...")
        for i, (state, county) in enumerate(sorted(uncached_counties), 1):
            ck = county_cache_key(state, county)
            if ck in cache:
                flat_2018.update(cache[ck])
                continue
            print(f"  [{i}/{len(uncached_counties)}]  state={state}  county={county} ...", end=" ", flush=True)
            try:
                records = fetch_county_year(state, county, api_key)
                cache[ck] = records
                flat_2018.update(records)
                print(f"{len(records)} tracts")
            except Exception as e:
                print(f"ERROR: {e}")
                cache[ck] = {}
            time.sleep(REQUEST_DELAY)
        json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), indent=2)
        print(f"  Cache updated → {CACHE_FILE.name}")
    else:
        print(f"  All {len(needed_2010)} 2010 predecessor tracts already in cache ✓")

    # ── 4. Interpolate counts for each missing 2020 GEOID ────────────────────
    print("\nStep 4: Interpolating raw ACS counts for missing 2020 GEOIDs...")
    injected = 0
    skipped  = 0

    for g20, preds in crosswalk.items():
        state20, county20 = geoid_to_state_county(g20)

        # Normalize weights (in case area doesn't sum to 1.0 due to water area etc.)
        available = [(g10, w) for g10, w in preds if g10 in flat_2018 and flat_2018[g10]]
        if not available:
            skipped += 1
            continue

        total_w = sum(w for _, w in available)
        if total_w == 0:
            # All weights zero (can happen for fully water tracts) — use equal weights
            available = [(g10, 1.0) for g10, _ in available]
            total_w   = len(available)

        # Interpolate each raw count column.
        # w_i = AREALAND_PART_i / AREALAND_TRACT_20, so weights already express
        # each predecessor's share of the 2020 tract.  Normalizing by total_w
        # rescales to account for any predecessors whose ACS data is missing.
        # Result: area-weighted average count, treating available predecessors
        # as covering the full 2020 tract.
        interp: dict[str, str] = {}
        for col in COUNT_COLS:
            wsum = 0.0   # sum of (normalized_weight * count)
            wt   = 0.0   # sum of normalized weights where count is non-empty
            for g10, w in available:
                val = flat_2018[g10].get(col, "")
                if val not in ("", None):
                    try:
                        norm_w = w / total_w   # normalize weights to sum to 1
                        wsum  += norm_w * float(val)
                        wt    += norm_w
                    except (ValueError, TypeError):
                        pass
            # wsum / wt rescales for any cols suppressed in a subset of predecessors
            interp[col] = f"{wsum / wt:.2f}" if wt > 0 else ""

        # Weighted average for median_hh_income (approximation)
        med_num, med_den = 0.0, 0.0
        for g10, w in available:
            val = flat_2018[g10].get("median_hh_income", "")
            if val not in ("", None):
                try:
                    med_num += (w / total_w) * float(val)
                    med_den += w / total_w
                except (ValueError, TypeError):
                    pass
        interp["median_hh_income"] = f"{med_num / med_den:.0f}" if med_den > 0 else ""

        # Recompute derived rates from interpolated counts
        derived = compute_derived(interp)
        record  = {**interp, **derived}

        # Inject into cache under the 2020 GEOID key
        ck = county_cache_key(state20, county20)
        if ck not in cache:
            cache[ck] = {}
        cache[ck][g20] = record
        flat_2018[g20] = record
        injected += 1

    print(f"  Injected:  {injected} synthetic 2020-keyed records into cache")
    print(f"  Skipped:   {skipped} (no 2010 predecessor data available)")

    # ── 5. Save updated cache ─────────────────────────────────────────────────
    print("\nStep 5: Saving updated cache...")
    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"), indent=2)
    new_flat = load_flat_year(cache)
    print(f"  Cache saved → {CACHE_FILE.name}")
    print(f"  {YEAR} tract records in cache (before): {len(flat_2018) - injected}")
    print(f"  {YEAR} tract records in cache (after):  {len(new_flat)}")

    # ── 6. Regenerate communities_with_acs_{year}.csv ─────────────────────────
    print(f"\nStep 6: Regenerating communities_with_acs_{YEAR}.csv...")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "get_acs_data.py", "--year", str(YEAR)],
        capture_output=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"get_acs_data.py --year {YEAR} failed")

    # ── 7. Final null check ───────────────────────────────────────────────────
    print(f"\nStep 7: Verifying null counts in regenerated {YEAR} file...")
    out_csv = PROCESSED_DIR / f"communities_with_acs_{YEAR}.csv"
    rows_out = list(csv.DictReader(open(out_csv, newline="", encoding="utf-8")))
    nulls = sum(1 for r in rows_out if not r.get("renter_occupied", ""))
    print(f"  Null renter_occupied in {YEAR} file: {nulls}/{len(rows_out)}")
    if nulls < 50:
        print("  ✓ Null count is within acceptable range")
    else:
        print(f"  ⚠ Still {nulls} nulls — check crosswalk coverage for remaining GEOIDs")


if __name__ == "__main__":
    main()

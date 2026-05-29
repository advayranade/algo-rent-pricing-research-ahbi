"""
Pull Census ACS 5-year estimates for the census tracts that contain each
REIT community, then join the data back to communities_with_tracts.csv.

Tables pulled
─────────────
  B25003   Tenure — total renter-occupied (CLC denominator) +
           racial-iteration sub-tables A/B/H/I for homeownership gap
  B25070   Gross rent as % of household income (rent burden — AHBI #2)
  B25002   Occupancy status (vacancy rate — AHBI #3)
  B19013   Median household income (control variable)
  B25001   Total housing units (supply control)

Strategy: one API call per unique (state, county) pair pulls every tract in
that county at once — 176 calls total for the current dataset, vs 8,000+
if we queried tract-by-tract.

Usage:
    python src/get_acs_data.py [--year 2022]

    Optional: set CENSUS_API_KEY=<key> in .env for higher rate limits.
    Free tier (no key) allows 500 calls/day, which is plenty here.

Output:
    data/processed/communities_with_acs.csv
    (all columns from communities_with_tracts.csv + ACS columns + derived rates)
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "communities_with_tracts.csv"
OUTPUT_FILE   = PROCESSED_DIR / "communities_with_acs.csv"
CACHE_FILE    = PROCESSED_DIR / "acs_cache.json"

ACS_URL       = "https://api.census.gov/data/{year}/acs/acs5"
REQUEST_DELAY = 0.25   # well under Census rate limits

# ACS variable code → output column name
# All *E variables are estimates; *M equivalents exist but aren't pulled here.
VARIABLES = {
    # ── B25003: Tenure (owner vs renter) ────────────────────────────────────
    "B25003_001E": "tenure_total",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",

    # ── B25003 racial iterations — homeownership gap ─────────────────────────
    # A = White alone
    "B25003A_001E": "tenure_white_total",
    "B25003A_002E": "owner_occupied_white",
    "B25003A_003E": "renter_occupied_white",
    # B = Black or African American alone
    "B25003B_001E": "tenure_black_total",
    "B25003B_002E": "owner_occupied_black",
    "B25003B_003E": "renter_occupied_black",
    # H = White alone, not Hispanic or Latino
    "B25003H_001E": "tenure_white_nh_total",
    "B25003H_002E": "owner_occupied_white_nh",
    "B25003H_003E": "renter_occupied_white_nh",
    # I = Hispanic or Latino
    "B25003I_001E": "tenure_hispanic_total",
    "B25003I_002E": "owner_occupied_hispanic",
    "B25003I_003E": "renter_occupied_hispanic",

    # ── B25070: Gross rent as % of household income ───────────────────────────
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

    # ── B25002: Occupancy status ─────────────────────────────────────────────
    "B25002_001E": "occupancy_total",
    "B25002_002E": "occupied_units",
    "B25002_003E": "vacant_units",

    # ── B19013: Median household income ─────────────────────────────────────
    "B19013_001E": "median_hh_income",

    # ── B25001: Total housing units ──────────────────────────────────────────
    "B25001_001E": "total_housing_units",
}

# Columns that Census encodes as -666666666 / -999999999 when suppressed
SENTINEL_VALUES = {"-666666666", "-999999999", "-888888888"}

# Output columns appended to the input file
ACS_COLS = list(VARIABLES.values()) + [
    "renter_pct",            # renter_occupied / tenure_total
    "vacancy_rate",          # vacant_units / occupancy_total
    "rent_burden_30plus_pct",  # share of renters paying ≥30 % of income on rent
    "rent_burden_50plus_pct",  # share paying ≥50 %
    "homeownership_rate",    # owner_occupied / tenure_total
    "homeownership_rate_white_nh",
    "homeownership_rate_black",
    "homeownership_rate_hispanic",
    "homeownership_gap_black",    # white_nh rate − black rate
    "homeownership_gap_hispanic", # white_nh rate − hispanic rate
    "acs_year",
]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def county_cache_key(year: int, state: str, county: str) -> str:
    return f"{year}|{state}|{county}"


# ---------------------------------------------------------------------------
# Census API
# ---------------------------------------------------------------------------

def fetch_county_tracts(year: int, state: str, county: str, api_key: str) -> dict:
    """
    Pull all ACS variables for every tract in one county.
    Returns {tract_geoid: {col_name: value}} or raises on hard failure.
    """
    var_str = ",".join(VARIABLES.keys())
    params  = {
        "get": f"NAME,{var_str}",
        "for": "tract:*",
        "in":  f"state:{state} county:{county}",
    }
    if api_key:
        params["key"] = api_key

    resp = requests.get(ACS_URL.format(year=year), params=params, timeout=20)

    if resp.status_code == 404:
        return {}   # county/year combination not in this ACS vintage
    resp.raise_for_status()

    # Census returns 200 + HTML when the key is missing or invalid
    if resp.headers.get("content-type", "").startswith("text/html"):
        raise SystemExit(
            "Census API returned an HTML error page instead of data.\n"
            "A valid API key is required.\n\n"
            "  1. Get a free key at: https://api.census.gov/data/key_signup.html\n"
            "  2. Add to your .env:  CENSUS_API_KEY=your_key_here\n"
            "  3. Re-run the script."
        )

    rows = resp.json()
    header = rows[0]          # first row is the column names
    data   = rows[1:]

    # Build index: col_name → position in header
    idx = {col: i for i, col in enumerate(header)}

    result = {}
    for row in data:
        state_code  = row[idx["state"]]
        county_code = row[idx["county"]]
        tract_code  = row[idx["tract"]]
        geoid       = state_code + county_code + tract_code  # 11-char GEOID

        record = {}
        for api_var, col_name in VARIABLES.items():
            raw = row[idx[api_var]]
            record[col_name] = "" if raw in SENTINEL_VALUES else raw

        result[geoid] = record

    return result


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def safe_rate(numerator, denominator) -> str:
    """Return numerator/denominator as a rounded float string, or ''."""
    try:
        n, d = float(numerator), float(denominator)
        if d <= 0:
            return ""
        return f"{n / d:.6f}"
    except (TypeError, ValueError):
        return ""


def compute_derived(r: dict) -> dict:
    renter_pct = safe_rate(r.get("renter_occupied"), r.get("tenure_total"))

    vacancy_rate = safe_rate(r.get("vacant_units"), r.get("occupancy_total"))

    # Rent-burdened = paying ≥30 % of income on rent
    burden_30 = sum(
        float(r[c]) for c in
        ["rent_30_34pct", "rent_35_39pct", "rent_40_49pct", "rent_50plus_pct"]
        if r.get(c) not in ("", None)
    )
    rb_30 = safe_rate(burden_30, r.get("rent_burden_universe"))

    burden_50 = float(r["rent_50plus_pct"]) if r.get("rent_50plus_pct") not in ("", None) else 0.0
    rb_50 = safe_rate(burden_50, r.get("rent_burden_universe"))

    ho_overall  = safe_rate(r.get("owner_occupied"),         r.get("tenure_total"))
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
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2022,
                   help="ACS 5-year vintage year (default: 2022)")
    return p.parse_args()


def main():
    args    = parse_args()
    year    = args.year
    api_key = os.environ.get("CENSUS_API_KEY", "").strip()

    if not INPUT_FILE.exists():
        raise SystemExit(
            f"{INPUT_FILE.name} not found.\n"
            "Run get_census_tracts.py first."
        )

    all_rows = list(csv.DictReader(open(INPUT_FILE, newline="", encoding="utf-8")))
    tract_rows = [r for r in all_rows if r.get("census_tract_geoid")]

    print(f"ACS year:         {year}")
    print(f"Total rows:       {len(all_rows):,}")
    print(f"Rows with tracts: {len(tract_rows):,}")

    # Unique (state, county) pairs
    county_pairs = sorted({
        (r["state_fips"], r["county_fips"])
        for r in tract_rows
        if r["state_fips"] and r["county_fips"]
    })
    print(f"Unique counties:  {len(county_pairs)}")
    if api_key:
        print("Census API key:   set ✓")
    else:
        raise SystemExit(
            "CENSUS_API_KEY is not set — the Census API now requires a key.\n\n"
            "  1. Get a free key at: https://api.census.gov/data/key_signup.html\n"
            "  2. Add to your .env:  CENSUS_API_KEY=your_key_here\n"
            "  3. Re-run the script."
        )

    cache = load_cache()

    # Tract-level ACS lookup: geoid → {col: value}
    tract_acs: dict = {}

    to_fetch = [
        (s, c) for s, c in county_pairs
        if county_cache_key(year, s, c) not in cache
    ]
    print(f"Counties to fetch: {len(to_fetch)}  (cached: {len(county_pairs) - len(to_fetch)})\n")

    for i, (state, county) in enumerate(to_fetch, 1):
        print(f"  [{i}/{len(to_fetch)}]  state={state}  county={county} ...", end=" ", flush=True)
        try:
            records = fetch_county_tracts(year, state, county, api_key)
            cache[county_cache_key(year, state, county)] = records
            print(f"{len(records)} tracts")
        except Exception as e:
            print(f"ERROR: {e}")
            cache[county_cache_key(year, state, county)] = {}

        if i % 20 == 0:
            save_cache(cache)

        time.sleep(REQUEST_DELAY)

    if to_fetch:
        save_cache(cache)
        print(f"\nCache saved → {CACHE_FILE.name}")

    # Merge all cached county records into one flat tract lookup
    for key, records in cache.items():
        if key.startswith(f"{year}|"):
            tract_acs.update(records)

    # Write output CSV
    input_fields = list(all_rows[0].keys()) if all_rows else []
    output_fields = input_fields + ACS_COLS

    empty_acs = {col: "" for col in ACS_COLS}

    resolved = 0
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()

        for row in all_rows:
            geoid = row.get("census_tract_geoid", "")
            if geoid and geoid in tract_acs:
                acs = tract_acs[geoid]
                derived = compute_derived(acs)
                out = {**row, **acs, **derived, "acs_year": year}
                resolved += 1
            else:
                out = {**row, **empty_acs, "acs_year": year}
            writer.writerow(out)

    print(f"\nWrote {len(all_rows):,} rows → {OUTPUT_FILE.name}")
    print(f"  ACS data joined: {resolved:,}")
    print(f"  No ACS data:     {len(all_rows) - resolved:,}")


if __name__ == "__main__":
    main()

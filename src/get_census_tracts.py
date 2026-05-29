"""
Enrich communities_with_addresses.csv with Census tract data using the
U.S. Census Bureau Geocoder API (free, no key required).

For each row that has lat/lng, hits:
  https://geocoding.geo.census.gov/geocoder/geographies/coordinates
    ?x={lng}&y={lat}&benchmark=2020&vintage=2020&format=json

Results are cached in data/processed/census_cache.json so re-runs are
free and the script is safe to interrupt and resume.

Usage:
    python src/get_census_tracts.py

Output:
    data/processed/communities_with_tracts.csv
    New columns added: census_tract_geoid, census_tract_name, tract_number,
                       state_fips, county_fips, county_name, block_group,
                       census_status
"""

import csv
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INPUT_FILE    = PROCESSED_DIR / "communities_with_addresses.csv"
OUTPUT_FILE   = PROCESSED_DIR / "communities_with_tracts.csv"
CACHE_FILE    = PROCESSED_DIR / "census_cache.json"

CENSUS_URL    = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
REQUEST_DELAY = 0.1   # Census API is generous; 10 req/sec is safe

TRACT_FIELDS  = [
    "census_tract_geoid", "census_tract_name", "tract_number",
    "state_fips", "county_fips", "county_name",
    "block_group", "census_status",
]
# INPUT_FIELDS and OUTPUT_FIELDS are derived at runtime from the actual CSV header
# so new upstream columns (e.g. unit_count) pass through automatically.


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


def cache_key(lat: str, lng: str) -> str:
    # Round to 7 decimal places — Census API only uses ~5 for tract precision
    return f"{float(lat):.7f},{float(lng):.7f}"


# ---------------------------------------------------------------------------
# Census API lookup
# ---------------------------------------------------------------------------

def lookup_tract(lat: str, lng: str) -> dict:
    """
    Query the Census Geocoder for a lat/lng pair.
    Returns a dict of TRACT_FIELDS.
    """
    try:
        resp = requests.get(
            CENSUS_URL,
            params={
                "x":         lng,
                "y":         lat,
                "benchmark": "2020",
                "vintage":   "2020",
                "format":    "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return _empty_tract(f"HTTP_ERROR: {e}")
    except ValueError:
        return _empty_tract("INVALID_JSON")

    geos = data.get("result", {}).get("geographies", {})

    # --- Census Tract ---
    tracts = geos.get("Census Tracts", [])
    if not tracts:
        return _empty_tract("NO_TRACT")

    tract = tracts[0]

    # --- Block group from Census Blocks ---
    blocks = geos.get("Census Blocks", [])
    block_group = blocks[0].get("BLKGRP", "") if blocks else ""

    # --- County name ---
    counties = geos.get("Counties", [])
    county_name = counties[0].get("BASENAME", "") if counties else ""

    return {
        "census_tract_geoid": tract.get("GEOID", ""),
        "census_tract_name":  tract.get("NAME", ""),
        "tract_number":       tract.get("TRACT", ""),
        "state_fips":         tract.get("STATE", ""),
        "county_fips":        tract.get("COUNTY", ""),
        "county_name":        county_name,
        "block_group":        block_group,
        "census_status":      "OK",
    }


def _empty_tract(status: str) -> dict:
    return {field: "" for field in TRACT_FIELDS[:-1]} | {"census_status": status}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"{INPUT_FILE.name} not found.\n"
            "Run geocode_communities.py first to generate it."
        )

    all_rows = list(csv.DictReader(open(INPUT_FILE, newline="", encoding="utf-8")))
    rows_with_coords = [r for r in all_rows if r.get("lat") and r.get("lng")]

    print(f"Total rows:       {len(all_rows):,}")
    print(f"Has coordinates:  {len(rows_with_coords):,}")
    print(f"Missing coords:   {len(all_rows) - len(rows_with_coords):,}  (skipped — no lat/lng)")

    # Deduplicate by coordinate pair
    unique_coords = {}
    for row in rows_with_coords:
        key = cache_key(row["lat"], row["lng"])
        if key not in unique_coords:
            unique_coords[key] = (row["lat"], row["lng"])

    print(f"Unique coordinates to look up: {len(unique_coords):,}")

    cache = load_cache()
    cached_count   = sum(1 for k in unique_coords if k in cache)
    to_query_count = len(unique_coords) - cached_count
    print(f"Already cached: {cached_count:,} | Need to query: {to_query_count:,}")

    if to_query_count > 0:
        est_min = (to_query_count * REQUEST_DELAY) / 60
        print(f"Estimated time: ~{est_min:.1f} min\n")

    # Query Census API for anything not yet cached
    queried = 0
    for key, (lat, lng) in unique_coords.items():
        if key in cache:
            continue

        result = lookup_tract(lat, lng)
        cache[key] = result
        queried += 1

        tract_label = result["census_tract_geoid"] or result["census_status"]
        print(f"  [{queried}/{to_query_count}]  ({lat}, {lng})  →  {tract_label}")

        if queried % 100 == 0:
            save_cache(cache)

        time.sleep(REQUEST_DELAY)

    if queried > 0:
        save_cache(cache)
        print(f"\nCache saved → {CACHE_FILE.name}")

    # Write output CSV — derive fieldnames from actual input so new upstream
    # columns (e.g. unit_count) pass through automatically
    input_fields  = list(all_rows[0].keys()) if all_rows else []
    output_fields = input_fields + [f for f in TRACT_FIELDS if f not in input_fields]

    empty_tract = _empty_tract("NO_COORDS")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in all_rows:
            if row.get("lat") and row.get("lng"):
                key    = cache_key(row["lat"], row["lng"])
                result = cache.get(key, _empty_tract("NOT_IN_CACHE"))
            else:
                result = empty_tract
            writer.writerow({**row, **result})

    ok_count = sum(
        1 for r in all_rows
        if r.get("lat") and r.get("lng")
        and cache.get(cache_key(r["lat"], r["lng"]), {}).get("census_status") == "OK"
    )
    print(f"\nWrote {len(all_rows):,} rows → {OUTPUT_FILE.name}")
    print(f"  Tract resolved:  {ok_count:,}")
    print(f"  No tract:        {len(all_rows) - ok_count:,}")


if __name__ == "__main__":
    main()

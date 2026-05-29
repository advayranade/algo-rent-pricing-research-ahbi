import csv
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the project root (one level up from src/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CACHE_FILE    = PROCESSED_DIR / "places_cache.json"
OUTPUT_FILE   = PROCESSED_DIR / "communities_with_addresses.csv"

PLACES_URL    = "https://maps.googleapis.com/maps/api/place/textsearch/json"

# Seconds to wait between API calls (stay well under the 600 QPM free tier)
REQUEST_DELAY = 0.35

OUTPUT_FIELDS = [
    "ticker", "filing_year", "accession_number",
    "community_name", "city", "state",
    "formatted_address", "lat", "lng", "place_id", "api_status",
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


def cache_key(community_name: str, city: str, state: str) -> str:
    return f"{community_name.lower().strip()}|{city.lower().strip()}|{state.lower().strip()}"


# ---------------------------------------------------------------------------
# Google Places lookup
# ---------------------------------------------------------------------------

def lookup_place(community_name: str, city: str, state: str, api_key: str) -> dict:
    """
    Query Google Places Text Search for one property.
    Returns a dict with keys: formatted_address, lat, lng, place_id, api_status.
    """
    query = f"{community_name} {city} {state} apartment"

    try:
        resp = requests.get(
            PLACES_URL,
            params={"query": query, "key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return _empty_result(f"HTTP_ERROR: {e}")

    status = data.get("status", "UNKNOWN")

    if status == "OK" and data.get("results"):
        result = data["results"][0]
        loc    = result.get("geometry", {}).get("location", {})
        return {
            "formatted_address": result.get("formatted_address", ""),
            "lat":       loc.get("lat", ""),
            "lng":       loc.get("lng", ""),
            "place_id":  result.get("place_id", ""),
            "api_status": "OK",
        }

    # ZERO_RESULTS, REQUEST_DENIED, OVER_QUERY_LIMIT, etc.
    return _empty_result(status)


def _empty_result(status: str) -> dict:
    return {
        "formatted_address": "",
        "lat": "",
        "lng": "",
        "place_id": "",
        "api_status": status,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_rows() -> list:
    """Read every *_communities.csv and return combined list of dicts."""
    rows = []
    for csv_path in sorted(PROCESSED_DIR.glob("*_communities.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main():
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "GOOGLE_PLACES_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export GOOGLE_PLACES_API_KEY='your_key_here'"
        )

    all_rows = load_all_rows()

    print(f"Loaded {len(all_rows):,} total rows from {len(list(PROCESSED_DIR.glob('*_communities.csv')))} files")

    # Build deduplicated lookup set
    unique_props = {}
    for row in all_rows:
        key = cache_key(row["community_name"], row["city"], row["state"])
        if key not in unique_props:
            unique_props[key] = (row["community_name"], row["city"], row["state"])

    print(f"Unique (name, city, state) combinations: {len(unique_props):,}")

    # Load existing cache
    cache = load_cache()
    cached_count   = sum(1 for k in unique_props if k in cache)
    to_query_count = len(unique_props) - cached_count
    print(f"Already cached: {cached_count:,} | Need to query: {to_query_count:,}")

    if to_query_count > 0:
        est_seconds = to_query_count * REQUEST_DELAY
        est_minutes = est_seconds / 60
        print(f"Estimated API time: ~{est_minutes:.1f} min at {REQUEST_DELAY}s/request\n")

    # Query API for anything not yet cached
    queried = 0
    for key, (name, city, state) in unique_props.items():
        if key in cache:
            continue

        result = lookup_place(name, city, state, api_key)
        cache[key] = result
        queried += 1

        status_label = result["api_status"]
        addr_preview = result["formatted_address"][:60] if result["formatted_address"] else "(no result)"
        print(f"  [{queried}/{to_query_count}] {name}, {city}, {state} → {addr_preview}  [{status_label}]")

        # Save cache every 50 calls so progress survives interruption
        if queried % 50 == 0:
            save_cache(cache)

        time.sleep(REQUEST_DELAY)

    # Final cache save
    if queried > 0:
        save_cache(cache)
        print(f"\nCache saved to {CACHE_FILE.name}")

    # Write enriched output CSV
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in all_rows:
            key    = cache_key(row["community_name"], row["city"], row["state"])
            result = cache.get(key, _empty_result("NOT_IN_CACHE"))
            writer.writerow({
                "ticker":            row.get("ticker", ""),
                "filing_year":       row.get("filing_year", ""),
                "accession_number":  row.get("accession_number", ""),
                "community_name":    row["community_name"],
                "city":              row["city"],
                "state":             row["state"],
                "formatted_address": result["formatted_address"],
                "lat":               result["lat"],
                "lng":               result["lng"],
                "place_id":          result["place_id"],
                "api_status":        result["api_status"],
            })

    ok_count   = sum(1 for r in all_rows
                     if cache.get(cache_key(r["community_name"], r["city"], r["state"]), {}).get("api_status") == "OK")
    fail_count = len(all_rows) - ok_count
    print(f"\nWrote {len(all_rows):,} rows → {OUTPUT_FILE.name}")
    print(f"  Resolved:    {ok_count:,}")
    print(f"  No result:   {fail_count:,}")


if __name__ == "__main__":
    main()

"""
Extract AVB apartment community names and locations from SEC 10-K Schedule III
(Real Estate and Accumulated Depreciation) across all available filings.

Outputs: data/processed/avb_communities.csv
Columns: filing_year, accession_number, community_name, city, state
"""

import csv
import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

FILINGS_DIR = Path(__file__).parent.parent / "data" / "raw" / "sec-edgar-filings" / "AVB" / "10-K"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "avb_communities.csv"

# Matches "City, ST" or "City, DC" etc. — two-letter state code at end
CITY_STATE_RE = re.compile(r'^.+,\s+[A-Z]{2}\.?$')

# Section headers and non-property entries that appear in the table
SKIP_NAMES = {
    "SAME STORE", "OTHER STABILIZED", "REDEVELOPMENT", "DEVELOPMENT",
    "UNCONSOLIDATED", "Total", "TOTAL", "Community", "Community Name",
}

# Exact non-community names that slip through the header check
SKIP_EXACT = {
    "For-sale condominium inventory",
    "The Park Loggia Commercial",
    "The Park Loggia Retail",
}


def get_cell_text(td) -> str:
    """Return stripped visible text from a <td>, collapsing whitespace."""
    text = td.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    # Strip footnote markers like (1), (2) etc. at end
    text = re.sub(r'\s*\(\d+\)\s*$', '', text).strip()
    return text


def is_section_header(name: str) -> bool:
    """Return True if the name looks like a category header, not a community."""
    upper = name.upper()
    for skip in SKIP_NAMES:
        if upper.startswith(skip):
            return True
    # All-uppercase strings longer than 4 chars are usually headers
    if name == name.upper() and len(name) > 4 and not re.search(r'\d', name):
        return True
    return False


def extract_communities(filepath: Path) -> list[dict]:
    """
    Parse one full-submission.txt and return a list of
    {community_name, city, state} dicts from Schedule III.
    """
    text = filepath.read_text(errors='replace')

    # Locate the Schedule III section
    marker = 'REAL ESTATE AND ACCUMULATED DEPRECIATION'
    start = text.find(marker)
    if start == -1:
        print(f"  WARNING: Schedule III not found in {filepath.parent.name}")
        return []

    # Take a generous slice — Schedule III in these filings is 1–4 MB
    schedule_html = text[start:start + 5_000_000]

    # Parse just this section for speed
    soup = BeautifulSoup(schedule_html, 'html.parser')

    communities = []
    seen = set()

    for tr in soup.find_all('tr'):
        cells = tr.find_all('td')
        # We need at least 2 non-empty cells
        texts = [get_cell_text(td) for td in cells]
        texts = [t for t in texts if t]

        if len(texts) < 2:
            continue

        name = texts[0]
        location = texts[1]

        # Skip blank, header, or known non-property rows
        if not name or not location:
            continue
        if is_section_header(name):
            continue
        if name in SKIP_EXACT:
            continue

        # Location cell must match "City, ST"
        if not CITY_STATE_RE.match(location):
            continue

        # Parse city and state
        location_clean = location.rstrip('.')
        parts = location_clean.rsplit(',', 1)
        if len(parts) != 2:
            continue
        city = parts[0].strip()
        state = parts[1].strip()

        # Deduplicate within a single filing
        key = (name, city, state)
        if key in seen:
            continue
        seen.add(key)

        communities.append({
            'community_name': name,
            'city': city,
            'state': state,
        })

    return communities


def infer_filing_year(accession: str, filepath: Path) -> str:
    """
    Derive the fiscal year from the submission text (period of report header).
    Falls back to accession number date prefix.
    """
    try:
        header = filepath.read_text(errors='replace')[:2000]
        m = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', header)
        if m:
            return m.group(1)[:4]
    except Exception:
        pass
    # Accession format: XXXXXXXXXX-YY-NNNNNN — YY is last two digits of year
    parts = accession.split('-')
    if len(parts) == 3:
        yy = parts[1]
        # Files are annual; period year is typically filing year minus 1
        year = int("20" + yy) - 1
        return str(year)
    return "unknown"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    filing_dirs = sorted(FILINGS_DIR.iterdir())
    for filing_dir in filing_dirs:
        submission = filing_dir / "full-submission.txt"
        if not submission.exists():
            continue

        accession = filing_dir.name
        year = infer_filing_year(accession, submission)
        print(f"Processing {accession} (FY {year}) ...")

        communities = extract_communities(submission)
        print(f"  Found {len(communities)} communities")

        for c in communities:
            all_rows.append({
                'filing_year': year,
                'accession_number': accession,
                'community_name': c['community_name'],
                'city': c['city'],
                'state': c['state'],
            })

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filing_year', 'accession_number', 'community_name', 'city', 'state'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

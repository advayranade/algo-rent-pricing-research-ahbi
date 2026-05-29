"""
Extract apartment/property community names and locations from SEC EDGAR 10-K
filings for any REIT ticker.

Strategy: parse the entire main 10-K HTML document (the <TYPE>10-K section
inside the SGML wrapper) for table rows that match (property_name, City, ST).
This covers both REITs that include city/state in Schedule III (e.g. AVB) and
those that only include it in the Item 2 property tables (e.g. CPT).

Usage:
    python extract_reit_communities.py [TICKER]

    TICKER may also be supplied interactively when omitted.

Output: data/processed/{ticker}_communities.csv
Columns: ticker, filing_year, accession_number, community_name, city, state

Each ticker gets its own CSV file. Running again for the same ticker
overwrites it with freshly extracted data.
"""

import csv
import re
import sys
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).parent.parent
FILINGS_BASE = BASE_DIR / "data" / "raw" / "sec-edgar-filings"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# "City, ST" or "City, ST." — two-letter state/territory code at the end
CITY_STATE_RE = re.compile(r'^[^,]{2,},\s+[A-Z]{2}\.?$')

# Generic section-header/column-label prefixes to skip (case-insensitive match)
HEADER_PREFIXES = (
    "SAME STORE", "OTHER STABILIZED", "REDEVELOPMENT", "DEVELOPMENT",
    "UNCONSOLIDATED", "TOTAL", "SUBTOTAL", "CONSOLIDATED",
    "WHOLLY OWNED", "JOINT VENTURE", "ENCUMBRANCES",
    "COMMUNITY", "PROPERTY", "DESCRIPTION", "LOCATION", "NAME",
    "INITIAL COST", "TOTAL COST", "ACCUMULATED", "BUILDING",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cell_text(td) -> str:
    """Return stripped visible text from a <td>, collapsing whitespace."""
    text = td.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\s*]*\(\d+\)\s*$', '', text).strip()  # trailing footnote (1)
    text = re.sub(r'\s*\*+\s*$', '', text).strip()         # trailing asterisk
    return text


def is_header_or_junk(name: str) -> bool:
    """Return True when a cell looks like a table header or non-property junk."""
    if not name or len(name) < 4:
        return True

    # Pure numbers, dollar signs, or percentages
    if re.match(r'^[\$\d,.()\-% ]+$', name):
        return True

    upper = name.upper()

    for prefix in HEADER_PREFIXES:
        if upper.startswith(prefix):
            return True

    # All-uppercase with no digits → very likely a section header
    if name == upper and not re.search(r'[\d]', name):
        return True

    # Footnote lines like "1. Note text" or "(1) Note text"
    if re.match(r'^[\d\(][\d\)\.]\s', name):
        return True

    # Non-residential property types that appear in some REITs' Schedule III
    # (MAA, Post Properties, etc. include office/retail/land in the same table)
    NON_RESIDENTIAL = (
        " OFFICE", " RETAIL", " COMMERCIAL", " LAND", " PARCEL",
        "FOR-SALE CONDOMINIUM", "OTHER REAL ESTATE",
    )
    for suffix in NON_RESIDENTIAL:
        if upper.endswith(suffix) or upper.startswith(suffix.strip()):
            return True

    return False


def try_split_combined_cell(text: str) -> Optional[tuple]:
    """
    Some REITs (e.g. CPT) concatenate name and city/state in a single cell:
      'Camden McGowen Station Houston, TX'
      'Camden Shady Grove Rockville, MD'

    Find the last '<word(s)>, ST' suffix; everything before is the name.
    Handles multi-word cities (Los Angeles, St. Petersburg).
    Returns (name, 'City, ST') or None.
    """
    if CITY_STATE_RE.match(text):
        return None  # pure city/state cell — handled by the caller

    m = re.search(r'\s([A-Za-z][a-zA-Z\s.]+),\s+([A-Z]{2})\.?$', text)
    if not m:
        return None

    city = m.group(1).strip()
    state = m.group(2).strip()
    name = text[: m.start()].strip()

    if not name or len(name) < 4:
        return None

    return name, f"{city}, {state}"


def find_unit_count(texts: list, after_index: int) -> str:
    """
    Return the first integer in texts[after_index+1:] that is plausibly an
    apartment unit count.  Returns '' if nothing matches.

    Hard rules to avoid false positives:
      - Range 50–2,499  (eliminates trivially small or large values)
      - Exclude 1,900–2,030  (year-built values that EQR and others place
        before the unit count column)
      - Exclude multiples of 1,000  (rounded dollar figures in thousands)

    Schedule III rows look like: Name | City, ST | 198 | $2,124 | ...
    EQR rows look like:          Name | City, ST | 2003 | 420 | $...
    """
    for t in texts[after_index + 1:]:
        clean = t.replace(',', '').strip()
        if re.match(r'^\d+$', clean):
            n = int(clean)
            if 50 <= n <= 2499 and not (1900 <= n <= 2030) and n % 1000 != 0:
                return str(n)
    return ""


def find_name_location_units(texts: list) -> Optional[tuple]:
    """
    Extract (property_name, 'City, ST', unit_count) from a row's cell texts.

    Layout A — separate cells (most REITs, including AVB Schedule III):
        texts[0] = name  |  texts[i] = 'City, ST'  |  texts[i+1] = # homes

    Layout B — combined cell (e.g. CPT development/pipeline tables):
        texts[0] = 'Community Name City, ST'
    """
    if not texts:
        return None

    # Layout A: scan for standalone City, ST then look right for unit count
    if len(texts) >= 2:
        name = texts[0]
        for i, t in enumerate(texts[1:], start=1):
            if CITY_STATE_RE.match(t):
                units = find_unit_count(texts, i)
                return name, t, units

    # Layout B: fused name+location cell (unit count is texts[1] if numeric)
    result = try_split_combined_cell(texts[0])
    if result:
        name, location = result
        units = find_unit_count(texts, 0)
        return name, location, units

    return None


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------

def extract_10k_html(text: str) -> str:
    """
    Isolate the main 10-K HTML body from the SGML full-submission wrapper.

    The wrapper embeds multiple documents (10-K, exhibits, XBRL, etc.).
    We want only the first <TYPE>10-K block so we don't accidentally parse
    XBRL label files or exhibit HTML that contains misleading table rows.
    """
    # Locate the primary 10-K document entry
    m = re.search(r'<TYPE>10-K[\r\n]', text)
    if not m:
        return text  # no SGML wrapper — treat entire file as the document

    # Find the <TEXT> tag that starts the actual HTML payload
    text_tag_pos = text.find('<TEXT>', m.start())
    if text_tag_pos == -1:
        return text[m.start():]
    content_start = text_tag_pos + 6  # skip past "<TEXT>"

    # End at the closing </DOCUMENT> tag for this section
    doc_end = text.find('</DOCUMENT>', content_start)
    if doc_end == -1:
        doc_end = content_start + 20_000_000  # generous fallback

    return text[content_start:doc_end]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_communities(filepath: Path) -> list[dict]:
    """
    Parse one full-submission.txt and return a list of
    {community_name, city, state, unit_count} dicts found in the 10-K body.
    """
    text = filepath.read_text(errors='replace')
    doc_html = extract_10k_html(text)

    soup = BeautifulSoup(doc_html, 'html.parser')

    communities = []
    seen: set = set()

    for tr in soup.find_all('tr'):
        cells = tr.find_all('td')
        texts = [get_cell_text(td) for td in cells]
        texts = [t for t in texts if t]

        result = find_name_location_units(texts)
        if result is None:
            continue

        name, location, unit_count = result

        if is_header_or_junk(name):
            continue

        location_clean = location.rstrip('.')
        parts = location_clean.rsplit(',', 1)
        if len(parts) != 2:
            continue
        city = parts[0].strip()
        state = parts[1].strip()

        key = (name, city, state)
        if key in seen:
            continue
        seen.add(key)

        communities.append({'community_name': name, 'city': city, 'state': state,
                            'unit_count': unit_count})

    return communities


def infer_filing_year(accession: str, filepath: Path) -> str:
    """Derive fiscal year from the SGML header; falls back to accession date."""
    try:
        header = filepath.read_text(errors='replace')[:2000]
        m = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', header)
        if m:
            return m.group(1)[:4]
    except Exception:
        pass
    parts = accession.split('-')
    if len(parts) == 3:
        return str(int("20" + parts[1]) - 1)
    return "unknown"


# ---------------------------------------------------------------------------
# Input on CLI
# ---------------------------------------------------------------------------

def prompt_ticker() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].strip().upper()

    print("\nSEC EDGAR 10-K Community Extractor")
    print("-----------------------------------")
    raw = input("Enter REIT ticker symbol (e.g. AVB, EQR, UDR, CPT): ").strip().upper()
    if not raw:
        sys.exit("No ticker provided. Exiting.")
    return raw


def main():
    ticker = prompt_ticker()

    filings_dir = FILINGS_BASE / ticker / "10-K"
    if not filings_dir.exists():
        sys.exit(
            f"No filings found for {ticker}.\n"
            f"Expected directory: {filings_dir}\n"
            "Download filings first (e.g. with edgar_pull.py)."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{ticker.lower()}_communities.csv"

    if output_path.exists():
        print(f"\nExisting data found — overwriting {output_path.name} with fresh extract.")
    else:
        print(f"\nNew REIT detected — creating {output_path.name}.")

    all_rows = []

    for filing_dir in sorted(filings_dir.iterdir()):
        submission = filing_dir / "full-submission.txt"
        if not submission.exists():
            continue

        accession = filing_dir.name
        year = infer_filing_year(accession, submission)
        print(f"  Processing {accession} (FY {year}) ...")

        communities = extract_communities(submission)
        print(f"    Found {len(communities)} properties")

        if not communities:
            print(f"    WARNING: no (name, City ST) rows found — check filing format")

        for c in communities:
            all_rows.append({
                'ticker': ticker,
                'filing_year': year,
                'accession_number': accession,
                'community_name': c['community_name'],
                'city': c['city'],
                'state': c['state'],
                'unit_count': c.get('unit_count', ''),
            })

    if not all_rows:
        print("\nNo data extracted across any filing.")
        print("Possible reasons:")
        print("  - The 10-K main property table uses metro-area headers instead of")
        print("    per-row city/state (common in CPT, EQR, UDR — full portfolio")
        print("    coverage requires a different parsing strategy for those REITs).")
        print("  - Location data is only in an exhibit, not the main 10-K HTML body.")
        return

    fieldnames = ['ticker', 'filing_year', 'accession_number', 'community_name', 'city', 'state', 'unit_count']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows → {output_path}")


if __name__ == '__main__':
    main()

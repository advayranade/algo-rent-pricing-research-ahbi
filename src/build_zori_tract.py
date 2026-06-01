"""
Build tract-level ZORI rent growth from Zillow ZIP-level monthly data.
Uses Census 2020 ZCTA-to-tract relationship file for area-weighted crosswalk.
Output: data/processed/zori_by_tract.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# --- Step 1: Load and parse ZORI ---
print("Loading ZORI...")
zori = pd.read_csv(RAW_DIR / "zori_zip_month.csv", dtype={"RegionName": str})

date_cols = [c for c in zori.columns if c[:2] in ("20", "19") and "-" in c]
pre_cols  = [c for c in date_cols if c.startswith("2019-")]
post_cols = [c for c in date_cols if c.startswith("2023-")]

print(f"Pre-period months (2019): {len(pre_cols)}")
print(f"Post-period months (2023): {len(post_cols)}")

zori["zori_2019"] = zori[pre_cols].mean(axis=1)
zori["zori_2023"] = zori[post_cols].mean(axis=1)
zori["rent_growth"] = (zori["zori_2023"] - zori["zori_2019"]) / zori["zori_2019"]

zori["zip"] = zori["RegionName"].str.zfill(5)

zori_clean = zori[["zip", "zori_2019", "zori_2023", "rent_growth"]].dropna()
print(f"ZIPs with complete 2019+2023 ZORI: {len(zori_clean):,}")

# --- Step 2: Load Census ZCTA-to-tract crosswalk ---
print("\nLoading ZCTA-to-tract crosswalk...")
xwalk = pd.read_csv(
    RAW_DIR / "zcta_tract_rel_2020.txt",
    sep="|",
    dtype=str,
    encoding="utf-8-sig",
    usecols=["GEOID_ZCTA5_20", "GEOID_TRACT_20", "AREALAND_PART"],
)

xwalk = xwalk[xwalk["GEOID_ZCTA5_20"].notna() & (xwalk["GEOID_ZCTA5_20"] != "")]
xwalk["zip"]   = xwalk["GEOID_ZCTA5_20"].str.zfill(5)
xwalk["tract"] = xwalk["GEOID_TRACT_20"].str.zfill(11)
xwalk["area"]  = pd.to_numeric(xwalk["AREALAND_PART"], errors="coerce").fillna(0)

xwalk = xwalk[xwalk["area"] > 0][["zip", "tract", "area"]]

# Compute res_ratio: fraction of ZCTA's land area in each tract
zip_total_area = xwalk.groupby("zip")["area"].sum().rename("zip_total_area")
xwalk = xwalk.merge(zip_total_area, on="zip")
xwalk["res_ratio"] = xwalk["area"] / xwalk["zip_total_area"]

print(f"Crosswalk rows: {len(xwalk):,}")
print(f"Unique ZIPs in crosswalk: {xwalk['zip'].nunique():,}")

# --- Step 3: Merge and aggregate to tract ---
print("\nMerging ZORI into crosswalk...")
merged = xwalk.merge(zori_clean, on="zip", how="inner")
print(f"Matched ZIP-tract rows: {len(merged):,}")
print(f"Unique tracts with ZORI coverage: {merged['tract'].nunique():,}")

def weighted_mean(group, value_col, weight_col):
    w = group[weight_col]
    v = group[value_col]
    return (v * w).sum() / w.sum()

tract_zori = merged.groupby("tract").apply(
    lambda g: pd.Series({
        "zori_2019":   weighted_mean(g, "zori_2019",   "res_ratio"),
        "zori_2023":   weighted_mean(g, "zori_2023",   "res_ratio"),
        "rent_growth": weighted_mean(g, "rent_growth",  "res_ratio"),
        "n_zips":      len(g),
    })
).reset_index()

print(f"\nTract-level ZORI summary:")
print(tract_zori["rent_growth"].describe())
print(f"Tracts with ZORI: {len(tract_zori):,}")

# --- Step 4: Save ---
out_path = PROCESSED_DIR / "zori_by_tract.csv"
tract_zori.to_csv(out_path, index=False)
print(f"\nSaved → {out_path.name}")

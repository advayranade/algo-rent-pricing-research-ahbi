# ZORI Rent Growth Outcome — Implementation Plan
# Instructions for Claude Code (zillow-zori branch)

## Context

The homeownership gap regressions returned marginal results (Hispanic gap p=0.065, Black gap p=0.845) concentrated in majority-white tracts with a severely underpowered H2 (N=54 minority tracts). The root problem throughout this project has been the ACS 5-year smoothing — it cannot detect the acute 2021-2022 rent surge.

This branch replaces the ACS-based outcome with **Zillow ZORI (Zillow Observed Rent Index)** — actual observed rent transaction data published monthly at the ZIP code level. ZORI captures the rent surge as it happened, has no smoothing lag problem, and is the standard data source in the empirical rent pricing literature.

The `regression_master.csv` already has everything needed except the ZORI rent growth column. The task is to download ZORI, crosswalk ZIP→census tract, compute rent growth, add it to the regression dataset, and run H1 and H2.

**Outcome variable:** `rent_growth` = (ZORI_2023 − ZORI_2019) / ZORI_2019, expressed as a decimal (e.g., 0.20 = 20% rent growth)

**Pre-period ZORI:** average of Jan–Dec 2019 monthly values per ZIP  
**Post-period ZORI:** average of Jan–Dec 2023 monthly values per ZIP

---

## Step 1: Download ZORI Data

```bash
curl -L -o data/raw/zori_zip_month.csv \
  "https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_sa_month.csv"
```

This is the smoothed, seasonally adjusted ZORI for all rental home types at the ZIP code level. The file has one row per ZIP and monthly columns formatted as `YYYY-MM-DD`. It covers 2014 onward. Verify the download succeeded and the file is not empty before proceeding.

If the URL above returns a 404, check https://www.zillow.com/research/data/ for the current ZORI download link — Zillow occasionally updates filenames. Look for "ZORI (Smoothed, Seasonally Adjusted): All Homes" at the ZIP level.

---

## Step 2: Download HUD ZIP-to-Tract Crosswalk

The HUD USPS crosswalk maps ZIP codes to census tracts with residential address allocation weights.

```bash
# Q4 2019 crosswalk (matches our pre-period)
curl -L -o data/raw/hud_zip_tract_q42019.xlsx \
  "https://www.huduser.gov/portal/datasets/usps/ZIP_TRACT_122019.xlsx"
```

If that URL fails, navigate to https://www.huduser.gov/portal/datasets/usps_crosswalk.html and download the ZIP→TRACT file for Q4 2019 (labeled "12/2019" on the site). Save it to `data/raw/hud_zip_tract_q42019.xlsx`.

The relevant columns are:
- `ZIP` — 5-digit ZIP code (as string, zero-padded)
- `TRACT` — 11-digit census tract GEOID (as string, zero-padded)
- `RES_RATIO` — fraction of the ZIP's residential addresses falling in this tract (floats summing to 1.0 per ZIP)

---

## Step 3: Write `src/build_zori_tract.py`

This script does three things: computes pre/post ZORI per ZIP, crosswalks to census tracts, and outputs a tract-level rent growth file.

### 3a. Load and parse ZORI

```python
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

zori = pd.read_csv(RAW_DIR / "zori_zip_month.csv", dtype={"RegionName": str})

# Identify monthly columns
date_cols = [c for c in zori.columns if c[:2] in ("20", "19") and "-" in c]

# Pre-period: average of 2019 monthly columns
pre_cols  = [c for c in date_cols if c.startswith("2019-")]
# Post-period: average of 2023 monthly columns
post_cols = [c for c in date_cols if c.startswith("2023-")]

print(f"Pre-period months (2019): {len(pre_cols)}")
print(f"Post-period months (2023): {len(post_cols)}")

zori["zori_2019"] = zori[pre_cols].mean(axis=1)
zori["zori_2023"] = zori[post_cols].mean(axis=1)
zori["rent_growth"] = (zori["zori_2023"] - zori["zori_2019"]) / zori["zori_2019"]

# Zero-pad ZIP to 5 digits
zori["zip"] = zori["RegionName"].str.zfill(5)

# Drop ZIPs missing either period
zori_clean = zori[["zip", "zori_2019", "zori_2023", "rent_growth"]].dropna()
print(f"ZIPs with complete 2019+2023 ZORI: {len(zori_clean):,}")
```

### 3b. Load HUD crosswalk

```python
xwalk = pd.read_excel(RAW_DIR / "hud_zip_tract_q42019.xlsx",
                      dtype={"ZIP": str, "TRACT": str})

# Standardize column names (HUD sometimes uses lowercase)
xwalk.columns = xwalk.columns.str.upper()
xwalk["zip"]   = xwalk["ZIP"].str.zfill(5)
xwalk["tract"] = xwalk["TRACT"].str.zfill(11)
xwalk["res_ratio"] = pd.to_numeric(xwalk["RES_RATIO"], errors="coerce").fillna(0)

# Keep only rows with residential allocation
xwalk = xwalk[xwalk["res_ratio"] > 0][["zip", "tract", "res_ratio"]]
print(f"Crosswalk rows: {len(xwalk):,}")
```

### 3c. Merge ZORI into crosswalk and aggregate to tract

```python
# Join ZORI onto crosswalk by ZIP
merged = xwalk.merge(zori_clean, on="zip", how="inner")
print(f"Matched ZIP-tract rows: {len(merged):,}")
print(f"Unique tracts with ZORI coverage: {merged['tract'].nunique():,}")

# For each tract: weighted average of rent_growth, weighted by res_ratio
# Weight = how much of each contributing ZIP falls in this tract
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
```

### 3d. Save output

```python
out_path = PROCESSED_DIR / "zori_by_tract.csv"
tract_zori.to_csv(out_path, index=False)
print(f"Saved → {out_path.name}")
```

---

## Step 4: Add ZORI to `regression_master.csv`

Either modify `build_regression_dataset.py` to join `zori_by_tract.csv` during construction, or write a quick patch script:

```python
# src/add_zori_to_regression.py
import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

master = pd.read_csv(PROCESSED / "regression_master.csv",
                     dtype={"census_tract_geoid": str})
zori   = pd.read_csv(PROCESSED / "zori_by_tract.csv",
                     dtype={"tract": str})

zori["tract"] = zori["tract"].str.zfill(11)
master["census_tract_geoid"] = master["census_tract_geoid"].str.zfill(11)

out = master.merge(zori[["tract", "zori_2019", "zori_2023", "rent_growth", "n_zips"]],
                   left_on="census_tract_geoid", right_on="tract", how="left")

n_matched = out["rent_growth"].notna().sum()
print(f"Tracts with ZORI matched: {n_matched}/{len(out)} ({n_matched/len(out)*100:.1f}%)")
print(f"rent_growth mean: {out['rent_growth'].mean():.4f}")
print(f"rent_growth range: {out['rent_growth'].min():.4f} – {out['rent_growth'].max():.4f}")

out.to_csv(PROCESSED / "regression_master.csv", index=False)
print("regression_master.csv updated with ZORI columns.")
```

**Check the match rate.** Expect ~80-90% of tracts to match — some outer suburban tracts may not have ZORI coverage. If match rate is below 70%, investigate which metros are missing coverage and flag it.

---

## Step 5: Write `src/run_h1_zori_regression.py`

### Specification

```
rent_growth ~ log_clc + ahbi_pre
            + median_hh_income + renter_pct + pct_black + pct_hispanic
            + total_housing_units + C(metro)
```

`ahbi_pre` (2019 AHBI score) is included as a control — it captures baseline housing stress conditions and directly addresses the question of whether CLC predicts rent growth *beyond* existing market tightness and burden. This is the key control that separates CLC's effect from the pre-existing conditions that attracted corporate landlords.

### Output format

Run three models as in the previous H1 scripts:
- **Model 1 (primary):** log_clc, HC1 robust SEs
- **Model 1b (robustness):** raw clc_winsorized, HC1 robust SEs  
- **Model 2:** log_clc, metro-clustered SEs

Print a plain-English headline for each:
```
rent_growth coefficient: +X.XXXX
Interpretation: A doubling of REIT concentration is associated with
X.X percentage point higher rent growth (2019→2023).
```

Save full output to `data/processed/h1_zori_results.txt`.

---

## Step 6: Write `src/run_h2_zori_regression.py`

Same structure as `run_h2_regression.py` but with `rent_growth` as the outcome.

Primary interaction model:
```
rent_growth ~ log_clc * minority_tract + ahbi_pre + controls + C(metro)
```

Run:
- Primary interaction (minority_tract > 50%)
- Split-sample (minority vs. white tracts)
- Race-specific breakdown (majority_black, majority_hispanic)
- R1: Threshold sensitivity (40/50/60%)
- R2: Continuous minority share

Skip R4 (metro-specific CLC interaction) unless R1-R3 show a consistent signal — it was too noisy in the gap analysis.

Save to `data/processed/h2_zori_results.txt`.

---

## Step 7: Run Everything in Order

```bash
# 1. Build ZORI tract file
python src/build_zori_tract.py

# 2. Add ZORI to regression master
python src/add_zori_to_regression.py

# 3. Verify the dataset looks right
python -c "
import pandas as pd
df = pd.read_csv('data/processed/regression_master.csv')
print(df[['census_tract_geoid','log_clc','rent_growth','ahbi_pre']].describe())
print('ZORI nulls:', df['rent_growth'].isna().sum())
"

# 4. Run H1
python src/run_h1_zori_regression.py

# 5. Run H2
python src/run_h2_zori_regression.py
```

---

## What to Check in Results

**If H1 is positive and significant (p < 0.05):**
- This is the main result. CLC predicts rent growth using actual transaction data.
- Check the coefficient magnitude: 0.05 = 5 percentage point higher rent growth per doubling of CLC. That's economically meaningful.
- The AHBI null results become a methodological finding (ACS can't detect what ZORI can).

**If H1 is marginal (0.05 < p < 0.10):**
- Report as directionally consistent. Focus the paper on the methodological contribution (AHBI construction + CLC pipeline) with honest discussion of power constraints.

**If H1 is still null:**
- Check whether `ahbi_pre` is absorbing the CLC variance (collinearity). Try removing it and see if CLC becomes significant.
- Check rent_growth distribution for outliers — COVID-era ZIPs with extreme rent swings can inflate SEs.
- At this point the honest conclusion is that CLC does not predict differential rent growth across tracts within metro areas, even with transaction data.

**ZORI coverage check:**
- If fewer than 70% of tracts match, break down missing coverage by metro — if it's concentrated in Atlanta or Charlotte (Sun Belt suburban), those are lower-CLC tracts and the bias from dropping them is likely conservative (attenuates the CLC coefficient).

---

## Files Created by This Plan

| File | Purpose |
|---|---|
| `data/raw/zori_zip_month.csv` | Raw ZORI download from Zillow |
| `data/raw/hud_zip_tract_q42019.xlsx` | HUD ZIP→tract crosswalk |
| `data/processed/zori_by_tract.csv` | Tract-level ZORI 2019, 2023, rent_growth |
| `src/build_zori_tract.py` | Crosswalk and aggregation script |
| `src/add_zori_to_regression.py` | Patches regression_master.csv with ZORI |
| `src/run_h1_zori_regression.py` | H1 regression: rent_growth ~ log_clc |
| `src/run_h2_zori_regression.py` | H2 regression: interaction with minority_tract |
| `data/processed/h1_zori_results.txt` | H1 output |
| `data/processed/h2_zori_results.txt` | H2 output |

Do not modify any existing scripts. All ZORI work lives in new files so the gap analysis on `main` is preserved.

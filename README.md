# Algorithmic Rent Pricing & the Affordable Housing Burden Index (AHBI)

**Paper:** [Social Science Research Network Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6941879)

## Overview

This is our research codebase for the AHBI project. It examines whether institutional REIT concentration in rental markets predicts rent growth — and whether that effect falls disproportionately on communities of color. It contains scripts to:

* construct an original **Affordable Housing Burden Index (AHBI)** at the census tract level from SEC EDGAR filings, ACS data, and Zillow ZORI,
* build a **Corporate Landlord Concentration (CLC)** measure for 8 publicly traded apartment REITs across 10 U.S. metros,
* test three core hypotheses using OLS regression, XGBoost with SHAP, and spatial econometric models (Moran's I, SAR, SEM), and
* produce an interactive per-tract prediction tool with SHAP waterfall explanations.

## Research Hypotheses

**H1** — Do census tracts with higher REIT ownership share (2019) experience faster rent growth (2019→2023), as measured by ZORI?

**H2** — Does the CLC effect on rent growth differ significantly between majority-minority and majority-white tracts?

**H3** — Does an XGBoost model rank CLC in the top predictors of rent growth by SHAP importance, confirming H1 without imposing linearity?

## Repository Structure

* `src/`
  * `edgar_pull.py`: Download 10-K annual filings for 8 apartment REITs from SEC EDGAR.
  * `extract_reit_communities.py`: Parse property-level community tables from 10-K text.
  * `geocode_communities.py`: Convert community addresses to lat/lon and census tract GEOIDs via Google Maps API.
  * `get_acs_data.py`: Pull tract-level demographics, income, renter share, and housing units from the Census API.
  * `get_census_tracts.py`: Fetch census tract geometries.
  * `find_metro_areas.py`: Assign tracts to metropolitan areas.
  * `compute_c1_clc.py`: Corporate Landlord Concentration — REIT units / total renter-occupied units (C1).
  * `compute_c2_prb.py`: Rent Burden Rate — share of renters paying ≥30% of income on rent (C2).
  * `compute_c3_tightness.py`: Market Tightness — inverted vacancy rate (C3).
  * `compute_ahbi.py`: Combine normalized C1–C3 components into a single AHBI score.
  * `build_regression_dataset.py`: Assemble the master regression dataset (`regression_master.csv`).
  * `build_zori_tract.py`: Crosswalk Zillow ZORI from ZIP codes to census tracts using Census ZCTA→tract area weights.
  * `add_zori_to_regression.py`: Merge ZORI rent growth into the regression master.
  * `run_h1_zori_regression.py`: H1 OLS regression — rent growth ~ log_clc + controls, metro-clustered SEs.
  * `run_h2_zori_regression.py`: H2 interaction models — CLC × minority_tract + robustness checks.
  * `run_h3_xgboost.py`: H3 XGBoost with cross-validated tuning + SHAP feature importance and beeswarm plots.
  * `run_spatial_regression.py`: Moran's I diagnostic on OLS residuals + SAR and SEM spatial models.
  * `predict_tract.py`: Interactive CLI tool — input a tract GEOID, get predicted rent growth and a SHAP waterfall chart.

* `data/raw/`: SEC EDGAR filings, ZORI CSV, Census TIGER shapefiles, crosswalk files *(gitignored — see Data section below)*.
* `data/processed/`: Intermediate and final datasets including `regression_master.csv` *(gitignored)*.
* `output/`: SHAP plots, Moran scatter, and per-tract waterfall charts.

## Requirements

* Python 3.9+
* A Census API key is required for ACS data pulls.
* Optional: Google Maps API key for geocoding (needed only to re-run the geocoding step; pre-geocoded results are in `communities_with_tracts.csv`).

```bash
pip install pandas numpy statsmodels xgboost shap matplotlib geopandas \
            libpysal esda spreg scikit-learn openpyxl
```

Set your keys in a `.env` file:

```
CENSUS_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here   # optional
```

## Quickstart

Run the pipeline in order:

```bash
# Build AHBI components
python src/compute_c1_clc.py
python src/compute_c2_prb.py
python src/compute_c3_tightness.py
python src/compute_ahbi.py

# Assemble regression dataset and add ZORI
python src/build_regression_dataset.py
python src/build_zori_tract.py
python src/add_zori_to_regression.py

# Run hypotheses
python src/run_h1_zori_regression.py
python src/run_h2_zori_regression.py
python src/run_h3_xgboost.py
python src/run_spatial_regression.py
```

To predict rent growth for a specific tract:

```bash
python src/predict_tract.py 13121001102          # by GEOID
python src/predict_tract.py --list atlanta       # list available tracts in a metro
python src/predict_tract.py                      # interactive mode
```

## Data

All raw data is excluded from version control due to file size (SEC filings: 1.8 GB, TIGER shapefiles: 232 MB). Raw files are downloaded automatically by the pipeline scripts or can be obtained from:

* **SEC EDGAR** — `edgar_pull.py` handles this automatically.
* **Zillow ZORI** — downloaded by `build_zori_tract.py` from Zillow's public research data page.
* **Census TIGER 2020** — downloaded and cached by `run_spatial_regression.py` on first run.
* **Census ZCTA→Tract crosswalk** — downloaded by `build_zori_tract.py` from `www2.census.gov`.
* **ACS 5-Year Estimates** — fetched by `get_acs_data.py` via the Census API (key required).

The final analysis-ready dataset (`regression_master.csv`, ~212 KB) will be made available on OSF upon paper publication.

## REITs Tracked

AIV · AVB · CPT · EQR · ESS · IRT · MAA · UDR

**Geographic scope:** 583 census tracts across 10 metros — San Francisco, Los Angeles, San Diego, San Jose, Seattle, New York, Washington DC, Atlanta, Dallas-Fort Worth, Charlotte.

## Other Notes

* **Why ZORI instead of ACS rent estimates?** ACS 5-year estimates smooth over 60 months and cannot detect the acute 2021–2022 rent surge. ZORI uses observed transaction data published monthly.
* **Why no metro fixed effects in H1?** Metro FEs absorb all between-metro variation — precisely where the CLC signal lives. Metro-clustered SEs correct for within-metro error correlation without eliminating the between-metro comparison.
* **Spatial autocorrelation:** OLS residuals show Moran's I = 0.79, largely driven by the ZIP→tract ZORI crosswalk (adjacent tracts share the same underlying ZIP rent value). SAR and SEM models are included as robustness checks; results are in `data/processed/spatial_regression_results.txt`.

## License

*To be added upon publication.*

## Developer

**Advay Ranade**

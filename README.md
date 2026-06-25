# Algorithmic Rent Pricing & the Affordable Housing Burden Index (AHBI)

A research codebase examining whether institutional REIT concentration in rental markets predicts rent growth, and whether that effect falls disproportionately on communities of color. The project constructs an original **Affordable Housing Burden Index (AHBI)** at the census tract level, builds a **Corporate Landlord Concentration (CLC)** measure from SEC EDGAR filings, and tests three core hypotheses using OLS regression, XGBoost, and spatial econometric models.

---

## Research Questions

**H1 — Does corporate landlord concentration predict rent growth?**
Do census tracts with higher REIT ownership share (2019) experience faster rent growth (2019→2023), as measured by the Zillow Observed Rent Index (ZORI)?

**H2 — Is the effect concentrated in minority communities?**
Does the CLC effect on rent growth differ significantly between majority-minority and majority-white tracts, suggesting a racial channel in algorithmic pricing spillovers?

**H3 — What does a non-parametric model say?**
Does an XGBoost model trained on 2019 tract features rank CLC in the top predictors of rent growth by SHAP importance, confirming H1 without imposing linearity?

---

## Data Sources

| Source | What it provides |
|---|---|
| **SEC EDGAR** (10-K filings) | Property-level community listings for 8 publicly traded apartment REITs |
| **Google Maps Geocoding API** | Converts community addresses to lat/lon and census tract GEOIDs |
| **ACS 5-Year Estimates** | Tract-level demographics, income, renter share, housing units |
| **Zillow ZORI** | Monthly observed rent index by ZIP code (2015–2024) |
| **Census TIGER 2020** | Tract shapefiles for spatial weights construction |
| **Census ZCTA→Tract crosswalk** | Area-weighted ZIP-to-tract interpolation |
| **HUD USPS crosswalk** | Residential address allocation weights (ZIP→tract) |

**REITs tracked:** AIV · AVB · CPT · EQR · ESS · IRT · MAA · UDR

**Geographic scope:** 583 census tracts across 10 metros — San Francisco, Los Angeles, San Diego, San Jose, Seattle, New York, Washington DC, Atlanta, Dallas-Fort Worth, Charlotte.

---

## AHBI Construction

The **Affordable Housing Burden Index** is a composite of three normalized components, each measured at the census tract level for 2018, 2019, 2022, and 2023:

| Component | Variable | Direction |
|---|---|---|
| **C1 — Corporate Landlord Concentration (CLC)** | REIT units / total renter-occupied units | Higher = more burdened |
| **C2 — Rent Burden Rate (PRB)** | Share of renters paying ≥30% of income on rent | Higher = more burdened |
| **C3 — Market Tightness** | Vacancy rate (inverted) | Lower vacancy = more burdened |

Each component is normalized 0–1 across the sample, then averaged into a single AHBI score. Changes in AHBI between pre- and post-periods capture worsening or improving housing stress.

---

## Repository Structure

```
src/
├── edgar_pull.py                   # Download 10-K filings from SEC EDGAR
├── extract_reit_communities.py     # Parse community tables from 10-K text
├── geocode_communities.py          # Address → lat/lon → census tract GEOID
├── get_acs_data.py                 # Pull ACS variables via Census API
├── get_census_tracts.py            # Fetch tract geometries
├── find_metro_areas.py             # Assign tracts to metro areas
├── compute_c1_clc.py               # Corporate Landlord Concentration (C1)
├── compute_c2_prb.py               # Rent Burden Rate (C2)
├── compute_c3_tightness.py         # Market Tightness (C3)
├── compute_ahbi.py                 # Combine C1–C3 into AHBI index
├── compute_clc_ahbip1.py           # CLC normalization pass
├── build_regression_dataset.py     # Assemble regression_master.csv
├── build_zori_tract.py             # Crosswalk ZORI from ZIP → tract
├── add_zori_to_regression.py       # Merge ZORI into regression master
├── run_h1_zori_regression.py       # H1: OLS rent growth ~ log_clc
├── run_h2_zori_regression.py       # H2: Interaction with minority_tract
├── run_h3_xgboost.py               # H3: XGBoost + SHAP feature importance
├── run_spatial_regression.py       # Moran's I + SAR/SEM spatial models
└── predict_tract.py                # Interactive per-tract prediction tool

data/
├── raw/                            # SEC filings, ZORI CSV, TIGER shapefiles
└── processed/                      # Intermediate and final datasets

output/                             # SHAP plots, Moran scatter, tract waterfall charts
```

---

## Key Output Files

| File | Contents |
|---|---|
| `data/processed/regression_master.csv` | One row per tract — CLC, AHBI, demographics, ZORI rent growth |
| `data/processed/h1_zori_results.txt` | H1 OLS regression output (3 models) |
| `data/processed/h2_zori_results.txt` | H2 interaction models + robustness checks |
| `data/processed/h3_xgboost_results.txt` | XGBoost metrics + SHAP feature importance table |
| `data/processed/spatial_regression_results.txt` | Moran's I + SAR/SEM comparison table |
| `output/h3_shap_bar.png` | SHAP mean absolute value bar chart |
| `output/h3_shap_beeswarm.png` | SHAP beeswarm across all tracts |
| `output/morans_i_scatter.png` | Moran scatter: OLS residuals vs. spatial lag |

---

## Running the Pipeline

### Prerequisites

```bash
pip install pandas numpy statsmodels xgboost shap matplotlib geopandas \
            libpysal esda spreg scikit-learn openpyxl
```

A Census API key is required for ACS data — set it in `.env` as `CENSUS_API_KEY`.

### Full pipeline (in order)

```bash
# 1. Pull SEC filings and extract REIT communities
python src/edgar_pull.py
python src/extract_reit_communities.py

# 2. Geocode and assign to census tracts
python src/geocode_communities.py
python src/get_census_tracts.py

# 3. Fetch ACS data and compute AHBI components
python src/get_acs_data.py
python src/compute_c1_clc.py
python src/compute_c2_prb.py
python src/compute_c3_tightness.py
python src/compute_ahbi.py

# 4. Build regression dataset and add ZORI
python src/build_regression_dataset.py
python src/build_zori_tract.py
python src/add_zori_to_regression.py

# 5. Run hypotheses
python src/run_h1_zori_regression.py
python src/run_h2_zori_regression.py
python src/run_h3_xgboost.py
python src/run_spatial_regression.py
```

### Predict rent growth for a specific tract

```bash
# By GEOID
python src/predict_tract.py 13121001102

# List tracts in a metro
python src/predict_tract.py --list atlanta

# Interactive mode
python src/predict_tract.py
```

---

## Methodological Notes

**Why ZORI instead of ACS rent estimates?**
ACS 5-year estimates smooth over 60 months and cannot detect the acute 2021–2022 rent surge. ZORI uses observed transaction data published monthly, making it the appropriate instrument for measuring rent growth over a defined pre/post window.

**Why metro-clustered SEs without metro fixed effects (H1)?**
Metro fixed effects absorb all between-metro variation — precisely where the CLC signal is strongest (high-CLC metros like Dallas and Charlotte saw ~30% rent growth; low-CLC metros like San Francisco saw ~5%). Clustering at the metro level corrects for within-metro error correlation without eliminating the between-metro comparison.

**Spatial autocorrelation finding:**
OLS residuals exhibit a Moran's I of 0.79, primarily driven by the ZIP→tract ZORI crosswalk (adjacent tracts sharing the same underlying ZIP rent value). SAR and SEM models absorb this structure but produce a sign reversal in the CLC coefficient, suggesting the OLS result reflects metro-level geographic clustering rather than within-neighborhood CLC effects. The ZIP-level regression (each observation has its own native ZORI value) is the cleanest resolution to this.

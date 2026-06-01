# Regression Pivot — Homeownership Gap Outcome
# Instructions for Claude Code

## What Changed and Why

The primary outcome variable has been changed from **ΔAHBI** to **racial homeownership gap change**. Three OLS regressions on ΔAHBI (2018→2023, 2018→2022, 2019→2022) all returned negative, mostly insignificant CLC coefficients (best result: p = 0.056, 2018→2022). The dominant predictor in every model was `ahbi_pre` (coef ≈ -0.44), indicating strong mean reversion mechanically correlated with CLC. Full results saved at `data/processed/h1_results.txt`.

The new primary outcomes are:
- `delta_gap_black` = homeownership_gap_black_2022 − homeownership_gap_black_2019
- `delta_gap_hispanic` = homeownership_gap_hispanic_2022 − homeownership_gap_hispanic_2019

Where `homeownership_gap_black` = white_nh homeownership rate − Black homeownership rate (already computed in ACS files as a positive number when white rate > Black rate). A positive Δgap means the racial gap **widened** — the hypothesis is that high-CLC tracts show positive Δgap.

The good news: **2019 ACS data already exists** (`communities_with_acs_2019.csv`) and the gap columns (`homeownership_gap_black`, `homeownership_gap_hispanic`) are already computed in all ACS vintages. This is mostly a script modification job.

---

## Step 1: Verify the Source Data

Before writing any code, confirm:

```python
import pandas as pd

df19 = pd.read_csv("data/processed/communities_with_acs_2019.csv")
df22 = pd.read_csv("data/processed/communities_with_acs_2022.csv")

# These columns must exist and have non-null values
print(df19[["census_tract_geoid","homeownership_gap_black","homeownership_gap_hispanic"]].describe())
print(df22[["census_tract_geoid","homeownership_gap_black","homeownership_gap_hispanic"]].describe())
```

Check null counts. If either gap column has >10% nulls, flag it — nulls occur when ACS suppresses data for small racial subgroups in a tract.

---

## Step 2: Modify `src/build_regression_dataset.py`

### 2a. Load the post-period ACS file for gap columns

The current script only loads `communities_with_acs_{pre_year}` for demographics. Add a second load for `communities_with_acs_{post_year}` to get the post-period gap values.

```python
# Add after loading acs_demo (pre-period):
acs_post: dict = {}
acs_post_path = PROCESSED_DIR / f"communities_with_acs_{post_year}.csv"
if not acs_post_path.exists():
    raise SystemExit(f"Missing: {acs_post_path.name}")
for r in csv.DictReader(open(acs_post_path, newline="", encoding="utf-8")):
    g = r.get("census_tract_geoid", "")
    if g and g not in acs_post:
        acs_post[g] = r
```

### 2b. Add gap columns to `output_fields()`

Replace the existing `output_fields()` function with this:

```python
def output_fields(pre_year: int, post_year: int) -> list:
    return [
        # Identifiers
        "metro", "census_tract_geoid", "county_fips", "state_fips",
        # Treatment
        f"clc_{pre_year}", "clc_winsorized", "log_clc",
        "reit_units_total", "num_reit_properties",
        # Primary outcomes — homeownership gap
        "gap_black_pre", "gap_black_post", "delta_gap_black",
        "gap_hispanic_pre", "gap_hispanic_post", "delta_gap_hispanic",
        # Secondary outcome — AHBI (retained for robustness/descriptives)
        f"ahbi_{pre_year}", f"ahbi_{post_year}", "delta_ahbi",
        # Demographic controls (pre-period ACS)
        "median_hh_income", "renter_pct", "total_housing_units",
        "pct_black", "pct_hispanic", "pct_white_nh",
        "homeownership_rate", "homeownership_rate_black",
        "homeownership_rate_hispanic", "homeownership_rate_white_nh",
        # Baseline gap (pre-period, used as control)
        "ahbi_pre",
    ]
```

### 2c. Add CLC winsorization

After loading `clc_pre`, compute winsorized CLC at the 99th percentile and log-transform it. Do this AFTER the inner join, so you're winsorizing within the analysis sample only.

```python
import numpy as np

# After building rows_out, compute winsorized CLC
clc_values = [float(r[f"clc_{pre_year}"]) for r in rows_out if r.get(f"clc_{pre_year}")]
p99 = np.percentile(clc_values, 99)
print(f"CLC 99th percentile: {p99:.4f}  (raw max: {max(clc_values):.4f})")

for r in rows_out:
    raw = to_float(r.get(f"clc_{pre_year}", ""), None)
    if raw is not None:
        winsorized = min(raw, p99)
        r["clc_winsorized"] = f"{winsorized:.6f}"
        r["log_clc"] = f"{np.log1p(winsorized):.6f}"
    else:
        r["clc_winsorized"] = ""
        r["log_clc"] = ""
```

### 2d. Compute gap outcomes inside the main loop

Inside the `for g in geoids:` loop, add:

```python
d_pre  = acs_demo.get(g, {})   # already exists
d_post = acs_post.get(g, {})   # new

gap_black_pre  = to_float(d_pre.get("homeownership_gap_black", ""),   None)
gap_black_post = to_float(d_post.get("homeownership_gap_black", ""),  None)
gap_hisp_pre   = to_float(d_pre.get("homeownership_gap_hispanic", ""),  None)
gap_hisp_post  = to_float(d_post.get("homeownership_gap_hispanic", ""), None)

delta_gap_black   = (gap_black_post  - gap_black_pre)  if (gap_black_post  is not None and gap_black_pre  is not None) else None
delta_gap_hispanic= (gap_hisp_post   - gap_hisp_pre)   if (gap_hisp_post   is not None and gap_hisp_pre   is not None) else None
```

Then add these to the row dict:
```python
"gap_black_pre":      f"{gap_black_pre:.6f}"   if gap_black_pre   is not None else "",
"gap_black_post":     f"{gap_black_post:.6f}"  if gap_black_post  is not None else "",
"delta_gap_black":    f"{delta_gap_black:.6f}" if delta_gap_black is not None else "",
"gap_hispanic_pre":   f"{gap_hisp_pre:.6f}"    if gap_hisp_pre    is not None else "",
"gap_hispanic_post":  f"{gap_hisp_post:.6f}"   if gap_hisp_post   is not None else "",
"delta_gap_hispanic": f"{delta_gap_hispanic:.6f}" if delta_gap_hispanic is not None else "",
"ahbi_pre":           f"{ahbi_pre_val:.6f}",
```

### 2e. Update the summary printout

Add to the printed summary:
```python
gap_b = [float(r["delta_gap_black"]) for r in rows_out if r["delta_gap_black"]]
gap_h = [float(r["delta_gap_hispanic"]) for r in rows_out if r["delta_gap_hispanic"]]
print(f"delta_gap_black mean:    {sum(gap_b)/len(gap_b):.4f}  (n={len(gap_b)})")
print(f"delta_gap_hispanic mean: {sum(gap_h)/len(gap_h):.4f}  (n={len(gap_h)})")
```

A positive mean means the gap widened on average — this is the expected direction.

### 2f. Run it

```bash
python src/build_regression_dataset.py --pre-year 2019 --post-year 2022
```

---

## Step 3: Rewrite `src/run_h1_regression.py`

Replace the regression script entirely. The new version runs **four models**:

**Model 1:** `delta_gap_black ~ log_clc + gap_black_pre + controls + C(metro)` — HC1 robust SEs
**Model 1b:** Same with raw winsorized CLC instead of log_clc — robustness check
**Model 2:** `delta_gap_hispanic ~ log_clc + gap_hispanic_pre + controls + C(metro)` — HC1 robust SEs
**Model 3:** Both gap outcomes in a joint model (SUR or just report separately)

Key implementation notes:

- Load `regression_master.csv`
- Use `log_clc` as the primary CLC variable (already computed in the dataset)
- Use `gap_black_pre` and `gap_hispanic_pre` as baseline controls (analogous to `ahbi_pre` in the old model — controls for mean reversion in the gap itself)
- Keep all other controls: `median_hh_income`, `renter_pct`, `pct_black`, `pct_hispanic`, `total_housing_units`, `C(metro)`
- Drop rows where the outcome is missing (tracts with suppressed ACS racial data)
- Report N separately for each model since missingness differs between Black and Hispanic gap outcomes

The headline result format should report:
```
MODEL A: delta_gap_black
  clc coefficient, 95% CI, p-value, N, R²
  Interpretation: positive coef = CLC predicts wider Black-white homeownership gap

MODEL B: delta_gap_hispanic  
  clc coefficient, 95% CI, p-value, N, R²
  Interpretation: positive coef = CLC predicts wider Hispanic-white homeownership gap
```

Save full output to `data/processed/h1_gap_results.txt`.

---

## Step 4: Update `src/run_h2_regression.py`

Update H2 to test the interaction on the gap outcomes instead of ΔAHBI:

**Model:** `delta_gap_black ~ log_clc * minority_tract + gap_black_pre + controls + C(metro)`

Where `minority_tract` = 1 if (pct_black + pct_hispanic) > 0.50.

Run for both `delta_gap_black` and `delta_gap_hispanic` as outcomes.

The four-cell summary table format from the previous H2 output was good — keep that format.

Save to `data/processed/h2_gap_results.txt`.

---

## Step 5: Run Everything and Check Output

```bash
# 1. Rebuild regression dataset
python src/build_regression_dataset.py --pre-year 2019 --post-year 2022

# 2. Run H1 on gap outcomes
python src/run_h1_regression.py

# 3. Run H2 on gap outcomes
python src/run_h2_regression.py
```

**What to check in the output:**
- `delta_gap_black` and `delta_gap_hispanic` means: should be positive on average (the gap widened nationally 2019→2022 due to COVID homebuying surge benefiting white households more)
- CLC coefficient direction: positive = gap widens more in high-CLC tracts (supports hypothesis)
- `gap_black_pre` control coefficient: expect negative (mean reversion) — fine, this is correct
- N for each model: expect some loss vs. H1 (racial subgroup suppression in small tracts)
- Any tract with CLC > 10 after winsorization: investigate — may indicate data issue

---

## What NOT to Change

- `get_acs_data.py` — no new ACS pulls needed, 2019 data exists
- `compute_c1_clc.py`, `compute_c2_prb.py`, `compute_c3_tightness.py` — component files untouched
- `compute_ahbi.py` — AHBI is retained for descriptive use, no changes needed
- Any 2018 data files — 2018 CLC/AHBI are no longer the primary analysis period

---

## Notes

- The `homeownership_gap_black` column in the ACS files is defined as `white_nh_rate − black_rate`. A positive value means white NH homeownership exceeds Black homeownership (the typical direction). A positive `delta_gap` means the disparity grew — this is the harmful outcome the hypothesis predicts.
- Expect more null rows for the Hispanic gap than the Black gap in some metros (smaller sample sizes in ACS racial subgroup tables for Hispanic tenure).
- The winsorized CLC cap (99th percentile) should be reported in the paper as a robustness decision. Also run a sensitivity check with winsorization at 95th percentile and confirm results are stable.

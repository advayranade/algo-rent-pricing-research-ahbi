"""
Spatial Regression: Moran's I Diagnostic + SAR/SEM Models

Addresses spatial autocorrelation in H1 ZORI OLS residuals by:
  1. Downloading Census TIGER 2020 tract polygons → computing centroids
  2. Building KNN (k=5) spatial weights matrix
  3. Running Moran's I on OLS residuals
  4. Using LM tests to select SAR vs SEM
  5. Running both spatial models and comparing to OLS

Output:
  data/processed/spatial_regression_results.txt
  output/morans_i_scatter.png
"""
from pathlib import Path
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import libpysal
import esda
import spreg

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TIGER_DIR     = BASE_DIR / "data" / "raw" / "tiger"
OUTPUT_DIR    = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
TIGER_DIR.mkdir(exist_ok=True)

FEATURES = [
    "log_clc", "ahbi_pre", "median_hh_income",
    "renter_pct", "pct_black", "pct_hispanic", "total_housing_units",
]
FEATURE_LABELS = [
    "log_clc", "ahbi_pre", "median_hh_income",
    "renter_pct", "pct_black", "pct_hispanic", "total_housing_units",
]
TARGET = "rent_growth"


# ── Step 1: Download and cache Census TIGER tract centroids ───────────────────

def get_tiger_centroids(state_fips_list):
    """Download TIGER 2020 tract shapefiles, return GeoDataFrame of centroids."""
    gdfs = []
    for fips in state_fips_list:
        cache_dir = TIGER_DIR / f"tl_2020_{fips}_tract"
        shp_path  = cache_dir / f"tl_2020_{fips}_tract.shp"

        if not shp_path.exists():
            url = (f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/"
                   f"tl_2020_{fips}_tract.zip")
            print(f"  Downloading TIGER state {fips}...", end=" ", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    zdata = resp.read()
                cache_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                    zf.extractall(cache_dir)
                print("done")
            except Exception as e:
                print(f"FAILED ({e}) — skipping state {fips}")
                continue
        else:
            print(f"  State {fips}: using cached shapefile")

        gdf = gpd.read_file(shp_path)
        gdfs.append(gdf)

    combined = pd.concat(gdfs, ignore_index=True)
    # Reproject to a projected CRS for accurate centroids, then back to WGS84
    combined_proj = combined.to_crs("EPSG:5070")  # Albers Equal Area (CONUS)
    combined["centroid"] = combined_proj.geometry.centroid
    combined_cent = gpd.GeoDataFrame(geometry=combined["centroid"], crs="EPSG:5070").to_crs("EPSG:4326")
    combined["lon"] = combined_cent.geometry.x
    combined["lat"] = combined_cent.geometry.y
    combined["GEOID"] = combined["GEOID"].str.zfill(11)
    return combined[["GEOID", "lon", "lat"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_coef(name, coef, se, pval):
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else "   "))
    return f"  {name:<30} {coef:>+9.4f}  SE={se:.4f}  p={pval:.3f}{sig}"


def moran_line(mi):
    sig = "✓ SIGNIFICANT" if mi.p_sim < 0.05 else "✗ NOT SIGNIFICANT"
    return (f"Moran's I = {mi.I:.4f}  E[I] = {mi.EI:.4f}  "
            f"z = {mi.z_norm:.3f}  p = {mi.p_norm:.4f}  {sig}")


def pseudo_r2(model, y):
    ss_res = np.sum((y - model.predy.flatten()) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def main():
    # ── Load regression data ──────────────────────────────────────────────────
    path = PROCESSED_DIR / "regression_master.csv"
    if not path.exists():
        raise SystemExit("regression_master.csv not found.")

    df = pd.read_csv(path, dtype={"census_tract_geoid": str})
    df["census_tract_geoid"] = df["census_tract_geoid"].str.zfill(11)

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_m = df.dropna(subset=FEATURES + [TARGET, "metro"]).copy().reset_index(drop=True)
    print(f"Analysis sample: {len(df_m):,} tracts across {df_m['metro'].nunique()} metros\n")

    state_fips = sorted(df_m["census_tract_geoid"].str[:2].unique().tolist())
    print(f"States to download: {state_fips}")

    # ── Step 1: Centroids ─────────────────────────────────────────────────────
    print("\n── Step 1: Getting tract centroids ──")
    centroids = get_tiger_centroids(state_fips)
    df_m = df_m.merge(centroids, left_on="census_tract_geoid", right_on="GEOID", how="left")
    missing = df_m["lon"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} tracts have no centroid — dropping them")
        df_m = df_m.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    print(f"  {len(df_m):,} tracts with centroids")

    # ── Step 2: Spatial weights ───────────────────────────────────────────────
    print("\n── Step 2: Building spatial weights ──")
    gdf = gpd.GeoDataFrame(
        df_m,
        geometry=gpd.points_from_xy(df_m["lon"], df_m["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")  # project to metres for distance-based KNN

    W_knn5 = libpysal.weights.KNN.from_dataframe(gdf, k=5)
    W_knn5.transform = "r"
    W_knn3 = libpysal.weights.KNN.from_dataframe(gdf, k=3)
    W_knn3.transform = "r"
    W_knn8 = libpysal.weights.KNN.from_dataframe(gdf, k=8)
    W_knn8.transform = "r"

    print(f"  KNN k=5: {W_knn5.n} units, mean neighbors = {W_knn5.mean_neighbors:.1f}, "
          f"islands = {W_knn5.islands}")

    # ── Step 3: Baseline OLS + Moran's I ─────────────────────────────────────
    print("\n── Step 3: Baseline OLS + spatial diagnostics ──")
    y = df_m[TARGET].values.reshape(-1, 1)

    # Scale features to prevent numerical overflow in LM test matrix ops.
    # median_hh_income in raw dollars (~$30k-200k) causes divide-by-zero in spreg.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_m[FEATURES].values)

    ols = spreg.OLS(
        y, X_scaled,
        w=W_knn5, spat_diag=True,
        name_y=TARGET, name_x=FEATURES,
    )

    residuals = ols.u.flatten()
    mi_knn5 = esda.moran.Moran(residuals, W_knn5, permutations=999)
    mi_knn3 = esda.moran.Moran(residuals, W_knn3, permutations=999)
    mi_knn8 = esda.moran.Moran(residuals, W_knn8, permutations=999)

    print(f"  OLS R² = {ols.r2:.4f}")
    print(f"  {moran_line(mi_knn5)}")

    # ── Step 4: LM test decision ──────────────────────────────────────────────
    print("\n── Step 4: Lagrange Multiplier tests ──")
    # spreg 1.9 stores LM tests as (statistic, p_value) tuples on the OLS object
    lm_lag_stat,  lm_lag_p  = ols.lm_lag
    lm_err_stat,  lm_err_p  = ols.lm_error
    rlm_lag_stat, rlm_lag_p = ols.rlm_lag
    rlm_err_stat, rlm_err_p = ols.rlm_error

    # Guard against NaN from numerical instability
    def safe_p(p): return float(p) if np.isfinite(p) else 1.0
    lm_lag_p, lm_err_p = safe_p(lm_lag_p), safe_p(lm_err_p)
    rlm_lag_p, rlm_err_p = safe_p(rlm_lag_p), safe_p(rlm_err_p)

    print(f"  LM-Lag   p = {lm_lag_p:.4f}  {'*' if lm_lag_p < 0.05 else ''}")
    print(f"  LM-Error p = {lm_err_p:.4f}  {'*' if lm_err_p < 0.05 else ''}")
    print(f"  RLM-Lag  p = {rlm_lag_p:.4f}  {'*' if rlm_lag_p < 0.05 else ''}")
    print(f"  RLM-Err  p = {rlm_err_p:.4f}  {'*' if rlm_err_p < 0.05 else ''}")

    run_sar = rlm_lag_p < 0.05
    run_sem = rlm_err_p < 0.05
    if run_sar and not run_sem:
        decision = "Robust LM-Lag significant → SAR (spatial lag model)"
    elif run_sem and not run_sar:
        decision = "Robust LM-Error significant → SEM (spatial error model)"
    elif run_sar and run_sem:
        decision = "Both robust LM tests significant → running both SAR and SEM"
        run_sar = run_sem = True
    else:
        decision = "Neither robust LM test significant → spatial autocorrelation may be mild"
        run_sar = run_sem = True  # run both anyway for completeness
    print(f"\n  Decision: {decision}")

    # ── Step 5: SAR model ─────────────────────────────────────────────────────
    print("\n── Step 5a: SAR (Spatial Lag) ──")
    sar = spreg.GM_Lag(
        y, X_scaled,
        w=W_knn5,
        name_y=TARGET, name_x=FEATURES,
    )
    sar_resid = sar.u.flatten()
    mi_sar = esda.moran.Moran(sar_resid, W_knn5, permutations=999)
    psr2_sar = pseudo_r2(sar, y.flatten())
    sar_rho = float(np.asarray(sar.rho).flat[0])
    print(f"  SAR ρ (spatial lag) = {sar_rho:.4f}")
    print(f"  Pseudo-R² = {psr2_sar:.4f}")
    print(f"  Moran's I on SAR residuals: {moran_line(mi_sar)}")

    # ── Step 5b: SEM model ────────────────────────────────────────────────────
    print("\n── Step 5b: SEM (Spatial Error) ──")
    sem = spreg.GM_Error(
        y, X_scaled,
        w=W_knn5,
        name_y=TARGET, name_x=FEATURES,
    )
    sem_resid = sem.u.flatten()
    mi_sem = esda.moran.Moran(sem_resid, W_knn5, permutations=999)
    psr2_sem = float(sem.pr2)
    sem_lam = float(np.asarray(sem.betas[-1]).flat[0])  # lambda is last beta
    print(f"  SEM λ (spatial error) = {sem_lam:.4f}")
    print(f"  Pseudo-R² = {psr2_sem:.4f}")
    print(f"  Moran's I on SEM residuals: {moran_line(mi_sem)}")

    # ── Step 6: Moran scatter plot ────────────────────────────────────────────
    print("\n── Step 6: Saving Moran's I scatter plot ──")
    W_knn5_raw = libpysal.weights.KNN.from_dataframe(gdf, k=5)  # non-standardized for plot
    W_knn5_raw.transform = "r"
    lag_resid = libpysal.weights.spatial_lag.lag_spatial(W_knn5, residuals)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(residuals, lag_resid, alpha=0.5, s=18, color="#457b9d", edgecolors="none")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    # Fit line
    m, b = np.polyfit(residuals, lag_resid, 1)
    xr = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(xr, m * xr + b, color="#e63946", linewidth=1.5,
            label=f"Moran's I = {mi_knn5.I:.3f}  p = {mi_knn5.p_norm:.4f}")
    ax.set_xlabel("OLS Residuals", fontsize=11)
    ax.set_ylabel("Spatial Lag of Residuals (W·e)", fontsize=11)
    ax.set_title("Moran's I Scatter: OLS Residuals vs. Spatial Lag\n"
                 "(rent_growth ~ log_clc + controls, KNN k=5)", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    scatter_path = OUTPUT_DIR / "morans_i_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {scatter_path.name}")

    # ── Build text output ─────────────────────────────────────────────────────
    # Extract log_clc index (position 0 in FEATURES list)
    clc_idx = FEATURES.index("log_clc")

    # betas[0] = intercept for all models; features start at index 1
    def get_coef(model, feat_idx, stat_attr="z_stat"):
        i = feat_idx + 1
        c  = float(model.betas[i])
        se = float(model.std_err[i])
        p  = float(getattr(model, stat_attr)[i][1])
        return c, se, p

    ols_c, ols_se, ols_p = get_coef(ols, clc_idx, stat_attr="t_stat")
    sar_c, sar_se, sar_p = get_coef(sar, clc_idx, stat_attr="z_stat")
    sem_c, sem_se, sem_p = get_coef(sem, clc_idx, stat_attr="z_stat")

    out = []
    out.append("=" * 70)
    out.append("SPATIAL REGRESSION RESULTS")
    out.append(f"Sample: {len(df_m):,} tracts | Spatial weights: KNN k=5, row-standardized")
    out.append("=" * 70)

    out.append("\n\n── MORAN'S I: OLS RESIDUALS ──")
    out.append(f"  {moran_line(mi_knn5)}")
    out.append(f"\n  Robustness across weight specifications:")
    out.append(f"  {'Weights':<20}  {'Moran I':>8}  {'z':>7}  {'p':>7}")
    out.append("  " + "─" * 48)
    for label, mi in [("KNN k=3", mi_knn3), ("KNN k=5 (primary)", mi_knn5), ("KNN k=8", mi_knn8)]:
        out.append(f"  {label:<20}  {mi.I:>8.4f}  {mi.z_norm:>7.3f}  {mi.p_norm:>7.4f}")

    out.append("\n\n── LM TESTS (model selection) ──")
    out.append(f"  LM-Lag   p = {lm_lag_p:.4f}  {'← significant' if lm_lag_p < 0.05 else ''}")
    out.append(f"  LM-Error p = {lm_err_p:.4f}  {'← significant' if lm_err_p < 0.05 else ''}")
    out.append(f"  RLM-Lag  p = {rlm_lag_p:.4f}  {'← significant' if rlm_lag_p < 0.05 else ''}")
    out.append(f"  RLM-Err  p = {rlm_err_p:.4f}  {'← significant' if rlm_err_p < 0.05 else ''}")
    out.append(f"\n  → {decision}")

    out.append("\n\n── COMPARISON TABLE: OLS vs SAR vs SEM ──")
    out.append(f"  {'':30}  {'OLS':>10}  {'SAR':>10}  {'SEM':>10}")
    out.append("  " + "─" * 65)
    out.append(f"  {'log_clc coefficient':30}  {ols_c:>+10.4f}  {sar_c:>+10.4f}  {sem_c:>+10.4f}")
    out.append(f"  {'log_clc p-value':30}  {ols_p:>10.3f}  {sar_p:>10.3f}  {sem_p:>10.3f}")
    out.append(f"  {'R² / Pseudo-R²':30}  {ols.r2:>10.4f}  {psr2_sar:>10.4f}  {psr2_sem:>10.4f}")
    out.append(f"  {'Moran I (residuals)':30}  {mi_knn5.I:>10.4f}  {mi_sar.I:>10.4f}  {mi_sem.I:>10.4f}")
    out.append(f"  {'Moran I p-value':30}  {mi_knn5.p_norm:>10.4f}  {mi_sar.p_norm:>10.4f}  {mi_sem.p_norm:>10.4f}")

    out.append(f"\n  SAR ρ (spatial lag coef):   {sar_rho:.4f}")
    out.append(f"  SEM λ (spatial error coef): {sem_lam:.4f}")

    out.append("\n\n── SAR FULL COEFFICIENT TABLE ──")
    out.append(f"  {'Intercept':<30} {float(sar.betas[0]):>+9.4f}  SE={float(sar.std_err[0]):.4f}  p={float(sar.z_stat[0][1]):.3f}")
    for i, feat in enumerate(FEATURES):
        c, se, p = float(sar.betas[i+1]), float(sar.std_err[i+1]), float(sar.z_stat[i+1][1])
        out.append(fmt_coef(feat, c, se, p))
    out.append(f"  {'W_rent_growth (ρ)':<30} {sar_rho:>+9.4f}")

    out.append("\n\n── SEM FULL COEFFICIENT TABLE ──")
    out.append(f"  {'Intercept':<30} {float(sem.betas[0]):>+9.4f}  SE={float(sem.std_err[0]):.4f}  p={float(sem.z_stat[0][1]):.3f}")
    for i, feat in enumerate(FEATURES):
        c  = float(sem.betas[i+1])
        se = float(sem.std_err[i+1]) if i+1 < len(sem.std_err) else float("nan")
        p  = float(sem.z_stat[i+1][1]) if i+1 < len(sem.z_stat) else float("nan")
        out.append(fmt_coef(feat, c, se, p))
    out.append(f"  {'λ (spatial error)':<30} {sem_lam:>+9.4f}")

    out.append("\n\n── INTERPRETATION ──")
    if mi_knn5.p_norm < 0.05:
        out.append(f"  OLS residuals show significant spatial autocorrelation (I={mi_knn5.I:.4f}, "
                   f"p={mi_knn5.p_norm:.4f}).")
        out.append(f"  SAR/SEM models reduce this: SAR residual I={mi_sar.I:.4f}, "
                   f"SEM residual I={mi_sem.I:.4f}.")
    else:
        out.append("  OLS residuals do not show significant spatial autocorrelation.")
        out.append("  Spatial models included for completeness.")

    clc_direction = "consistent" if np.sign(ols_c) == np.sign(sar_c) == np.sign(sem_c) else "inconsistent"
    out.append(f"\n  log_clc sign across models: {clc_direction}")
    out.append(f"  OLS: {ols_c:+.4f} (p={ols_p:.3f})  →  "
               f"SAR: {sar_c:+.4f} (p={sar_p:.3f})  →  "
               f"SEM: {sem_c:+.4f} (p={sem_p:.3f})")

    output = "\n".join(out)
    print("\n\n" + output)

    result_path = PROCESSED_DIR / "spatial_regression_results.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"\nSaved → {result_path.name}")


if __name__ == "__main__":
    main()

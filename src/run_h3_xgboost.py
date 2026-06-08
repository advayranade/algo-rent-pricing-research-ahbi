"""
H3: XGBoost Predictive Model for ZORI Rent Growth (2019→2023)

Goal: Train an XGBoost regressor on 2019 pre-period tract features to predict
rent_growth, then use SHAP values to show which features matter most.

Key questions:
  1. Does log_clc rank in the top 3 features by SHAP importance?
  2. Do SHAP values show CLC's impact is larger in minority tracts (H2 story,
     surfaced non-parametrically)?

Output:
  data/processed/h3_xgboost_results.txt  — model metrics + feature importance table
  output/h3_shap_bar.png                 — mean |SHAP| bar chart
  output/h3_shap_beeswarm.png            — SHAP beeswarm plot
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR    = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "log_clc",
    "ahbi_pre",
    "pct_black",
    "pct_hispanic",
    "pct_white_nh",
    "renter_pct",
    "median_hh_income",
    "total_housing_units",
]
TARGET = "rent_growth"
FEATURE_LABELS = {
    "log_clc":             "REIT Concentration (log CLC)",
    "ahbi_pre":            "Housing Burden Index (AHBI)",
    "pct_black":           "% Black",
    "pct_hispanic":        "% Hispanic",
    "pct_white_nh":        "% White Non-Hispanic",
    "renter_pct":          "Renter Share",
    "median_hh_income":    "Median HH Income",
    "total_housing_units": "Total Housing Units",
}

RANDOM_STATE = 42


def metro_stratify_key(metro_series):
    """Integer codes for metro-stratified split."""
    return metro_series.astype("category").cat.codes


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    path = PROCESSED_DIR / "regression_master.csv"
    if not path.exists():
        raise SystemExit("regression_master.csv not found.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} tracts")

    if TARGET not in df.columns:
        raise SystemExit("rent_growth not found. Run add_zori_to_regression.py first.")

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_m = df.dropna(subset=FEATURES + [TARGET, "metro"]).copy()
    print(f"Complete observations: {len(df_m):,}")
    print(f"Metros: {df_m['metro'].nunique()} — {df_m['metro'].unique().tolist()}")
    print(f"rent_growth: mean={df_m[TARGET].mean():.4f}  std={df_m[TARGET].std():.4f}  "
          f"range=[{df_m[TARGET].min():.4f}, {df_m[TARGET].max():.4f}]\n")

    X = df_m[FEATURES].values
    y = df_m[TARGET].values
    strat_key = metro_stratify_key(df_m["metro"])

    # ── 80/20 train/test split stratified by metro ────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_m.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=strat_key,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Cross-validated hyperparameter tuning ─────────────────────────────────
    # n_estimators is excluded from the grid — early stopping will select it
    # automatically in the final fit. Grid search focuses on regularization params.
    param_grid = {
        "max_depth":        [2, 3, 4],
        "learning_rate":    [0.01, 0.05, 0.10],
        "subsample":        [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda":       [1, 5, 10],
        "min_child_weight": [1, 5, 10],
    }
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,          # high ceiling; early stopping will trim this
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Running grid search (5-fold CV)...")
    gs = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    best_params = gs.best_params_
    print(f"Best params: {best_params}")
    print(f"Best CV RMSE: {-gs.best_score_:.4f}\n")

    # ── Fit final model with early stopping ───────────────────────────────────
    # Hold out 20% of train as a validation set for early stopping.
    # This lets XGBoost stop adding trees once validation RMSE plateaus.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=RANDOM_STATE
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"Early stopping: best iteration = {model.best_iteration}  "
          f"(of 500 max trees)")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    y_pred_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test   = r2_score(y_test, y_pred_test)
    print(f"Test RMSE: {rmse_test:.4f}")
    print(f"Test R²:   {r2_test:.4f}\n")

    # ── Evaluate on train set (for overfit check) ─────────────────────────────
    y_pred_train = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train   = r2_score(y_train, y_pred_train)

    # ── SHAP values on full sample ────────────────────────────────────────────
    print("Computing SHAP values on full sample...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":       FEATURES,
        "label":         [FEATURE_LABELS[f] for f in FEATURES],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_df["rank"] = shap_df.index + 1
    clc_rank = shap_df.loc[shap_df["feature"] == "log_clc", "rank"].values[0]
    print(f"\nSHAP feature importance ranking:")
    print(shap_df[["rank", "label", "mean_abs_shap"]].to_string(index=False))
    print(f"\nlog_clc rank: #{clc_rank}")

    # ── H2 non-parametric check: CLC SHAP in minority vs white tracts ─────────
    minority_mask = (df_m["pct_black"] + df_m["pct_hispanic"]) > 0.50
    clc_idx       = FEATURES.index("log_clc")
    clc_shap_min  = shap_values[minority_mask.values, clc_idx]
    clc_shap_wht  = shap_values[~minority_mask.values, clc_idx]
    print(f"\nCLC SHAP by tract type:")
    print(f"  Minority tracts (>50%):  mean SHAP = {clc_shap_min.mean():+.4f}  N={minority_mask.sum()}")
    print(f"  White tracts (≤50%):     mean SHAP = {clc_shap_wht.mean():+.4f}  N={(~minority_mask).sum()}")
    h2_nonparam = (
        "→ H2 NON-PARAMETRIC SUPPORT: CLC raises rent growth more in minority tracts."
        if clc_shap_min.mean() > clc_shap_wht.mean()
        else "→ H2 NOT SUPPORTED non-parametrically: CLC SHAP similar or lower in minority tracts."
    )
    print(h2_nonparam)

    # ── Plot 1: SHAP mean |SHAP| bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e63946" if f == "log_clc" else "#457b9d" for f in shap_df["feature"]]
    ax.barh(shap_df["label"][::-1], shap_df["mean_abs_shap"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance: XGBoost Rent Growth Model\n(SHAP mean absolute values)", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    # Annotate CLC bar
    clc_row = shap_df[shap_df["feature"] == "log_clc"].iloc[0]
    ax.annotate(
        f"#{int(clc_row['rank'])} REIT Concentration",
        xy=(clc_row["mean_abs_shap"], len(FEATURES) - int(clc_row["rank"])),
        xytext=(clc_row["mean_abs_shap"] + 0.001, len(FEATURES) - int(clc_row["rank"])),
        fontsize=9, color="#e63946",
    )
    plt.tight_layout()
    bar_path = OUTPUT_DIR / "h3_shap_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {bar_path.name}")

    # ── Plot 2: SHAP beeswarm ─────────────────────────────────────────────────
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X,
        feature_names=[FEATURE_LABELS[f] for f in FEATURES],
    )
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_explanation, show=False, max_display=len(FEATURES))
    plt.title("SHAP Beeswarm: XGBoost Rent Growth Model", fontsize=12, pad=12)
    plt.tight_layout()
    beeswarm_path = OUTPUT_DIR / "h3_shap_beeswarm.png"
    fig2.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {beeswarm_path.name}")

    # ── Build text output ─────────────────────────────────────────────────────
    out = []
    out.append("=" * 70)
    out.append("H3: XGBOOST PREDICTIVE MODEL — ZORI RENT GROWTH (2019→2023)")
    out.append("=" * 70)
    out.append(f"\nSample: {len(df_m):,} tracts  |  Train: {len(X_train)}  Test: {len(X_test)}")
    out.append(f"Features: {', '.join(FEATURES)}")
    out.append(f"Target: rent_growth = (ZORI_2023 − ZORI_2019) / ZORI_2019")

    out.append("\n\n── BEST HYPERPARAMETERS (5-fold CV grid search) ──")
    for k, v in best_params.items():
        out.append(f"  {k:<20} {v}")
    out.append(f"  CV RMSE (best):      {-gs.best_score_:.4f}")
    out.append(f"  n_estimators (early stopping): {model.best_iteration} of 500 max")

    out.append("\n\n── MODEL PERFORMANCE ──")
    out.append(f"  Train RMSE: {rmse_train:.4f}   Train R²: {r2_train:.4f}")
    out.append(f"  Test  RMSE: {rmse_test:.4f}   Test  R²: {r2_test:.4f}")
    overfit = (r2_train - r2_test) / max(r2_train, 1e-6) * 100
    out.append(f"  Overfit (train-test R² gap): {overfit:.1f}%")

    out.append("\n\n── SHAP FEATURE IMPORTANCE (full sample) ──")
    out.append(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    out.append("  " + "─" * 55)
    for _, row in shap_df.iterrows():
        marker = " ◄ KEY" if row["feature"] == "log_clc" else ""
        out.append(f"  {int(row['rank']):<6} {row['label']:<35} {row['mean_abs_shap']:>12.4f}{marker}")

    out.append(f"\n  log_clc rank: #{clc_rank} of {len(FEATURES)}")
    if clc_rank <= 3:
        out.append("  ✓ H3 SUPPORTED: REIT concentration ranks in top 3 by SHAP importance.")
    else:
        out.append(f"  ✗ H3 NOT IN TOP 3: REIT concentration ranks #{clc_rank}.")

    out.append("\n\n── H2 NON-PARAMETRIC CHECK: CLC SHAP BY TRACT TYPE ──")
    out.append(f"  Minority tracts (pct_black + pct_hispanic > 50%): N={minority_mask.sum()}")
    out.append(f"    Mean CLC SHAP: {clc_shap_min.mean():+.4f}  (std={clc_shap_min.std():.4f})")
    out.append(f"  White tracts (≤50%): N={(~minority_mask).sum()}")
    out.append(f"    Mean CLC SHAP: {clc_shap_wht.mean():+.4f}  (std={clc_shap_wht.std():.4f})")
    out.append(f"\n  {h2_nonparam}")

    out.append("\n\n── PLOTS ──")
    out.append(f"  output/h3_shap_bar.png       — mean |SHAP| bar chart")
    out.append(f"  output/h3_shap_beeswarm.png  — SHAP beeswarm plot")

    output = "\n".join(out)
    print("\n" + output)

    result_path = PROCESSED_DIR / "h3_xgboost_results.txt"
    result_path.write_text(output, encoding="utf-8")
    print(f"\nSaved → {result_path.name}")


if __name__ == "__main__":
    main()

"""
Predict rent growth for a specific census tract using the H3 XGBoost model.

Usage:
    python src/predict_tract.py                        # interactive prompt
    python src/predict_tract.py 6001402900             # pass GEOID directly
    python src/predict_tract.py --list atlanta         # list available tracts in a metro
"""

import sys
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR    = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "log_clc", "ahbi_pre", "pct_black", "pct_hispanic",
    "pct_white_nh", "renter_pct", "median_hh_income", "total_housing_units"
]

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

BEST_PARAMS = {
    "max_depth":        4,
    "learning_rate":    0.05,
    "n_estimators":     94,
    "subsample":        0.6,
    "colsample_bytree": 0.8,
    "reg_lambda":       1,
    "min_child_weight": 1,
    "random_state":     42,
}


def load_and_train():
    df = pd.read_csv(PROCESSED_DIR / "regression_master.csv",
                     dtype={"census_tract_geoid": str})

    for col in FEATURES + ["rent_growth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_clean = df.dropna(subset=FEATURES + ["rent_growth"]).copy()
    df_clean["census_tract_geoid"] = df_clean["census_tract_geoid"].str.zfill(11)

    X = df_clean[FEATURES].values
    y = df_clean["rent_growth"].values

    model = xgb.XGBRegressor(**BEST_PARAMS, verbosity=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    return model, explainer, df_clean, X


def list_tracts(metro_query, df):
    mask = df["metro"].str.lower().str.contains(metro_query.lower())
    subset = df[mask][["census_tract_geoid", "metro", "rent_growth"]].copy()
    subset["rent_growth_pct"] = (subset["rent_growth"] * 100).round(1)
    if subset.empty:
        print(f"\nNo tracts found matching '{metro_query}'.")
        print(f"Available metros: {sorted(df['metro'].unique())}")
    else:
        print(f"\nTracts in metros matching '{metro_query}':\n")
        print(f"  {'GEOID':<15}  {'Metro':<22}  {'Actual rent growth'}")
        print("  " + "─" * 55)
        for _, row in subset.iterrows():
            print(f"  {row['census_tract_geoid']:<15}  {row['metro']:<22}  {row['rent_growth_pct']:>+.1f}%")


def predict_tract(geoid, model, explainer, df, X):
    geoid = str(geoid).zfill(11)

    row = df[df["census_tract_geoid"] == geoid]
    if row.empty:
        # Try without zero-padding
        row = df[df["census_tract_geoid"].str.lstrip("0") == geoid.lstrip("0")]

    if row.empty:
        print(f"\n  ✗ Tract {geoid} not found in the dataset.")
        print(f"    Use --list <metro> to see available tracts.")
        return

    idx   = row.index[0]
    tract = row.iloc[0]

    # Get position in cleaned array for SHAP
    pos = df.index.get_loc(idx)
    x_tract = X[pos].reshape(1, -1)

    predicted = model.predict(x_tract)[0]
    actual    = tract["rent_growth"]
    error     = predicted - actual

    shap_values = explainer.shap_values(x_tract)[0]
    base_value  = explainer.expected_value

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  TRACT: {geoid}")
    print(f"  Metro: {tract.get('metro', 'N/A')}")
    print("═" * 60)

    print(f"\n  Predicted rent growth (2019→2023):  {predicted*100:>+.1f}%")
    print(f"  Actual rent growth (2019→2023):     {actual*100:>+.1f}%")
    print(f"  Prediction error:                   {error*100:>+.1f} ppt")

    print(f"\n  {'FEATURE':<35}  {'Value':>10}  {'SHAP (impact)':>14}")
    print("  " + "─" * 65)
    shap_pairs = sorted(zip(FEATURES, tract[FEATURES].values, shap_values),
                        key=lambda x: abs(x[2]), reverse=True)
    for feat, val, sv in shap_pairs:
        label    = FEATURE_LABELS.get(feat, feat)
        arrow    = "▲" if sv > 0 else "▼"
        sv_str   = f"{arrow} {abs(sv)*100:.2f} ppt"
        if feat in ("pct_black","pct_hispanic","pct_white_nh","renter_pct"):
            val_str = f"{val*100:.1f}%"
        elif feat == "median_hh_income":
            val_str = f"${val:,.0f}"
        elif feat == "total_housing_units":
            val_str = f"{val:,.0f}"
        else:
            val_str = f"{val:.3f}"
        print(f"  {label:<35}  {val_str:>10}  {sv_str:>14}")

    print(f"\n  Model baseline (avg rent growth):   {base_value*100:>+.1f}%")
    print(f"  Sum of SHAP adjustments:            {sum(shap_values)*100:>+.1f} ppt")
    print(f"  Final prediction:                   {predicted*100:>+.1f}%")

    # ── Waterfall plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    labels_ordered = [FEATURE_LABELS.get(f, f) for f, _, _ in shap_pairs]
    vals_ordered   = [sv for _, _, sv in shap_pairs]

    colors = ["#e05c5c" if v > 0 else "#5c8de0" for v in vals_ordered]
    bars   = ax.barh(labels_ordered[::-1], [v*100 for v in vals_ordered[::-1]],
                     color=colors[::-1], height=0.6)

    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SHAP impact on predicted rent growth (percentage points)")
    ax.set_title(f"Why the model predicted {predicted*100:+.1f}% rent growth\n"
                 f"Tract {geoid} | {tract.get('metro','')}")

    for bar, val in zip(bars, [v*100 for v in vals_ordered[::-1]]):
        ax.text(val + (0.2 if val >= 0 else -0.2), bar.get_y() + bar.get_height()/2,
                f"{val:+.2f} ppt", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"tract_{geoid}_shap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  SHAP waterfall saved → output/tract_{geoid}_shap.png")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geoid", nargs="?", help="Census tract GEOID")
    parser.add_argument("--list", metavar="METRO", help="List tracts in a metro")
    args = parser.parse_args()

    print("\nLoading data and training model...")
    model, explainer, df, X = load_and_train()
    print(f"Ready. {len(df):,} tracts loaded.\n")

    if args.list:
        list_tracts(args.list, df)
        return

    if args.geoid:
        predict_tract(args.geoid, model, explainer, df, X)
        return

    # Interactive mode
    print("Enter a census tract GEOID to get a prediction.")
    print("Type 'list <metro>' to see available tracts, or 'quit' to exit.\n")
    while True:
        try:
            user = input("  GEOID > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() == "quit":
            break
        if user.lower().startswith("list"):
            parts = user.split(maxsplit=1)
            list_tracts(parts[1] if len(parts) > 1 else "", df)
        else:
            predict_tract(user, model, explainer, df, X)
        print()


if __name__ == "__main__":
    main()

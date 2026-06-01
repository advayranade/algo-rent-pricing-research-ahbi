"""
Patch regression_master.csv with ZORI columns (zori_2019, zori_2023, rent_growth, n_zips).
Runs after build_zori_tract.py.
"""
import pandas as pd
from pathlib import Path

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

master = pd.read_csv(PROCESSED / "regression_master.csv", dtype={"census_tract_geoid": str})
zori   = pd.read_csv(PROCESSED / "zori_by_tract.csv",    dtype={"tract": str})

zori["tract"] = zori["tract"].str.zfill(11)
master["census_tract_geoid"] = master["census_tract_geoid"].str.zfill(11)

# Drop any pre-existing ZORI columns to avoid duplication on re-runs
for col in ["zori_2019", "zori_2023", "rent_growth", "n_zips"]:
    if col in master.columns:
        master = master.drop(columns=[col])

out = master.merge(
    zori[["tract", "zori_2019", "zori_2023", "rent_growth", "n_zips"]],
    left_on="census_tract_geoid",
    right_on="tract",
    how="left",
).drop(columns=["tract"])

n_matched = out["rent_growth"].notna().sum()
n_total   = len(out)
print(f"Tracts with ZORI matched: {n_matched}/{n_total} ({n_matched/n_total*100:.1f}%)")
print(f"rent_growth mean:  {out['rent_growth'].mean():.4f}")
print(f"rent_growth range: {out['rent_growth'].min():.4f} – {out['rent_growth'].max():.4f}")

if n_matched / n_total < 0.70:
    print("\nWARNING: Match rate below 70% — checking missing coverage by metro")
    missing = out[out["rent_growth"].isna()]["metro"].value_counts()
    print(missing)

out.to_csv(PROCESSED / "regression_master.csv", index=False)
print("\nregression_master.csv updated with ZORI columns.")

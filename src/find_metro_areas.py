import pandas as pd

df = pd.read_csv("~/code/algorithmic-rent-pricing-ahbi/data/processed/communities_with_tracts.csv")

county_counts = (
    df.groupby("county_name")
    .size()
    .reset_index(name="num_properties")
    .sort_values("num_properties", ascending=False)
)

print(county_counts.to_string())
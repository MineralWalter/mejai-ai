import pandas as pd

df = pd.read_parquet(
    "data/parquet/participants/europe_part_00164.parquet"
)

print(
    df[df["match_id"] == "EUW1_7921883844"].to_string()
)
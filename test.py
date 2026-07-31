import pandas as pd

df = pd.read_parquet(
    "data/parquet/events/part_00000.parquet"
)

items = (
    df[df["event_type"] == "ITEM_PURCHASED"]
    ["item_id"]
    .value_counts()
)

print(items.head(20))
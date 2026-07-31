import os
import pandas as pd

DATA_DIR = "data/parquet"

def ensure_data_dir():
    # Create parquet output folder if missing.
    os.makedirs(DATA_DIR,exist_ok=True)

def append_to_parquet(data: list[dict],filename: str):
    # Append rows into parquet file. Creates file if it does not exist.
    if not data:
        return
    ensure_data_dir()

    filepath = os.path.join(DATA_DIR,filename)
    new_df = pd.DataFrame(data)

    if os.path.exists(filepath):

        old_df = pd.read_parquet(filepath)
        df = pd.concat([old_df,new_df],ignore_index=True)
    else:
        df = new_df

    df.to_parquet(filepath,index=False)
    print(f"Saved {len(data)} rows -> {filename}")

def save_match_result(result: dict):
    """
    Save processed match output.
    Expected:

    {
        "match": dict,
        "participants": list,
        "snapshots": list,
        "events": list
    }
    """
    append_to_parquet([result["match"]],"matches.parquet")
    append_to_parquet(result["participants"],"participants.parquet")
    append_to_parquet(result["snapshots"],"snapshots.parquet")
    append_to_parquet(result["events"],"events.parquet")


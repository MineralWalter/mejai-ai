import os
import pandas as pd


DATA_DIR = "data/parquet"


def ensure_data_dir():
    """
    Create parquet output folders if missing.
    """

    folders = ["matches","participants","snapshots","events"]

    for folder in folders:

        os.makedirs(os.path.join(DATA_DIR, folder),exist_ok=True)


def write_parquet_batch(data: list[dict],folder: str,batch_id: int):
    """
    Write one parquet batch.

    Example:
        data/parquet/events/part_00001.parquet
    """

    if not data:
        return
    
    filepath = os.path.join(DATA_DIR,folder,f"part_{batch_id:05d}.parquet")
    if os.path.exists(filepath):
        raise FileExistsError(f"Already exists: {filepath}")
    
    df = pd.DataFrame(data)

    temp_path = filepath + ".tmp"
    df.to_parquet(temp_path,index=False)
    os.rename(temp_path,filepath)
    
    print(f"Saved {len(data)} rows -> {filepath}")

def save_match_batch(
    matches: list[dict],
    participants: list[dict],
    snapshots: list[dict],
    events: list[dict],
    batch_id: int
):
    """
    Save one processed batch of matches.
    Expected:
    matches:
        list of match dictionaries
    participants:
        list of participant dictionaries
    snapshots:
        list of timeline snapshots
    events:
        list of gameplay events
    """
    ensure_data_dir()

    write_parquet_batch(
        matches,
        "matches",
        batch_id
    )


    write_parquet_batch(
        participants,
        "participants",
        batch_id
    )


    write_parquet_batch(
        snapshots,
        "snapshots",
        batch_id
    )


    write_parquet_batch(
        events,
        "events",
        batch_id
    )
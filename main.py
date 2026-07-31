import math
import os

import pandas as pd

from src.process_match import process_match
from src.parquet_writer import save_match_batch

MATCHS_FILE = "checkpoints/matches_checkpoint.csv"

BATCH_SIZE = 400

PARQUET_MATCH_DIR = "data/parquet/matches"


def load_match_ids() -> list[str]:
    """
    Load all Riot match IDs.
    """
    df = pd.read_csv(MATCHS_FILE)
    return df["match_id"].tolist()


def batch_exists(batch_id: int) -> bool:
    """
    Returns True if this batch has already been written.
    """
    filepath = os.path.join(
        PARQUET_MATCH_DIR,
        f"part_{batch_id:05d}.parquet"
    )

    return os.path.exists(filepath)

TEST_LIMIT = 100

def main():

    match_ids = load_match_ids()
    if TEST_LIMIT:
        match_ids = match_ids[:TEST_LIMIT]

    total_batches = math.ceil(
        len(match_ids) / BATCH_SIZE
    )

    print(f"Loaded {len(match_ids)} matches.")
    print(f"{total_batches} batches of {BATCH_SIZE}.")

    for batch_id in range(total_batches):

        if batch_exists(batch_id):
            print(f"[Batch {batch_id}] Already processed.")
            continue

        start = batch_id * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(match_ids))

        print(
            f"\n========== Batch {batch_id} =========="
        )

        batch_matches = []
        batch_participants = []
        batch_snapshots = []
        batch_events = []

        for match_id in match_ids[start:end]:

            result = process_match(match_id)

            if result is None:
                print(f"Failed: {match_id}")
                continue

            batch_matches.append(
                result["match"]
            )

            batch_participants.extend(
                result["participants"]
            )

            batch_snapshots.extend(
                result["snapshots"]
            )

            batch_events.extend(
                result["events"]
            )

        save_match_batch(
            matches=batch_matches,
            participants=batch_participants,
            snapshots=batch_snapshots,
            events=batch_events,
            batch_id=batch_id
        )

        print(
            f"[Batch {batch_id}] "
            f"Matches={len(batch_matches)} | "
            f"Participants={len(batch_participants)} | "
            f"Snapshots={len(batch_snapshots)} | "
            f"Events={len(batch_events)}"
        )

    print("\nAll batches completed.")


if __name__ == "__main__":
    main()
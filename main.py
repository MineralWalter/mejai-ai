import math
import os

import pandas as pd
from threading import Thread, Lock

from src.process_match import process_match
from src.parquet_writer import save_match_batch


MATCHES_FILE = "checkpoints/matches_checkpoint.csv"

BATCH_SIZE = 100

PARQUET_MATCH_DIR = "data/parquet/matches"

LANES = {
    "sea": (
        "VN2_",
        "SG2_",
        "TW2_",
        "OC1_",
        "PH2_",
        "TH2_"
    ),

    "asia": (
        "KR_",
        "JP1_"
    ),

    "europe": (
        "EUW1_",
        "EUN1_",
        "TR1_",
        "RU_"
    ),

    "americas": (
        "NA1_",
        "BR1_",
        "LA1_",
        "LA2_"
    )
}

TEST_LIMIT = None
# TEST_LIMIT = 100

print_lock = Lock()

def log(message):
    with print_lock:
        print(message)

def load_match_ids():

    df = pd.read_csv(
        MATCHES_FILE
    )

    return df["match_id"].tolist()


def assign_lane(match_id):

    for lane, prefixes in LANES.items():

        if match_id.startswith(prefixes):
            return lane

    return None


def split_lanes(match_ids):

    lanes = {
        "sea": [],
        "asia": [],
        "europe": [],
        "americas": []
    }

    for match_id in match_ids:

        lane = assign_lane(match_id)

        if lane:
            lanes[lane].append(match_id)
        else:
            log(f"Unknown match routing: {match_id}")

    return lanes


PARQUET_DIR = "data/parquet"

def batch_exists(lane, batch_id):
    folders = [
        "matches",
        "participants",
        "snapshots",
        "events"
    ]

    for folder in folders:

        filepath = os.path.join(
            PARQUET_DIR,
            folder,
            f"{lane}_part_{batch_id:05d}.parquet"
        )

        if not os.path.exists(filepath):
            return False

    return True

def process_lane(lane, match_ids):

    total_batches = math.ceil(len(match_ids) / BATCH_SIZE)

    log(f"[{lane}] {len(match_ids)} matches | {total_batches} batches")


    for batch_id in range(total_batches):

        if batch_exists(lane, batch_id):
            log(f"[{lane}] Batch {batch_id} already exists")
            continue

        start = batch_id * BATCH_SIZE
        end = min(start + BATCH_SIZE,len(match_ids))

        current_batch = match_ids[start:end]

        log(f"[{lane}] Starting batch {batch_id}")


        batch_matches = []
        batch_participants = []
        batch_snapshots = []
        batch_events = []


        for i, match_id in enumerate(current_batch):
            if i % 10 == 0:
                log(f"[{lane}] Batch {batch_id}: {i}/{len(current_batch)}") # Just for me to see things 

            try:
                result = process_match(match_id)

            except Exception as e:
                log(f"[{lane}] Crash processing {match_id}: {e}")
                continue

            if result is None:
                log(f"[{lane}] Failed {match_id}")
                continue


            batch_matches.append(result["match"])
            batch_participants.extend(result["participants"])
            batch_snapshots.extend(result["snapshots"])
            batch_events.extend(result["events"])

        save_match_batch(
            matches=batch_matches,
            participants=batch_participants,
            snapshots=batch_snapshots,
            events=batch_events,
            batch_id=batch_id,
            lane=lane
        )

        log(
            f"[{lane}] Batch {batch_id} saved | "
            f"M={len(batch_matches)} "
            f"P={len(batch_participants)} "
            f"S={len(batch_snapshots)} "
            f"E={len(batch_events)}"
        )

    log(
        f"[{lane}] COMPLETE"
    )

def main():

    match_ids = load_match_ids()
    if TEST_LIMIT:
        match_ids = match_ids[:TEST_LIMIT]


    log(f"Loaded {len(match_ids)} matches")

    lanes = split_lanes(match_ids)
    threads = []
    for lane, ids in lanes.items():
        if not ids:
            continue

        thread = Thread(target=process_lane,args=(lane, ids))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    log("ALL LANES COMPLETE")


if __name__ == "__main__":
    main()
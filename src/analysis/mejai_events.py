import json
from pathlib import Path

import pandas as pd

from src.research.config import MEJAI_EVENT_CATALOGUE, MEJAI_ITEM_ID, PARQUET_DIR
from src.research.utils import get_valid_match_ids


LANES = ["sea","asia","europe","americas",]

ITEM_ID_COLUMNS = ["item_id","before_item_id","after_item_id",] 

def log(message=""):
    print(message)

def find_event_files():
    """
    Each result is:(lane, filepath)
    """
    files = []
    event_directory = PARQUET_DIR / "events"

    for lane in LANES:
        lane_files = sorted(event_directory.glob(f"{lane}_part_*.parquet"))
        files.extend((lane, filepath)for filepath in lane_files)
    return files



def load_valid_event_file(lane,filepath,columns=None,):
    """
    Read one event file and keep only matches marked eligible
    by the valid-match manifest.
    """

    valid_match_ids = get_valid_match_ids(lane)

    try:
        frame = pd.read_parquet(filepath,columns=columns,engine="pyarrow",)

    except Exception as error:
        log(f"[ERROR] Could not read "f"{filepath}: {error}")
        return pd.DataFrame()

    if frame.empty:
        return frame

    if "match_id" not in frame.columns:
        log(f"[ERROR] Event file has no match_id: "f"{filepath}")
        return pd.DataFrame()

    frame = frame.copy()
    frame["match_id"] = (frame["match_id"].astype(str))

    return frame[
        frame["match_id"].isin(valid_match_ids)].copy()


# ============================================================
# EVENT-TYPE INSPECTION
# ============================================================

def inspect_event_types(files):
    counts = {}

    for file_number, (lane,filepath,) in enumerate(files,start=1,):
        frame = load_valid_event_file(lane,filepath,columns=["match_id","event_type",],)

        if frame.empty:
            continue

        for event_type, count in (frame["event_type"].value_counts(dropna=False).items()):
            key = str(event_type)

            counts[key] = (counts.get(key, 0)+ int(count))

        if (file_number % 100 == 0 or file_number == len(files)):
            log(f"  Event types: "f"{file_number:,}/{len(files):,} files")

    return dict(sorted(counts.items(),key=lambda item: item[0],))


def normalize_item_columns(frame):
    """
    Ensure the three item-ID fields exist and are numeric.
    """

    frame = frame.copy()

    for column in ITEM_ID_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

        frame[column] = pd.to_numeric(frame[column],errors="coerce",)

    return frame


def find_mejai_candidates(frame):
    if frame.empty:
        return pd.DataFrame()

    frame = normalize_item_columns(frame)
    candidate_mask = pd.Series(False,index=frame.index,)

    for column in ITEM_ID_COLUMNS:
        candidate_mask = (candidate_mask | frame[column].eq(MEJAI_ITEM_ID))

    return frame[candidate_mask].copy()


def inspect_item_events(files):
    """
    Find Mejai-related item events while excluding invalid matches before catalogue construction.
    """

    mejai_event_frames = []

    for file_number, (lane,filepath,) in enumerate(files, start=1,):
        frame = load_valid_event_file(lane, filepath,)

        if frame.empty:
            continue

        if not any(column in frame.columns for column in ITEM_ID_COLUMNS):
            continue

        candidates = find_mejai_candidates(frame)

        if candidates.empty:
            continue

        candidates["storage_partition"] = lane
        candidates["source_file"] = str(filepath)

        mejai_event_frames.append(candidates)

        if (file_number % 100 == 0 or file_number == len(files)):
            current_count = sum(len(candidate_frame)for candidate_frame in mejai_event_frames)

            log(f"  Mejai scan: "f"{file_number:,}/{len(files):,} files | "f"{current_count:,} candidate events")

    if not mejai_event_frames:
        return pd.DataFrame()

    return pd.concat(mejai_event_frames,ignore_index=True,)

def serialize_value(value):
    if value is None:
        return None

    try:
        missing = pd.isna(value)

        if isinstance(missing,bool,) and missing:
            return None

    except (TypeError, ValueError):
        pass

    if hasattr(value,"tolist",):
        return value.tolist()

    if hasattr(value,"item",):
        return value.item()

    return value


def build_catalogue(frame):
    if frame.empty:
        return []

    catalogue = []

    for row in frame.itertuples(index=False,name=None,):
        record = {column: serialize_value(value) for column, value in zip(frame.columns,row,)}

        catalogue.append(record)

    return catalogue


def save_catalogue(catalogue):
    MEJAI_EVENT_CATALOGUE.parent.mkdir(parents=True,exist_ok=True,)

    temporary_path = Path(str(MEJAI_EVENT_CATALOGUE)+ ".tmp")

    with open(temporary_path,"w",encoding="utf-8",) as file:
        json.dump(catalogue,file,indent=2,ensure_ascii=False,)

    temporary_path.replace(MEJAI_EVENT_CATALOGUE)


# ============================================================
# REPORTING
# ============================================================

def print_candidate_examples(mejai_events,):
    display_columns = [
        column for column in [
            "match_id",
            "timestamp",
            "event_type",
            "participant_id",
            "item_id",
            "before_item_id",
            "after_item_id",
            "storage_partition",
            "source_file",
        ]
        if column in mejai_events.columns
    ]

    log("")
    log("MEJAI CANDIDATE EXAMPLES")

    log(mejai_events[display_columns].head(50).to_string(index=False))

    if len(mejai_events) > 50:
        log("")
        log(f"Showing 50 of "f"{len(mejai_events):,} events.")


def print_mejai_event_summary(mejai_events,):
    log("")
    log("MEJAI EVENT TYPES")

    event_type_counts = (mejai_events["event_type"].value_counts(dropna=False))

    for event_type, count in (event_type_counts.items()):
        log(f"{event_type}: "f"{count:,}")

    log("")
    log("Mejai matches by storage partition:")

    partition_counts = (mejai_events[["storage_partition","match_id",]].drop_duplicates()["storage_partition"].value_counts())

    log(partition_counts.to_string())

# MAIN

def main():
    log("MEJAI EVENT DISCOVERY")
    log("=" * 70)

    files = find_event_files()
    log(f"Event files found: "f"{len(files):,}")

    if not files:
        log("[ERROR] No event Parquet files found")
        return

    log("")
    log("VALID EVENT TYPES")

    event_type_counts = inspect_event_types(files)

    for event_type, count in (event_type_counts.items()):
        log(f"{event_type}: "f"{count:,}")

    log("")
    log("SEARCHING FOR MEJAI ")

    mejai_events = inspect_item_events(files)

    log(f"Mejai candidate events: "f"{len(mejai_events):,}")

    if mejai_events.empty:
        log("")
        log("[WARNING] No Mejai candidates found.")
        return

    mejai_events = (mejai_events.sort_values(["match_id","participant_id","timestamp",],kind="stable",)
                    .reset_index(drop=True))

    print_candidate_examples(mejai_events)
    print_mejai_event_summary(mejai_events)
    catalogue = build_catalogue(mejai_events)
    save_catalogue(catalogue)

    log("")
    log(f"Catalogue written to: "f"{MEJAI_EVENT_CATALOGUE}")
    log(f"Total candidate events: "f"{len(catalogue):,}")
    log("")

    log("[PASSED] MEJAI EVENT CATALOGUE REBUILT ""FROM ELIGIBLE MATCHES")

if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

PARQUET_DIR = Path("data/parquet")

LANES = [
    "sea",
    "asia",
    "europe",
    "americas",
]

SNAPSHOT_TABLE = "snapshots"
MATCH_TABLE = "matches"
PARTICIPANT_TABLE = "participants"
EVENT_TABLE = "events"

def log(message):
    print(message)


def determine_lane(match_id):
    """
    Determine the dataset region from the League match ID prefix.

    Returns:
        "sea"
        "asia"
        "europe"
        "americas"
        None
    """

    match_id = str(match_id).upper()

    if match_id.startswith(("VN", "SG", "TH", "PH", "ID", "MY")):
        return "sea"

    if match_id.startswith(("KR", "JP", "TW", "HK")):
        return "asia"

    if match_id.startswith(("EUW", "EUN", "TR", "RU")):
        return "europe"

    if match_id.startswith(("NA", "BR", "LA1", "LA2", "OC")):
        return "americas"

    return None

def find_files(table, lane):
    """
    Find all Parquet files belonging to a table and region.
    """

    directory = PARQUET_DIR / table

    if not directory.exists():
        return []

    return sorted(
        directory.glob(
            f"{lane}*part*.parquet"
        )
    )


# ============================================================
# PARQUET LOADING
# ============================================================

def load_table_for_lane(table, lane, columns=None, match_ids=None):
    files = find_files(table, lane)

    if not files:
        return pd.DataFrame()

    frames = []

    for filepath in files:
        try:
            filters = None

            if match_ids:
                filters = [("match_id", "in", list(match_ids))]

            if columns is not None:
                available_columns = pd.read_parquet(filepath,engine="pyarrow",).columns

                selected_columns = [column for column in columns if column in available_columns]

                if not selected_columns:
                    continue

                frame = pd.read_parquet(filepath,columns=selected_columns,filters=filters,engine="pyarrow",)

            else:
                frame = pd.read_parquet(filepath,filters=filters,engine="pyarrow",)

            frames.append(frame)

        except Exception as error:
            log(f"[ERROR] Could not read {filepath}: {error}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

# ============================================================
# MATCH FILTERING
# ============================================================

def filter_to_match_ids(df,match_ids,):
    #Keep only rows belonging to the requested matches.
    if df.empty:
        return df

    if not match_ids:
        return df.iloc[0:0].copy()

    df = df.copy()

    df["match_id"] = (df["match_id"].astype(str))

    return df[df["match_id"].isin(match_ids)].copy()

# ============================================================
# TABLE PREPARATION
# ============================================================

def prepare_matches(df):
    # Standardise match table types.

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (df["match_id"].astype(str))

    return df


def prepare_participants(df):
    # Standardise participant table types.

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (df["match_id"].astype(str))

    if "participant_id" in df.columns:

        df["participant_id"] = pd.to_numeric(df["participant_id"],errors="coerce",)
        df = df.dropna(subset=["participant_id"])
        df["participant_id"] = (df["participant_id"].astype(int))

    return df


def prepare_snapshots(df):
    """
    Standardise snapshot table types.

    Snapshot timestamps are milliseconds
    from game start.
    """

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (df["match_id"].astype(str))

    if "participant_id" in df.columns:

        df["participant_id"] = pd.to_numeric(df["participant_id"],errors="coerce",)

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_numeric(df["timestamp"],errors="coerce",)

    df = df.dropna(subset=["participant_id","timestamp",])

    df["participant_id"] = (df["participant_id"].astype(int))

    df["timestamp"] = (df["timestamp"].astype(int))

    return df

def prepare_events(df):
    """
    Standardise event table types
    """

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (df["match_id"].astype(str))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"],errors="coerce",)

    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(df["participant_id"],errors="coerce",)
        df["participant_id"] = df["participant_id"].astype("Int64")

    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = (df["timestamp"].astype(int))

    return df


# ============================================================
# RAW DATA LOADING
# ============================================================

def load_research_raw_data(lifecycles):
    if lifecycles.empty:
        return {}

    match_ids = set(lifecycles["match_id"].astype(str))

    data = {}

    for lane in LANES:
        lane_match_ids = {match_id for match_id in match_ids if determine_lane(match_id) == lane}

        if not lane_match_ids:
            continue

        log("")
        log(f"========== LOADING {lane.upper()} ==========")

        matches = load_table_for_lane(MATCH_TABLE,lane,match_ids=lane_match_ids,)

        participants = load_table_for_lane(PARTICIPANT_TABLE,lane,match_ids=lane_match_ids,)

        snapshots = load_table_for_lane(SNAPSHOT_TABLE,lane,match_ids=lane_match_ids,)

        events = load_table_for_lane(EVENT_TABLE,lane,match_ids=lane_match_ids,)

        matches = prepare_matches(matches)
        participants = prepare_participants(participants)
        snapshots = prepare_snapshots(snapshots)
        events = prepare_events(events)

        log(f"Matches loaded: {len(matches):,}")
        log(f"Participants loaded: {len(participants):,}")
        log(f"Snapshots loaded: {len(snapshots):,}")
        log(f"Events loaded: {len(events):,}")

        data[lane] = {
            "matches": matches,
            "participants": participants,
            "snapshots": snapshots,
            "events": events,
        }
    
    return data
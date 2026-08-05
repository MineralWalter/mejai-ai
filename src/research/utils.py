import pandas as pd
import pyarrow.parquet as pq

from src.research.config import (
    PARQUET_DIR,
    VALID_MATCH_MANIFEST,
)


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


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message)


# ============================================================
# VALID-MATCH MANIFEST
# ============================================================

def load_valid_match_manifest():
    """
    Load and validate the authoritative analysis-eligible
    match manifest.

    The result is cached so it is only read once during a run.
    """

    if not VALID_MATCH_MANIFEST.exists():
        raise FileNotFoundError(
            f"Valid-match manifest not found: {VALID_MATCH_MANIFEST}\n"
            "Run:\n"
            "py -m src.prepare.build_valid_match_manifest"
        )

    manifest = pd.read_parquet(
        VALID_MATCH_MANIFEST,
        engine="pyarrow",
    ).copy()

    required_columns = {
        "match_id",
        "storage_partition",
        "analysis_eligible",
    }

    missing_columns = sorted(
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Valid-match manifest is missing columns: "
            f"{missing_columns}"
        )

    manifest["match_id"] = (
        manifest["match_id"]
        .astype(str)
    )

    manifest["storage_partition"] = (
        manifest["storage_partition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    manifest["analysis_eligible"] = (
        manifest["analysis_eligible"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    duplicate_match_ids = manifest[
        "match_id"
    ].duplicated(
        keep=False
    )

    if duplicate_match_ids.any():
        duplicate_examples = manifest.loc[
            duplicate_match_ids,
            [
                "match_id",
                "storage_partition",
            ],
        ].head(20)

        raise ValueError(
            "Duplicate match IDs found in valid-match manifest:\n"
            f"{duplicate_examples.to_string(index=False)}"
        )

    return manifest

def get_valid_match_ids(lane=None):
    """
    Return analysis-eligible match IDs.

    If lane is supplied, return only eligible matches from that
    storage partition.
    """

    manifest = load_valid_match_manifest()

    eligible = manifest[
        manifest["analysis_eligible"]
    ]

    if lane is not None:
        lane = str(lane).strip().lower()

        if lane not in LANES:
            raise ValueError(
                f"Unknown storage partition: {lane}"
            )

        eligible = eligible[
            eligible["storage_partition"] == lane
        ]

    return frozenset(
        eligible["match_id"]
    )


# ============================================================
# REGION IDENTIFICATION
# ============================================================

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

    if match_id.startswith(
        (
            "VN",
            "SG",
            "TH",
            "PH",
            "ID",
            "MY",
        )
    ):
        return "sea"

    if match_id.startswith(
        (
            "KR",
            "JP",
            "TW",
            "HK",
        )
    ):
        return "asia"

    if match_id.startswith(
        (
            "EUW",
            "EUN",
            "TR",
            "RU",
        )
    ):
        return "europe"

    if match_id.startswith(
        (
            "NA",
            "BR",
            "LA1",
            "LA2",
            "OC",
        )
    ):
        return "americas"

    return None


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_files(table, lane):
    """
    Find all Parquet files belonging to a table and storage
    partition.
    """

    directory = PARQUET_DIR / table

    if not directory.exists():
        return []

    return sorted(
        directory.glob(
            f"{lane}*part*.parquet"
        )
    )


def get_parquet_columns(filepath):
    """
    Inspect a Parquet schema without loading the full file.
    """

    try:
        return set(
            pq.ParquetFile(
                filepath
            ).schema_arrow.names
        )

    except Exception as error:
        raise RuntimeError(
            f"Could not inspect Parquet schema for "
            f"{filepath}: {error}"
        ) from error


# ============================================================
# PARQUET LOADING
# ============================================================

def load_table_for_lane(
    table,
    lane,
    columns=None,
    match_ids=None,
):
    files = find_files(
        table,
        lane,
    )

    if not files:
        return pd.DataFrame()

    normalized_match_ids = None

    if match_ids is not None:
        normalized_match_ids = {
            str(match_id)
            for match_id in match_ids
        }

        if not normalized_match_ids:
            return pd.DataFrame()

    frames = []

    for filepath in files:
        try:
            filters = None

            if normalized_match_ids is not None:
                filters = [
                    (
                        "match_id",
                        "in",
                        list(normalized_match_ids),
                    )
                ]

            if columns is not None:
                available_columns = get_parquet_columns(
                    filepath
                )

                selected_columns = [
                    column
                    for column in columns
                    if column in available_columns
                ]

                if not selected_columns:
                    continue

                frame = pd.read_parquet(
                    filepath,
                    columns=selected_columns,
                    filters=filters,
                    engine="pyarrow",
                )

            else:
                frame = pd.read_parquet(
                    filepath,
                    filters=filters,
                    engine="pyarrow",
                )

            if not frame.empty:
                frames.append(
                    frame
                )

        except Exception as error:
            log(
                f"[ERROR] Could not read "
                f"{filepath}: {error}"
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# MATCH FILTERING
# ============================================================

def filter_to_match_ids(
    df,
    match_ids,
):
    """
    Keep only rows belonging to the requested matches.
    """

    if df.empty:
        return df

    if "match_id" not in df.columns:
        raise ValueError(
            "Cannot filter table because match_id is missing"
        )

    match_ids = {
        str(match_id)
        for match_id in match_ids
    }

    if not match_ids:
        return df.iloc[0:0].copy()

    df = df.copy()
    df["match_id"] = df["match_id"].astype(str)

    return df[
        df["match_id"].isin(
            match_ids
        )
    ].copy()


# ============================================================
# TABLE PREPARATION
# ============================================================

def prepare_matches(df):
    """
    Standardise match table types.
    """

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (
            df["match_id"]
            .astype(str)
        )

    return df


def prepare_participants(df):
    """
    Standardise participant table types.
    """

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (
            df["match_id"]
            .astype(str)
        )

    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(
            df["participant_id"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "participant_id",
            ]
        )

        df["participant_id"] = (
            df["participant_id"]
            .astype(int)
        )

    return df


def prepare_snapshots(df):
    """
    Standardise snapshot table types.

    Snapshot timestamps are milliseconds from game start.
    """

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (
            df["match_id"]
            .astype(str)
        )

    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(
            df["participant_id"],
            errors="coerce",
        )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "participant_id",
            "timestamp",
        ]
    )

    df["participant_id"] = (
        df["participant_id"]
        .astype(int)
    )

    df["timestamp"] = (
        df["timestamp"]
        .astype(int)
    )

    return df


def prepare_events(df):
    """
    Standardise event table types.
    """

    if df.empty:
        return df

    df = df.copy()

    if "match_id" in df.columns:
        df["match_id"] = (
            df["match_id"]
            .astype(str)
        )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce",
        )

    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(
            df["participant_id"],
            errors="coerce",
        ).astype("Int64")

    df = df.dropna(
        subset=[
            "timestamp",
        ]
    )

    df["timestamp"] = (
        df["timestamp"]
        .astype(int)
    )

    return df


# ============================================================
# CASE RAW DATA LOADING
# ============================================================

def load_research_raw_data(lifecycles):
    """
    Load raw data for eligible matches containing Mejai purchase
    cases.

    Lifecycle matches that fail the valid-match manifest are
    automatically excluded.
    """

    if lifecycles.empty:
        return {}

    lifecycle_match_ids = {
        str(match_id)
        for match_id in lifecycles[
            "match_id"
        ]
    }

    all_valid_match_ids = get_valid_match_ids()

    invalid_lifecycle_match_ids = (
        lifecycle_match_ids
        - all_valid_match_ids
    )

    if invalid_lifecycle_match_ids:
        log("")
        log(
            "Lifecycle matches excluded by valid-match "
            f"manifest: {len(invalid_lifecycle_match_ids):,}"
        )

        for match_id in sorted(
            invalid_lifecycle_match_ids
        )[:20]:
            log(
                f"  excluded: {match_id}"
            )

    data = {}

    for lane in LANES:
        valid_lane_match_ids = get_valid_match_ids(
            lane
        )

        lane_match_ids = (
            lifecycle_match_ids
            & valid_lane_match_ids
        )

        if not lane_match_ids:
            continue

        log("")
        log(
            f"========== LOADING "
            f"{lane.upper()} =========="
        )

        log(
            f"Eligible lifecycle matches requested: "
            f"{len(lane_match_ids):,}"
        )

        matches = load_table_for_lane(
            MATCH_TABLE,
            lane,
            match_ids=lane_match_ids,
        )

        participants = load_table_for_lane(
            PARTICIPANT_TABLE,
            lane,
            match_ids=lane_match_ids,
        )

        snapshots = load_table_for_lane(
            SNAPSHOT_TABLE,
            lane,
            match_ids=lane_match_ids,
        )

        events = load_table_for_lane(
            EVENT_TABLE,
            lane,
            match_ids=lane_match_ids,
        )

        matches = prepare_matches(
            matches
        )

        participants = prepare_participants(
            participants
        )

        snapshots = prepare_snapshots(
            snapshots
        )

        events = prepare_events(
            events
        )

        matches = filter_to_match_ids(
            matches,
            lane_match_ids,
        )

        participants = filter_to_match_ids(
            participants,
            lane_match_ids,
        )

        snapshots = filter_to_match_ids(
            snapshots,
            lane_match_ids,
        )

        events = filter_to_match_ids(
            events,
            lane_match_ids,
        )

        log(
            f"Matches loaded: "
            f"{len(matches):,}"
        )

        log(
            f"Participants loaded: "
            f"{len(participants):,}"
        )

        log(
            f"Snapshots loaded: "
            f"{len(snapshots):,}"
        )

        log(
            f"Events loaded: "
            f"{len(events):,}"
        )

        data[lane] = {
            "matches": matches,
            "participants": participants,
            "snapshots": snapshots,
            "events": events,
        }

    return data


# ============================================================
# CONTROL-GROUP RAW DATA LOADING
# ============================================================

def load_control_raw_data(lifecycles):
    """
    Load raw data for eligible matches that do not contain a
    recorded Mejai purchase.
    """

    if lifecycles.empty:
        return {}

    mejai_match_ids = {
        str(match_id)
        for match_id in lifecycles[
            "match_id"
        ]
    }

    data = {}

    for lane in LANES:
        log("")
        log(
            f"========== LOADING "
            f"{lane.upper()} CONTROLS =========="
        )

        matches = load_table_for_lane(
            MATCH_TABLE,
            lane,
        )

        matches = prepare_matches(
            matches
        )

        if matches.empty:
            log(
                f"[WARNING] No match table found "
                f"for {lane}"
            )
            continue

        valid_lane_match_ids = get_valid_match_ids(
            lane
        )

        raw_lane_match_ids = set(
            matches["match_id"]
            .astype(str)
        )

        eligible_lane_match_ids = (
            raw_lane_match_ids
            & valid_lane_match_ids
        )

        excluded_match_ids = (
            raw_lane_match_ids
            - eligible_lane_match_ids
        )

        lane_mejai_match_ids = (
            eligible_lane_match_ids
            & mejai_match_ids
        )

        control_match_ids = (
            eligible_lane_match_ids
            - mejai_match_ids
        )

        log(
            f"Raw matches available: "
            f"{len(raw_lane_match_ids):,}"
        )

        log(
            f"Matches excluded by manifest: "
            f"{len(excluded_match_ids):,}"
        )

        log(
            f"Eligible matches available: "
            f"{len(eligible_lane_match_ids):,}"
        )

        log(
            f"Eligible matches containing Mejai: "
            f"{len(lane_mejai_match_ids):,}"
        )

        log(
            f"Eligible non-Mejai control matches: "
            f"{len(control_match_ids):,}"
        )

        if excluded_match_ids:
            for match_id in sorted(
                excluded_match_ids
            )[:10]:
                log(
                    f"  excluded: {match_id}"
                )

        if not control_match_ids:
            continue

        # ----------------------------------------------------
        # Load participants only from eligible non-Mejai matches.
        # ----------------------------------------------------

        participants = load_table_for_lane(
            PARTICIPANT_TABLE,
            lane,
            match_ids=control_match_ids,
        )

        participants = prepare_participants(
            participants
        )

        participants = filter_to_match_ids(
            participants,
            control_match_ids,
        )

        # ----------------------------------------------------
        # Load snapshots only from eligible non-Mejai matches.
        # ----------------------------------------------------

        snapshots = load_table_for_lane(
            SNAPSHOT_TABLE,
            lane,
            match_ids=control_match_ids,
        )

        snapshots = prepare_snapshots(
            snapshots
        )

        snapshots = filter_to_match_ids(
            snapshots,
            control_match_ids,
        )

        eligible_matches = filter_to_match_ids(
            matches,
            control_match_ids,
        )

        log(
            f"Control matches loaded: "
            f"{len(eligible_matches):,}"
        )

        log(
            f"Control participants loaded: "
            f"{len(participants):,}"
        )

        log(
            f"Control snapshots loaded: "
            f"{len(snapshots):,}"
        )

        data[lane] = {
            "matches": eligible_matches,
            "participants": participants,
            "snapshots": snapshots,
        }

    return data
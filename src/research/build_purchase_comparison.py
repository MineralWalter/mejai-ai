import json

import numpy as np
import pandas as pd

from src.research.config import (
    LIFECYCLE_FILE,
    V2_CONTROL_POOL,
)
from src.research.utils import (
    determine_lane,
    load_research_raw_data,
    load_control_raw_data,
)


# ============================================================
# CONFIG
# ============================================================

MAX_TIME_GAP_MS = 60_000
MAX_CONTROLS_PER_CASE = 30

MAX_LEVEL_GAP = 1

# Primary broad-state candidate calipers.
MAX_PLAYER_GOLD_GAP_RATIO = 0.25
MAX_PLAYER_GOLD_DIFF_VS_ROLE_GAP = 2_500
MAX_REST_OF_TEAM_GOLD_DIFF_GAP = 4_000

# Secondary state variables are retained for ranking and diagnostics,
# but are not used as strict exclusion criteria.
PLAYER_XP_DIFF_SCORE_SCALE = 2_500
REST_OF_TEAM_XP_DIFF_SCORE_SCALE = 4_000
TIME_GAP_SCORE_SCALE_MS = 60_000

TIME_BUCKET_MS = 30_000

PLAYER_STATE_COLUMNS = [
    "current_gold",
    "total_gold",
    "level",
    "xp",
    "minions_killed",
    "jungle_minions_killed",
]

TEAM_SOURCE_COLUMNS = [
    "total_gold",
    "xp",
    "minions_killed",
]


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message)


# ============================================================
# LOAD AND PREPARE INPUTS
# ============================================================

def load_lifecycles():
    if not LIFECYCLE_FILE.exists():
        log(f"[ERROR] Lifecycle file not found: {LIFECYCLE_FILE}")
        return pd.DataFrame()

    try:
        with open(LIFECYCLE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as error:
        log(f"[ERROR] Could not read lifecycle file: {error}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def prepare_lifecycles(df):
    required = [
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "status",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        log(f"[ERROR] Missing lifecycle columns: {missing}")
        return pd.DataFrame()

    df = df.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")
    df["purchase_timestamp"] = pd.to_numeric(
        df["purchase_timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "match_id",
            "participant_id",
            "purchase_timestamp",
            "status",
        ]
    )

    df["participant_id"] = df["participant_id"].astype(int)
    df["purchase_timestamp"] = df["purchase_timestamp"].astype(int)

    return df.reset_index(drop=True)


def prepare_participants(participants):
    if participants.empty:
        return participants

    participants = participants.copy()
    participants["match_id"] = participants["match_id"].astype(str)
    participants["participant_id"] = pd.to_numeric(
        participants["participant_id"],
        errors="coerce",
    )
    participants = participants.dropna(subset=["match_id", "participant_id"])
    participants["participant_id"] = participants["participant_id"].astype(int)

    return participants


def prepare_snapshots(snapshots, participants):
    if snapshots.empty:
        return pd.DataFrame()

    snapshots = snapshots.copy()
    snapshots["match_id"] = snapshots["match_id"].astype(str)
    snapshots["participant_id"] = pd.to_numeric(
        snapshots["participant_id"],
        errors="coerce",
    )
    snapshots["timestamp"] = pd.to_numeric(
        snapshots["timestamp"],
        errors="coerce",
    )

    snapshots = snapshots.dropna(
        subset=["match_id", "participant_id", "timestamp"]
    )
    snapshots["participant_id"] = snapshots["participant_id"].astype(int)
    snapshots["timestamp"] = snapshots["timestamp"].astype(int)

    if not participants.empty:
        metadata_columns = ["match_id", "participant_id"]

        for column in [
            "team_id",
            "team_position",
            "champion_id",
            "champion_name",
            "win",
        ]:
            if column in participants.columns:
                metadata_columns.append(column)

        metadata = participants[metadata_columns].drop_duplicates(
            subset=["match_id", "participant_id"]
        )

        already_attached = [
            column
            for column in metadata_columns
            if column not in ["match_id", "participant_id"]
            and column in snapshots.columns
        ]

        if already_attached:
            snapshots = snapshots.drop(columns=already_attached)

        snapshots = snapshots.merge(
            metadata,
            on=["match_id", "participant_id"],
            how="left",
            validate="many_to_one",
        )

    return snapshots


# ============================================================
# VECTORIZED TEAM-STATE FEATURES
# ============================================================

def add_team_state_features(snapshots):
    """
    Add whole-team state and same-role opponent state.

    `enemy_*` remains the whole opposing-team aggregate. The
    `role_opponent_*` fields identify the player at the same
    team_position on the opposing team.
    """

    if snapshots.empty:
        return snapshots

    required = [
        "match_id",
        "timestamp",
        "team_id",
        "participant_id",
        "team_position",
        "total_gold",
        "xp",
    ]
    missing = [column for column in required if column not in snapshots.columns]

    if missing:
        log(f"[ERROR] Snapshot table missing state columns: {missing}")
        return pd.DataFrame()

    snapshots = snapshots.dropna(
        subset=["team_id", "participant_id", "team_position"]
    ).copy()

    snapshots["team_id"] = pd.to_numeric(snapshots["team_id"], errors="coerce")
    snapshots["participant_id"] = pd.to_numeric(
        snapshots["participant_id"], errors="coerce"
    )
    snapshots = snapshots.dropna(subset=["team_id", "participant_id"])
    snapshots["team_id"] = snapshots["team_id"].astype(int)
    snapshots["participant_id"] = snapshots["participant_id"].astype(int)
    snapshots["team_position"] = (
        snapshots["team_position"].astype(str).str.strip().str.upper()
    )

    team_keys = ["match_id", "timestamp", "team_id"]
    match_keys = ["match_id", "timestamp"]

    source_map = {
        "total_gold": "total_gold",
        "xp": "xp",
        "minions_killed": "cs",
    }

    for source_column, output_name in source_map.items():
        if source_column not in snapshots.columns:
            continue

        team_total = snapshots.groupby(
            team_keys,
            sort=False,
            observed=True,
        )[source_column].transform("sum")

        match_total = snapshots.groupby(
            match_keys,
            sort=False,
            observed=True,
        )[source_column].transform("sum")

        enemy_total = match_total - team_total

        snapshots[f"team_{output_name}"] = team_total
        snapshots[f"enemy_{output_name}"] = enemy_total
        snapshots[f"team_{output_name}_diff"] = team_total - enemy_total

    opponent_columns = [
        "match_id",
        "timestamp",
        "team_position",
        "team_id",
        "participant_id",
        "total_gold",
        "xp",
    ]
    opponents = snapshots[opponent_columns].rename(
        columns={
            "team_id": "role_opponent_team_id",
            "participant_id": "role_opponent_participant_id",
            "total_gold": "role_opponent_total_gold",
            "xp": "role_opponent_xp",
        }
    )

    snapshots = snapshots.merge(
        opponents,
        on=["match_id", "timestamp", "team_position"],
        how="left",
    )
    snapshots = snapshots[
        snapshots["team_id"] != snapshots["role_opponent_team_id"]
    ].copy()

    snapshots = snapshots.sort_values(
        [
            "match_id",
            "timestamp",
            "participant_id",
            "role_opponent_participant_id",
        ],
        kind="stable",
    ).drop_duplicates(
        subset=["match_id", "timestamp", "participant_id"],
        keep="first",
    )

    snapshots["player_gold_diff_vs_role_opponent"] = (
        snapshots["total_gold"] - snapshots["role_opponent_total_gold"]
    )
    snapshots["player_xp_diff_vs_role_opponent"] = (
        snapshots["xp"] - snapshots["role_opponent_xp"]
    )
    snapshots["rest_of_team_gold_diff"] = (
        snapshots["team_total_gold_diff"]
        - snapshots["player_gold_diff_vs_role_opponent"]
    )
    snapshots["rest_of_team_xp_diff"] = (
        snapshots["team_xp_diff"]
        - snapshots["player_xp_diff_vs_role_opponent"]
    )

    snapshots["time_bucket"] = snapshots["timestamp"] // TIME_BUCKET_MS

    return snapshots


# ============================================================
# CASE SNAPSHOT LOOKUP
# ============================================================

def build_snapshot_lookup(snapshots):
    if snapshots.empty:
        return {}

    snapshots = snapshots.sort_values(
        ["match_id", "participant_id", "timestamp"]
    )

    return {
        key: group.reset_index(drop=True)
        for key, group in snapshots.groupby(
            ["match_id", "participant_id"],
            sort=False,
            observed=True,
        )
    }


def get_player_snapshot(
    snapshot_lookup,
    match_id,
    participant_id,
    target_timestamp,
):
    player = snapshot_lookup.get((str(match_id), int(participant_id)))

    if player is None or player.empty:
        return None

    timestamps = player["timestamp"].to_numpy()
    position = np.searchsorted(
        timestamps,
        int(target_timestamp),
        side="right",
    ) - 1

    if position < 0:
        return None

    return player.iloc[position]


# ============================================================
# CONTROL INDEX
# ============================================================

def build_control_index(control_snapshots):
    """
    Return row-index arrays keyed by:

        (team_position, snapshot_level, time_bucket)

    The region is already separated by the caller, so it is not
    needed in the key.
    """

    required = ["team_position", "level", "time_bucket"]
    missing = [column for column in required if column not in control_snapshots]

    if missing:
        log(f"[ERROR] Control snapshots missing index columns: {missing}")
        return {}

    indexed = control_snapshots.dropna(subset=required).copy()
    indexed["level"] = indexed["level"].astype(int)

    return indexed.groupby(
        ["team_position", "level", "time_bucket"],
        sort=False,
        observed=True,
    ).indices


def get_control_row_indices(
    control_index,
    team_position,
    case_level,
    target_timestamp,
):
    minimum_bucket = max(
        0,
        (int(target_timestamp) - MAX_TIME_GAP_MS) // TIME_BUCKET_MS,
    )
    maximum_bucket = int(target_timestamp) // TIME_BUCKET_MS

    index_parts = []

    for level in range(
        max(1, int(case_level) - MAX_LEVEL_GAP),
        int(case_level) + MAX_LEVEL_GAP + 1,
    ):
        for bucket in range(minimum_bucket, maximum_bucket + 1):
            rows = control_index.get((team_position, level, bucket))
            if rows is not None and len(rows):
                index_parts.append(rows)

    if not index_parts:
        return np.empty(0, dtype=np.int64)

    return np.concatenate(index_parts)


# ============================================================
# MATCHING
# ============================================================

def relative_gap_array(values, reference):
    values = values.astype(float)
    reference = float(reference)
    denominator = np.maximum.reduce(
        [
            np.abs(values),
            np.full(len(values), abs(reference), dtype=float),
            np.ones(len(values), dtype=float),
        ]
    )
    return np.abs(values - reference) / denominator


def score_control_candidates(case_snapshot, candidates, target_timestamp):
    """
    Build a broad candidate pool using a compact generalized game state.

    Hard candidate restrictions:
        - same region, handled by the caller
        - same position and nearby level, handled by the index
        - control snapshot at or before the case purchase time
        - player total gold
        - player gold advantage versus same-role opponent
        - rest-of-team gold advantage

    XP-based state is used for ranking only, not strict exclusion.
    """

    if candidates.empty:
        return candidates

    required_case_values = {
        "level": case_snapshot.get("level", np.nan),
        "total_gold": case_snapshot.get("total_gold", np.nan),
        "player_gold_diff_vs_role_opponent": case_snapshot.get(
            "player_gold_diff_vs_role_opponent", np.nan
        ),
        "player_xp_diff_vs_role_opponent": case_snapshot.get(
            "player_xp_diff_vs_role_opponent", np.nan
        ),
        "rest_of_team_gold_diff": case_snapshot.get(
            "rest_of_team_gold_diff", np.nan
        ),
        "rest_of_team_xp_diff": case_snapshot.get(
            "rest_of_team_xp_diff", np.nan
        ),
    }

    if any(pd.isna(value) for value in required_case_values.values()):
        return pd.DataFrame()

    required_control_columns = [
        "timestamp",
        "match_id",
        "participant_id",
        "level",
        "total_gold",
        "player_gold_diff_vs_role_opponent",
        "player_xp_diff_vs_role_opponent",
        "rest_of_team_gold_diff",
        "rest_of_team_xp_diff",
    ]
    missing = [
        column for column in required_control_columns
        if column not in candidates.columns
    ]
    if missing:
        return pd.DataFrame()

    candidates = candidates.copy()
    candidates = candidates[
        candidates["timestamp"].between(
            int(target_timestamp) - MAX_TIME_GAP_MS,
            int(target_timestamp),
            inclusive="both",
        )
    ]

    if candidates.empty:
        return candidates

    candidates = (
        candidates.sort_values("timestamp")
        .drop_duplicates(
            subset=["match_id", "participant_id"],
            keep="last",
        )
        .copy()
    )

    level_gap = np.abs(
        candidates["level"].to_numpy(dtype=float)
        - float(required_case_values["level"])
    )
    player_gold_gap = relative_gap_array(
        candidates["total_gold"].to_numpy(dtype=float),
        required_case_values["total_gold"],
    )
    player_gold_diff_gap = np.abs(
        candidates["player_gold_diff_vs_role_opponent"].to_numpy(dtype=float)
        - float(required_case_values["player_gold_diff_vs_role_opponent"])
    )
    rest_team_gold_diff_gap = np.abs(
        candidates["rest_of_team_gold_diff"].to_numpy(dtype=float)
        - float(required_case_values["rest_of_team_gold_diff"])
    )

    valid = (
        (level_gap <= MAX_LEVEL_GAP)
        & (player_gold_gap <= MAX_PLAYER_GOLD_GAP_RATIO)
        & (player_gold_diff_gap <= MAX_PLAYER_GOLD_DIFF_VS_ROLE_GAP)
        & (rest_team_gold_diff_gap <= MAX_REST_OF_TEAM_GOLD_DIFF_GAP)
    )

    if not valid.any():
        return pd.DataFrame()

    candidates = candidates.loc[valid].copy()

    time_gap = (
        int(target_timestamp)
        - candidates["timestamp"].to_numpy(dtype=float)
    )
    player_xp_diff_gap = np.abs(
        candidates["player_xp_diff_vs_role_opponent"].to_numpy(dtype=float)
        - float(required_case_values["player_xp_diff_vs_role_opponent"])
    )
    rest_team_xp_diff_gap = np.abs(
        candidates["rest_of_team_xp_diff"].to_numpy(dtype=float)
        - float(required_case_values["rest_of_team_xp_diff"])
    )

    candidates["control_match_state_score"] = (
        player_gold_gap[valid] / MAX_PLAYER_GOLD_GAP_RATIO
        + player_gold_diff_gap[valid] / MAX_PLAYER_GOLD_DIFF_VS_ROLE_GAP
        + rest_team_gold_diff_gap[valid] / MAX_REST_OF_TEAM_GOLD_DIFF_GAP
        + 0.35 * level_gap[valid] / max(MAX_LEVEL_GAP, 1)
        + 0.25 * time_gap / TIME_GAP_SCORE_SCALE_MS
        + 0.20 * player_xp_diff_gap / PLAYER_XP_DIFF_SCORE_SCALE
        + 0.20 * rest_team_xp_diff_gap / REST_OF_TEAM_XP_DIFF_SCORE_SCALE
    )

    candidates["control_snapshot_age_ms"] = (
        int(target_timestamp) - candidates["timestamp"]
    )

    return candidates.nsmallest(
        MAX_CONTROLS_PER_CASE,
        "control_match_state_score",
    )


# ============================================================
# OUTPUT ROW CONSTRUCTION
# ============================================================

def build_output_row(mejai_case, case_snapshot, control_snapshot):
    case_match_id = str(mejai_case["match_id"])
    case_participant_id = int(mejai_case["participant_id"])
    purchase_timestamp = int(mejai_case["purchase_timestamp"])

    row = {
        "case_id": (
            f"{case_match_id}_"
            f"{case_participant_id}_"
            f"{purchase_timestamp}"
        ),
        "match_id": case_match_id,
        "mejai_participant_id": case_participant_id,
        "mejai_purchase_timestamp": purchase_timestamp,
        "mejai_lifecycle_status": mejai_case["status"],
        "control_match_id": str(control_snapshot["match_id"]),
        "control_participant_id": int(control_snapshot["participant_id"]),
        "control_snapshot_timestamp": int(control_snapshot["timestamp"]),
        "control_snapshot_age_ms": int(
            control_snapshot["control_snapshot_age_ms"]
        ),
        "control_time_seconds": float(control_snapshot["timestamp"]) / 1000,
        "control_match_state_score": float(
            control_snapshot["control_match_state_score"]
        ),
        "control_group": 1,
        "team_position": control_snapshot.get("team_position", None),
        "champion_id": control_snapshot.get("champion_id", None),
        "champion_name": control_snapshot.get("champion_name", None),
        "team_id": control_snapshot.get("team_id", None),
        "outcome_win": control_snapshot.get("win", None),
    }

    for column in PLAYER_STATE_COLUMNS:
        if column in control_snapshot.index:
            row[f"control_player_{column}"] = control_snapshot[column]
        if column in case_snapshot.index:
            row[f"mejai_player_{column}"] = case_snapshot[column]

    for column in [
        "role_opponent_participant_id",
        "role_opponent_total_gold",
        "role_opponent_xp",
        "player_gold_diff_vs_role_opponent",
        "player_xp_diff_vs_role_opponent",
        "rest_of_team_gold_diff",
        "rest_of_team_xp_diff",
        "team_total_gold",
        "enemy_total_gold",
        "team_total_gold_diff",
        "team_xp",
        "enemy_xp",
        "team_xp_diff",
        "team_cs",
        "enemy_cs",
        "team_cs_diff",
    ]:
        if column in control_snapshot.index:
            row[f"control_{column}"] = control_snapshot[column]
        if column in case_snapshot.index:
            row[f"mejai_{column}"] = case_snapshot[column]

    return row


# ============================================================
# BUILD CONTROL DATASET
# ============================================================

def build_control_dataset(lifecycles, case_data, control_data):
    output_rows = []
    total_cases = len(lifecycles)
    processed_cases = 0
    matched_cases = 0

    log(f"Mejai purchase cases available: {total_cases:,}")

    for lane, case_tables in case_data.items():
        if lane not in control_data:
            continue

        lane_cases = lifecycles[
            lifecycles["match_id"].map(determine_lane) == lane
        ]

        if lane_cases.empty:
            continue

        log("")
        log(f"========== MATCHING {lane.upper()} ==========")

        # ----------------------------------------------------
        # Prepare case snapshots.
        # ----------------------------------------------------

        case_participants = prepare_participants(case_tables["participants"])
        case_snapshots = prepare_snapshots(
            case_tables["snapshots"],
            case_participants,
        )
        case_snapshots = add_team_state_features(case_snapshots)

        if case_snapshots.empty:
            log(f"[WARNING] No usable case snapshots for {lane}")
            continue

        case_snapshot_lookup = build_snapshot_lookup(case_snapshots)

        # ----------------------------------------------------
        # Prepare control snapshots with vectorized team state.
        # ----------------------------------------------------

        control_tables = control_data[lane]
        control_participants = prepare_participants(
            control_tables["participants"]
        )
        control_snapshots = prepare_snapshots(
            control_tables["snapshots"],
            control_participants,
        )

        log(f"Preparing vectorized control state for {len(control_snapshots):,} snapshots...")
        control_snapshots = add_team_state_features(control_snapshots)

        required_control_columns = [
            "match_id",
            "participant_id",
            "timestamp",
            "team_position",
            "level",
            "total_gold",
            "team_total_gold",
            "team_xp",
            "team_cs",
            "team_total_gold_diff",
            "team_xp_diff",
            "team_cs_diff",
            "role_opponent_participant_id",
            "role_opponent_total_gold",
            "role_opponent_xp",
            "player_gold_diff_vs_role_opponent",
            "player_xp_diff_vs_role_opponent",
            "rest_of_team_gold_diff",
            "rest_of_team_xp_diff",
            "time_bucket",
        ]

        missing = [
            column
            for column in required_control_columns
            if column not in control_snapshots.columns
        ]

        if missing:
            log(f"[ERROR] Control snapshots missing columns for {lane}: {missing}")
            continue

        control_snapshots = control_snapshots.dropna(
            subset=[
                "team_position",
                "level",
                "total_gold",
                "team_total_gold",
                "team_xp",
                "team_cs",
                "team_total_gold_diff",
                "team_xp_diff",
                "team_cs_diff",
                "role_opponent_participant_id",
                "role_opponent_total_gold",
                "role_opponent_xp",
                "player_gold_diff_vs_role_opponent",
                "player_xp_diff_vs_role_opponent",
                "rest_of_team_gold_diff",
                "rest_of_team_xp_diff",
            ]
        ).reset_index(drop=True)

        log(f"Building control index for {len(control_snapshots):,} usable snapshots...")
        control_index = build_control_index(control_snapshots)
        log(f"Control index buckets: {len(control_index):,}")

        # ----------------------------------------------------
        # Match each Mejai purchase.
        # ----------------------------------------------------

        for _, mejai_case in lane_cases.iterrows():
            case_snapshot = get_player_snapshot(
                case_snapshot_lookup,
                mejai_case["match_id"],
                mejai_case["participant_id"],
                mejai_case["purchase_timestamp"],
            )

            processed_cases += 1

            if case_snapshot is None:
                continue

            snapshot_age = (
                int(mejai_case["purchase_timestamp"])
                - int(case_snapshot["timestamp"])
            )

            if snapshot_age < 0 or snapshot_age > MAX_TIME_GAP_MS:
                continue

            case_position = case_snapshot.get("team_position", None)
            case_level = case_snapshot.get("level", np.nan)

            if case_position is None or pd.isna(case_level):
                continue

            row_indices = get_control_row_indices(
                control_index,
                case_position,
                int(case_level),
                int(mejai_case["purchase_timestamp"]),
            )

            if len(row_indices) == 0:
                continue

            candidate_rows = control_snapshots.iloc[row_indices]

            matches = score_control_candidates(
                case_snapshot,
                candidate_rows,
                int(mejai_case["purchase_timestamp"]),
            )

            if matches.empty:
                continue

            matched_cases += 1

            for _, control_snapshot in matches.iterrows():
                output_rows.append(
                    build_output_row(
                        mejai_case,
                        case_snapshot,
                        control_snapshot,
                    )
                )

            if processed_cases % 1000 == 0:
                log(
                    f"Cases processed: {processed_cases:,} / {total_cases:,} | "
                    f"matched: {matched_cases:,}"
                )

        # Release large lane-specific structures before the next region.
        del control_index
        del control_snapshots
        del case_snapshot_lookup
        del case_snapshots

    log(f"Cases processed: {processed_cases:,} / {total_cases:,}")
    log(f"Cases with at least one control: {matched_cases:,}")

    return pd.DataFrame(output_rows)


# ============================================================
# SUMMARY AND SAVE
# ============================================================

def print_summary(df):
    log("")
    log("=" * 70)
    log("GENERALISED CONTROL CANDIDATE POOL SUMMARY")
    log("=" * 70)
    log(f"Control candidate rows: {len(df):,}")

    if df.empty:
        return

    unique_control_players = df[
        ["control_match_id", "control_participant_id"]
    ].drop_duplicates()

    controls_per_case = df.groupby("case_id").size()

    log(f"Unique Mejai purchase cases represented: {df['case_id'].nunique():,}")
    log(f"Unique control matches represented: {df['control_match_id'].nunique():,}")
    log(f"Unique control players represented: {len(unique_control_players):,}")

    log("")
    log("Controls retained per case:")
    log(controls_per_case.value_counts().sort_index().to_string())

    log("")
    log("Control outcome:")
    log(df["outcome_win"].value_counts(dropna=False).to_string())

    log("")
    log("Control champion distribution:")
    log(df["champion_name"].value_counts().head(15).to_string())

    log("")
    log("Control snapshot age:")
    log(df["control_snapshot_age_ms"].describe().to_string())

    log("")
    log("Control matching score:")
    log(df["control_match_state_score"].describe().to_string())


def save_dataset(df):
    V2_CONTROL_POOL.parent.mkdir(parents=True,exist_ok=True,)

    df.to_parquet(V2_CONTROL_POOL,index=False,)

    log(f"[SAVED] Control candidates written to: "f"{V2_CONTROL_POOL}")


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 70)
    log("GENERALISED MEJAI CONTROL CANDIDATE POOL")
    log("=" * 70)

    lifecycles = prepare_lifecycles(load_lifecycles())

    if lifecycles.empty:
        log("[ERROR] No usable lifecycle data")
        return

    log("")
    log("Loading Mejai case data...")
    case_data = load_research_raw_data(lifecycles)

    if not case_data:
        log("[ERROR] No Mejai case data loaded")
        return

    log("")
    log("Loading non-Mejai control data...")
    control_data = load_control_raw_data(lifecycles)

    if not control_data:
        log("[ERROR] No non-Mejai control data loaded")
        return

    log("")
    log("Building generalized control candidate pool...")
    controls = build_control_dataset(
        lifecycles,
        case_data,
        control_data,
    )

    if controls.empty:
        log("[ERROR] No control candidates could be constructed")
        return

    controls = controls.drop_duplicates(
        subset=[
            "case_id",
            "control_match_id",
            "control_participant_id",
        ]
    ).reset_index(drop=True)

    print_summary(controls)
    save_dataset(controls)

    log("")
    log("[PASSED] GENERALISED CONTROL CANDIDATE POOL CONSTRUCTED")


if __name__ == "__main__":
    main()
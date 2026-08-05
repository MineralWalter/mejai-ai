import json
from pathlib import Path

import pandas as pd

from src.research.config import (
    CASE_DATASET,
    LIFECYCLE_FILE,
)
from src.research.utils import (
    determine_lane,
    get_valid_match_ids,
    load_research_raw_data,
)


MAX_SNAPSHOT_AGE_MS = 60_000


def log(message):
    print(message)


# ============================================================
# LIFECYCLE LOADING
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
    required_columns = [
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

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
    df["status"] = df["status"].astype(str).str.strip().str.upper()

    duplicate_cases = df.duplicated(
        subset=[
            "match_id",
            "participant_id",
            "purchase_timestamp",
        ],
        keep=False,
    )

    if duplicate_cases.any():
        examples = df.loc[
            duplicate_cases,
            [
                "match_id",
                "participant_id",
                "purchase_timestamp",
                "status",
            ],
        ].head(20)

        raise ValueError(
            "Duplicate lifecycle cases found:\n"
            + examples.to_string(index=False)
        )

    valid_match_ids = get_valid_match_ids()
    invalid_match_ids = set(df["match_id"]) - set(valid_match_ids)

    if invalid_match_ids:
        raise ValueError(
            "Lifecycle data contains invalid matches:\n"
            + "\n".join(sorted(invalid_match_ids)[:20])
        )

    return df.reset_index(drop=True)


# ============================================================
# LOOKUPS
# ============================================================

def build_participant_lookup(participants):
    if participants.empty:
        return {}

    lookup = {}

    for _, row in participants.iterrows():
        key = (
            str(row["match_id"]),
            int(row["participant_id"]),
        )
        lookup[key] = row

    return lookup


def build_match_lookup(matches):
    if matches.empty:
        return {}

    lookup = {}

    for _, row in matches.iterrows():
        lookup[str(row["match_id"])] = row

    return lookup


# ============================================================
# SNAPSHOT FEATURES
# ============================================================

def build_snapshot_index(snapshots, participants):
    if snapshots.empty:
        return {
            "by_participant": {},
            "by_timestamp": {},
        }

    snapshots = snapshots.copy()

    if not participants.empty:
        team_lookup = (
            participants[
                [
                    "match_id",
                    "participant_id",
                    "team_id",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "match_id",
                    "participant_id",
                ]
            )
        )

        snapshots = snapshots.merge(
            team_lookup,
            on=[
                "match_id",
                "participant_id",
            ],
            how="left",
            validate="many_to_one",
        )

    snapshots = snapshots.sort_values(
        [
            "participant_id",
            "timestamp",
        ],
        kind="stable",
    )

    by_participant = {
        int(participant_id): group.reset_index(drop=True)
        for participant_id, group in snapshots.groupby(
            "participant_id",
            sort=False,
        )
    }

    by_timestamp = {
        int(timestamp): group
        for timestamp, group in snapshots.groupby(
            "timestamp",
            sort=False,
        )
    }

    return {
        "by_participant": by_participant,
        "by_timestamp": by_timestamp,
    }


def get_latest_player_snapshot(
    snapshot_index,
    participant_id,
    purchase_timestamp,
):
    player_snapshots = snapshot_index["by_participant"].get(
        int(participant_id)
    )

    if player_snapshots is None or player_snapshots.empty:
        return None

    timestamps = player_snapshots["timestamp"].to_numpy()

    position = timestamps.searchsorted(
        purchase_timestamp,
        side="right",
    ) - 1

    if position < 0:
        return None

    return player_snapshots.iloc[position]


def build_snapshot_features(
    snapshot_index,
    participant_id,
    player_team_id,
    purchase_timestamp,
):
    player_snapshot = get_latest_player_snapshot(
        snapshot_index,
        participant_id,
        purchase_timestamp,
    )

    if player_snapshot is None:
        return None

    features = {}
    snapshot_timestamp = int(player_snapshot["timestamp"])

    features["snapshot_timestamp"] = snapshot_timestamp
    features["snapshot_age_ms"] = purchase_timestamp - snapshot_timestamp

    player_columns = [
        "current_gold",
        "total_gold",
        "level",
        "xp",
        "minions_killed",
        "jungle_minions_killed",
        "position_x",
        "position_y",
    ]

    for column in player_columns:
        if column in player_snapshot.index:
            features[f"player_{column}"] = player_snapshot[column]

    same_timestamp = snapshot_index["by_timestamp"].get(snapshot_timestamp)

    if same_timestamp is None or same_timestamp.empty:
        return features

    if "team_id" not in same_timestamp.columns:
        return features

    team_snapshots = same_timestamp[
        same_timestamp["team_id"] == player_team_id
    ]
    enemy_snapshots = same_timestamp[
        same_timestamp["team_id"] != player_team_id
    ]

    aggregate_columns = [
        "current_gold",
        "total_gold",
        "xp",
        "minions_killed",
        "jungle_minions_killed",
    ]

    for column in aggregate_columns:
        if column in team_snapshots.columns:
            features[f"team_{column}_sum"] = team_snapshots[column].sum()

        if column in enemy_snapshots.columns:
            features[f"enemy_{column}_sum"] = enemy_snapshots[column].sum()

    if (
        "total_gold" in team_snapshots.columns
        and "total_gold" in enemy_snapshots.columns
    ):
        features["team_total_gold_diff"] = (
            team_snapshots["total_gold"].sum()
            - enemy_snapshots["total_gold"].sum()
        )

    if (
        "current_gold" in team_snapshots.columns
        and "current_gold" in enemy_snapshots.columns
    ):
        features["team_current_gold_diff"] = (
            team_snapshots["current_gold"].sum()
            - enemy_snapshots["current_gold"].sum()
        )

    if "xp" in team_snapshots.columns and "xp" in enemy_snapshots.columns:
        features["team_xp_diff"] = (
            team_snapshots["xp"].sum()
            - enemy_snapshots["xp"].sum()
        )

    if (
        "minions_killed" in team_snapshots.columns
        and "minions_killed" in enemy_snapshots.columns
    ):
        features["team_cs_diff"] = (
            team_snapshots["minions_killed"].sum()
            - enemy_snapshots["minions_killed"].sum()
        )

    return features


# ============================================================
# PLAYER / MATCH CONTEXT
# ============================================================

def build_player_context(participant_row):
    if participant_row is None:
        return {}

    features = {}
    allowed_columns = [
        "team_id",
        "team_position",
        "champion_id",
        "champion_name",
    ]

    for column in allowed_columns:
        if column in participant_row.index:
            features[column] = participant_row[column]

    return features


def build_match_context(match_row):
    if match_row is None:
        return {}

    features = {}
    allowed_columns = [
        "game_version",
        "queue_id",
        "map_id",
    ]

    for column in allowed_columns:
        if column in match_row.index:
            features[column] = match_row[column]

    return features


# ============================================================
# OUTCOME VARIABLES
# ============================================================

def build_outcomes(match_row, participant_row):
    outcomes = {}

    if participant_row is not None:
        if "win" in participant_row.index:
            outcomes["outcome_win"] = participant_row["win"]

        final_columns = [
            "gold_earned",
            "gold_spent",
            "champ_level",
            "champ_experience",
            "kills",
            "deaths",
            "assists",
            "damage_dealt_to_champions",
            "damage_taken",
        ]

        for column in final_columns:
            if column in participant_row.index:
                outcomes[f"outcome_final_{column}"] = participant_row[column]

    if match_row is not None:
        if "game_duration" in match_row.index:
            outcomes["outcome_game_duration"] = match_row["game_duration"]

        if "end_of_game_result" in match_row.index:
            outcomes["outcome_game_result"] = match_row["end_of_game_result"]

    return outcomes


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

def validate_temporal_order(case):
    required_columns = [
        "snapshot_timestamp",
        "purchase_timestamp",
        "snapshot_age_ms",
    ]

    if any(column not in case for column in required_columns):
        return False

    snapshot_timestamp = int(case["snapshot_timestamp"])
    purchase_timestamp = int(case["purchase_timestamp"])
    snapshot_age_ms = int(case["snapshot_age_ms"])

    return (
        snapshot_timestamp <= purchase_timestamp
        and 0 <= snapshot_age_ms <= MAX_SNAPSHOT_AGE_MS
    )


# ============================================================
# DATASET BUILDING
# ============================================================

def build_dataset(lifecycles, data):
    cases = []
    failures = {}

    for lane, tables in data.items():
        lane_lifecycles = lifecycles[
            lifecycles["match_id"].astype(str).map(determine_lane) == lane
        ]

        if lane_lifecycles.empty:
            continue

        matches = tables["matches"]
        participants = tables["participants"]
        snapshots = tables["snapshots"]

        match_lookup = build_match_lookup(matches)
        participant_lookup = build_participant_lookup(participants)
        snapshot_indexes = {}

        if not snapshots.empty:
            for match_id, match_snapshots in snapshots.groupby(
                "match_id",
                sort=False,
            ):
                match_participants = participants[
                    participants["match_id"].astype(str) == str(match_id)
                ]

                snapshot_indexes[str(match_id)] = build_snapshot_index(
                    match_snapshots,
                    match_participants,
                )

        for _, lifecycle in lane_lifecycles.iterrows():
            match_id = str(lifecycle["match_id"])
            participant_id = int(lifecycle["participant_id"])
            purchase_timestamp = int(lifecycle["purchase_timestamp"])

            match_row = match_lookup.get(match_id)

            if match_row is None:
                failures["missing_match"] = failures.get("missing_match", 0) + 1
                continue

            participant_row = participant_lookup.get(
                (
                    match_id,
                    participant_id,
                )
            )

            if participant_row is None:
                failures["missing_participant"] = (
                    failures.get("missing_participant", 0) + 1
                )
                continue

            if (
                "team_id" not in participant_row.index
                or pd.isna(participant_row["team_id"])
            ):
                failures["missing_team_id"] = (
                    failures.get("missing_team_id", 0) + 1
                )
                continue

            player_team_id = int(participant_row["team_id"])
            snapshot_index = snapshot_indexes.get(match_id)

            if snapshot_index is None:
                failures["missing_snapshots"] = (
                    failures.get("missing_snapshots", 0) + 1
                )
                continue

            snapshot_features = build_snapshot_features(
                snapshot_index,
                participant_id,
                player_team_id,
                purchase_timestamp,
            )

            if snapshot_features is None:
                failures["missing_pre_purchase_snapshot"] = (
                    failures.get("missing_pre_purchase_snapshot", 0) + 1
                )
                continue

            case = {
                "case_id": (
                    f"{match_id}_"
                    f"{participant_id}_"
                    f"{purchase_timestamp}"
                ),
                "match_id": match_id,
                "participant_id": participant_id,
                "region": lane,
                "purchase_timestamp": purchase_timestamp,
                "purchase_time_seconds": purchase_timestamp / 1000,
                "lifecycle_status": lifecycle["status"],
            }

            case.update(build_match_context(match_row))
            case.update(build_player_context(participant_row))
            case.update(snapshot_features)
            case.update(build_outcomes(match_row, participant_row))

            if not validate_temporal_order(case):
                failures["invalid_snapshot_timing"] = (
                    failures.get("invalid_snapshot_timing", 0) + 1
                )
                continue

            cases.append(case)

            if len(cases) % 1000 == 0:
                log(f"Cases processed: {len(cases):,}")

    return pd.DataFrame(cases), failures


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_dataset(dataset, lifecycles, failures):
    if dataset.empty:
        raise ValueError("Research dataset is empty")

    required_columns = {
        "case_id",
        "match_id",
        "participant_id",
        "region",
        "purchase_timestamp",
        "purchase_time_seconds",
        "lifecycle_status",
        "snapshot_timestamp",
        "snapshot_age_ms",
        "team_id",
        "team_position",
        "outcome_win",
        "outcome_game_result",
    }

    missing_columns = sorted(required_columns - set(dataset.columns))

    if missing_columns:
        raise ValueError(
            f"Research dataset is missing columns: {missing_columns}"
        )

    if dataset["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs found in research dataset")

    expected_case_ids = {
        (
            f"{row.match_id}_"
            f"{int(row.participant_id)}_"
            f"{int(row.purchase_timestamp)}"
        )
        for row in lifecycles.itertuples(index=False)
    }

    actual_case_ids = set(dataset["case_id"].astype(str))
    missing_case_ids = expected_case_ids - actual_case_ids
    unexpected_case_ids = actual_case_ids - expected_case_ids

    if missing_case_ids:
        examples = "\n".join(sorted(missing_case_ids)[:20])
        raise ValueError(
            f"{len(missing_case_ids):,} lifecycle cases were not constructed.\n"
            f"Examples:\n{examples}\n"
            f"Recorded failures: {failures}"
        )

    if unexpected_case_ids:
        examples = "\n".join(sorted(unexpected_case_ids)[:20])
        raise ValueError(
            f"{len(unexpected_case_ids):,} unexpected cases were constructed.\n"
            f"Examples:\n{examples}"
        )

    valid_match_ids = get_valid_match_ids()
    invalid_matches = set(dataset["match_id"].astype(str)) - set(valid_match_ids)

    if invalid_matches:
        raise ValueError(
            "Dataset contains invalid matches:\n"
            + "\n".join(sorted(invalid_matches)[:20])
        )

    if not dataset["participant_id"].between(1, 10).all():
        raise ValueError("Dataset contains participant IDs outside 1-10")

    calculated_age = (
        dataset["purchase_timestamp"]
        - dataset["snapshot_timestamp"]
    )

    if not calculated_age.eq(dataset["snapshot_age_ms"]).all():
        raise ValueError("Stored snapshot ages do not match the timestamps")

    if not dataset["snapshot_age_ms"].between(
        0,
        MAX_SNAPSHOT_AGE_MS,
        inclusive="both",
    ).all():
        raise ValueError(
            "Dataset contains future snapshots or snapshots older than 60 seconds"
        )

    if not dataset["queue_id"].eq(420).all():
        raise ValueError("Dataset contains non-ranked-solo cases")

    if not dataset["outcome_game_result"].eq("GameComplete").all():
        raise ValueError("Dataset contains incomplete matches")

    if dataset["outcome_win"].isna().any():
        raise ValueError("Dataset contains missing win outcomes")

    if failures:
        raise ValueError(
            "Case construction recorded failures:\n"
            f"{failures}"
        )

    return dataset


# ============================================================
# REPORTING AND OUTPUT
# ============================================================

def print_summary(df, failures):
    log("========== DATASET SUMMARY ==========")
    log(f"Research cases: {len(df):,}")
    log(f"Columns: {len(df.columns):,}")

    if not df.empty:
        log(f"Regions: {df['region'].nunique():,}")
        log("Lifecycle statuses:")

        for status, count in df["lifecycle_status"].value_counts().items():
            log(f"{status}: {count:,}")

        log("")
        log("Snapshot age:")
        log(df["snapshot_age_ms"].describe().to_string())

    if failures:
        log("========== BUILD EXCLUSIONS ==========")

        for reason, count in sorted(failures.items()):
            log(f"{reason}: {count:,}")


def save_dataset(df):
    CASE_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(CASE_DATASET) + ".tmp"
    )

    df.to_parquet(
        temporary_path,
        index=False,
        engine="pyarrow",
    )

    temporary_path.replace(CASE_DATASET)

    log("")
    log(f"Research dataset written to: {CASE_DATASET}")


def main():
    log("===========================================")
    log("       RESEARCH DATASET CONSTRUCTION")
    log("===========================================")

    lifecycles = load_lifecycles()

    if lifecycles.empty:
        log("[ERROR] No lifecycle data found")
        return

    lifecycles = prepare_lifecycles(lifecycles)

    if lifecycles.empty:
        log("[ERROR] No usable lifecycle data")
        return

    log(f"Lifecycle episodes: {len(lifecycles):,}")

    data = load_research_raw_data(lifecycles)

    if not data:
        log("[ERROR] No raw data loaded")
        return

    log("")
    log("========== BUILDING CASE DATASET ==========")

    dataset, failures = build_dataset(
        lifecycles,
        data,
    )

    dataset = validate_dataset(
        dataset,
        lifecycles,
        failures,
    )

    print_summary(
        dataset,
        failures,
    )

    save_dataset(dataset)

    log("")
    log("===========================================")
    log("[PASSED] RESEARCH DATASET CONSTRUCTED")
    log("===========================================")


if __name__ == "__main__":
    main()
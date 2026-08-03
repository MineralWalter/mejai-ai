from pathlib import Path
import json

import pandas as pd

from utils import load_research_raw_data,determine_lane

INPUT_FILE = Path("data/analysis/mejai_purchase_lifecycles.json")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_FILE = OUTPUT_DIR / "mejai_research_dataset.parquet"

def log(message):
    print(message)


# ============================================================
# LIFECYCLE LOADING
# ============================================================

def load_lifecycles():
    if not INPUT_FILE.exists():
        log(f"[ERROR] Lifecycle file not found: {INPUT_FILE}")
        return pd.DataFrame()

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as file:
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

    df["participant_id"] = pd.to_numeric(
        df["participant_id"],
        errors="coerce",
    )

    df["purchase_timestamp"] = pd.to_numeric(df["purchase_timestamp"],errors="coerce",)

    df = df.dropna(
        subset=[
            "match_id",
            "participant_id",
            "purchase_timestamp",
            "status",
        ])

    df["participant_id"] = df["participant_id"].astype(int)
    df["purchase_timestamp"] = df["purchase_timestamp"].astype(int)

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

    # --------------------------------------------------------
    # ATTACH TEAM ID TO EACH SNAPSHOT
    # --------------------------------------------------------

    if not participants.empty:

        team_lookup = (participants[["match_id","participant_id","team_id",]].drop_duplicates(subset=["match_id","participant_id",]))

        snapshots = snapshots.merge(team_lookup,on=["match_id","participant_id",],how="left",)

    snapshots = snapshots.sort_values(["participant_id","timestamp",])

    by_participant = {}

    for participant_id, group in snapshots.groupby(
        "participant_id",
        sort=False,
    ):
        by_participant[int(participant_id)] = group

    by_timestamp = {
        int(timestamp): group
        for timestamp, group in snapshots.groupby("timestamp",
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




def build_snapshot_features(snapshot_index,participant_id,player_team_id,purchase_timestamp,):
    player_snapshot = get_latest_player_snapshot(snapshot_index,participant_id,purchase_timestamp,)

    if player_snapshot is None:
        return None

    features = {}

    snapshot_timestamp = int(player_snapshot["timestamp"])

    features["snapshot_timestamp"] = snapshot_timestamp

    features["snapshot_age_ms"] = (purchase_timestamp- snapshot_timestamp)

    # --------------------------------------------------------
    # PLAYER BASE STATE
    # --------------------------------------------------------

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
            features[f"player_{column}"] = (
                player_snapshot[column]
            )

    # --------------------------------------------------------
    # ALL PLAYERS AT SAME SNAPSHOT TIMESTAMP
    # --------------------------------------------------------

    same_timestamp = snapshot_index[
        "by_timestamp"
    ].get(snapshot_timestamp)

    if same_timestamp is None or same_timestamp.empty:
        return features

    if "team_id" not in same_timestamp.columns:
        return features

    # --------------------------------------------------------
    # TEAM / ENEMY SNAPSHOTS
    # --------------------------------------------------------

    team_snapshots = same_timestamp[same_timestamp["team_id"] == player_team_id]

    enemy_snapshots = same_timestamp[same_timestamp["team_id"] != player_team_id]

    # --------------------------------------------------------
    # TEAM / ENEMY AGGREGATES
    # --------------------------------------------------------

    aggregate_columns = [
        "current_gold",
        "total_gold",
        "xp",
        "minions_killed",
        "jungle_minions_killed",
    ]

    for column in aggregate_columns:
        if column in team_snapshots.columns:
            features[f"team_{column}_sum"] = (team_snapshots[column].sum())

        if column in enemy_snapshots.columns:
            features[f"enemy_{column}_sum"] = (enemy_snapshots[column].sum())

    # --------------------------------------------------------
    # RELATIVE FEATURES
    # --------------------------------------------------------

    if ("total_gold" in team_snapshots.columns and "total_gold" in enemy_snapshots.columns):
        features["team_total_gold_diff"] = (
            team_snapshots["total_gold"].sum()
            - enemy_snapshots["total_gold"].sum()
        )

    if ("current_gold" in team_snapshots.columns and "current_gold" in enemy_snapshots.columns):
        features["team_current_gold_diff"] = (
            team_snapshots["current_gold"].sum()
            - enemy_snapshots["current_gold"].sum()
        )

    if ("xp" in team_snapshots.columns and "xp" in enemy_snapshots.columns):
        features["team_xp_diff"] = (
            team_snapshots["xp"].sum()
            - enemy_snapshots["xp"].sum()
        )

    if ("minions_killed" in team_snapshots.columns and "minions_killed" in enemy_snapshots.columns):
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
                outcomes[f"outcome_final_{column}"] = (
                    participant_row[column]
                )

    if match_row is not None:
        if "game_duration" in match_row.index:
            outcomes["outcome_game_duration"] = (
                match_row["game_duration"]
            )

        if "end_of_game_result" in match_row.index:
            outcomes["outcome_game_result"] = (
                match_row["end_of_game_result"]
            )

    return outcomes


# ============================================================
# CASE CONSTRUCTION
# ============================================================
'''
def build_case(lifecycle, tables):
    match_id = str(lifecycle["match_id"])
    participant_id = int(lifecycle["participant_id"])
    purchase_timestamp = int(lifecycle["purchase_timestamp"])

    matches = tables["matches"]
    participants = tables["participants"]
    snapshots = tables["snapshots"]

    match_lookup = build_match_lookup(matches)
    participant_lookup = build_participant_lookup(participants)

    match_row = match_lookup.get(match_id)

    if match_row is None:
        return None, "missing_match"

    participant_row = participant_lookup.get((match_id, participant_id))

    if participant_row is None:
        return None, "missing_participant"

    if "team_id" not in participant_row.index:
        return None, "missing_team_id"

    player_team_id = int(participant_row["team_id"])
    match_snapshots = snapshots[snapshots["match_id"] == match_id]

    snapshot_features = build_snapshot_features(
        match_snapshots,
        participant_id,
        player_team_id,
        purchase_timestamp,
    )

    if snapshot_features is None:
        return None, "missing_pre_purchase_snapshot"

    case = {
        "case_id": (
            f"{match_id}_"
            f"{participant_id}_"
            f"{purchase_timestamp}"),
        "match_id": match_id,
        "participant_id": participant_id,
        "region": determine_lane(match_id),
        "purchase_timestamp": purchase_timestamp,
        "purchase_time_seconds": (purchase_timestamp / 1000),
        "lifecycle_status": lifecycle["status"],
        }

    case.update(build_match_context(match_row))

    case.update(build_player_context(participant_row))

    case.update(snapshot_features)

    # Outcomes are deliberately kept under an
    # explicit outcome_ namespace.
    case.update(build_outcomes(match_row,participant_row,))

    return case, None
'''

# ============================================================
# TEMPORAL VALIDATION
# ============================================================

def validate_temporal_order(case):
    if "snapshot_timestamp" not in case:
        return False

    if "purchase_timestamp" not in case:
        return False

    return (case["snapshot_timestamp"] <= case["purchase_timestamp"])


# ============================================================
# DATASET BUILDING
# ============================================================
def build_dataset(lifecycles, data):
    cases = []
    failures = {}

    for lane, tables in data.items():

        lane_lifecycles = lifecycles[
            lifecycles["match_id"].astype(str).map(
                determine_lane
            ) == lane
        ]

        if lane_lifecycles.empty:
            continue

        matches = tables["matches"]
        participants = tables["participants"]
        snapshots = tables["snapshots"]

        match_lookup = build_match_lookup(matches)
        participant_lookup = build_participant_lookup(
            participants
        )


        snapshot_indexes = {}

        if not snapshots.empty:
            for match_id, match_snapshots in snapshots.groupby("match_id",sort=False,):
                snapshot_indexes[str(match_id)] = (build_snapshot_index(match_snapshots, participants))


        for _, lifecycle in lane_lifecycles.iterrows():

            match_id = str(lifecycle["match_id"])

            participant_id = int(lifecycle["participant_id"])

            purchase_timestamp = int(lifecycle["purchase_timestamp"])

            # ------------------------------------------------
            # MATCH
            # ------------------------------------------------

            match_row = match_lookup.get(match_id)

            if match_row is None:
                failures["missing_match"] = (failures.get("missing_match", 0) + 1)
                continue

            # ------------------------------------------------
            # PARTICIPANT
            # ------------------------------------------------

            participant_row = participant_lookup.get((match_id, participant_id))

            if participant_row is None:
                failures["missing_participant"] = (failures.get("missing_participant",0,) + 1)
                continue

            # ------------------------------------------------
            # TEAM ID
            # ------------------------------------------------

            if "team_id" not in participant_row.index:
                failures["missing_team_id"] = (failures.get("missing_team_id",0,) + 1)
                continue

            player_team_id = int(
                participant_row["team_id"]
            )

            # ------------------------------------------------
            # SNAPSHOT INDEX
            # ------------------------------------------------

            snapshot_index = snapshot_indexes.get(
                match_id
            )

            if snapshot_index is None:
                failures["missing_snapshots"] = (failures.get("missing_snapshots",0,) + 1)
                continue

            # ------------------------------------------------
            # SNAPSHOT FEATURES
            # ------------------------------------------------

            snapshot_features = (
                build_snapshot_features(
                    snapshot_index,
                    participant_id,
                    player_team_id,
                    purchase_timestamp,
                )
            )

            if snapshot_features is None:
                failures["missing_pre_purchase_snapshot"] = (failures.get("missing_pre_purchase_snapshot",0,) + 1)
                continue

            # ------------------------------------------------
            # CASE
            # ------------------------------------------------

            case = {
                "case_id": (
                    f"{match_id}_"
                    f"{participant_id}_"
                    f"{purchase_timestamp}"
                ),
                "match_id": match_id,
                "participant_id": participant_id,
                "region": lane,
                "purchase_timestamp": (purchase_timestamp),
                "purchase_time_seconds": (purchase_timestamp / 1000),
                "lifecycle_status": (lifecycle["status"]),
            }

            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            case.update(build_match_context(match_row))

            case.update(build_player_context(participant_row))

            case.update(snapshot_features)

            # ------------------------------------------------
            # OUTCOMES
            # ------------------------------------------------

            case.update(build_outcomes(match_row,participant_row,))

            # ------------------------------------------------
            # TEMPORAL VALIDATION
            # ------------------------------------------------

            if not validate_temporal_order(case):
                failures["future_snapshot"] = (
                    failures.get(
                        "future_snapshot",
                        0,
                    ) + 1
                )
                continue

            cases.append(case)

            # ------------------------------------------------
            # PROGRESS LOGGING
            # ------------------------------------------------

            if len(cases) % 1000 == 0:
                log(
                    f"Cases processed: "
                    f"{len(cases):,}"
                )

    return pd.DataFrame(cases), failures

def print_summary(df, failures):
    log("========== DATASET SUMMARY ==========")
    log(f"Research cases: {len(df):,}")
    log(f"Columns: {len(df.columns):,}")

    if not df.empty:
        log(f"Regions: {df['region'].nunique():,}")

        log("Lifecycle statuses:")

        for status, count in (df["lifecycle_status"].value_counts().items()):
            log(f"{status}: {count:,}")

    if failures:
        log("========== BUILD EXCLUSIONS ==========")

        for reason, count in sorted(failures.items()):
            log(f"{reason}: {count:,}")


def save_dataset(df):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True,)

    df.to_parquet(OUTPUT_FILE,index=False,)

    log("")
    log(f"Research dataset written to: {OUTPUT_FILE}")

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

    dataset, failures = build_dataset(lifecycles,data,)

    if dataset.empty:
        log("[ERROR] No research cases could be constructed")
        return

    dataset = dataset.drop_duplicates(subset=["case_id"]).reset_index(drop=True)

    print_summary(dataset,failures,)

    save_dataset(dataset)

    log("")
    log("===========================================")
    log("[PASSED] RESEARCH DATASET CONSTRUCTED")
    log("===========================================")

if __name__ == "__main__":
    main()
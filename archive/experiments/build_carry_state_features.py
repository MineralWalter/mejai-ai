from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PRIMARY_INPUT = Path("data/analysis/mejai_matched_primary_features.parquet")
SENSITIVITY_INPUT = Path("data/analysis/mejai_matched_sensitivity_features.parquet")

AUDIT_INVENTORY_FILE = Path(
    "data/analysis/purchase_feature_audit/parquet_file_inventory.csv"
)
PARQUET_ROOT = Path("data/parquet")

OUTPUT_DIR = Path("data/analysis")
PRIMARY_OUTPUT = OUTPUT_DIR / "mejai_matched_primary_carry_features.parquet"
SENSITIVITY_OUTPUT = OUTPUT_DIR / "mejai_matched_sensitivity_carry_features.parquet"
DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "carry_state_feature_diagnostics.csv"

LOG_EVERY_FILES = 25

MATCH_ALIASES = ["match_id", "matchId"]
PARTICIPANT_ALIASES = ["participant_id", "participantId"]
TIMESTAMP_ALIASES = [
    "timestamp",
    "snapshot_timestamp",
    "frame_timestamp",
    "observation_timestamp",
]
TEAM_ALIASES = ["team_id", "teamId"]
POSITION_ALIASES = [
    "team_position",
    "teamPosition",
    "position",
    "individual_position",
    "individualPosition",
]
GOLD_ALIASES = [
    "player_total_gold",
    "total_gold",
    "totalGold",
    "participant_total_gold",
    "participantTotalGold",
]
XP_ALIASES = [
    "player_xp",
    "xp",
    "experience",
    "participant_xp",
    "participantXp",
]

TEAM_GOLD_DIFF_ALIASES = [
    "team_total_gold_diff",
    "team_gold_diff",
    "teamGoldDiff",
]
TEAM_XP_DIFF_ALIASES = [
    "team_xp_diff",
    "teamXpDiff",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def log(message: str) -> None:
    print(message)


def first_existing(columns, candidates):
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def normalize_position(value) -> str:
    position = str(value).strip().upper()

    aliases = {
        "MID": "MIDDLE",
        "MIDDLE": "MIDDLE",
        "TOP": "TOP",
        "JGL": "JUNGLE",
        "JUNGLE": "JUNGLE",
        "ADC": "BOTTOM",
        "BOT": "BOTTOM",
        "BOTTOM": "BOTTOM",
        "SUP": "UTILITY",
        "SUPPORT": "UTILITY",
        "UTILITY": "UTILITY",
    }
    return aliases.get(position, position)


def opposing_team_id(team_id: int) -> int | None:
    return {100: 200, 200: 100}.get(int(team_id))


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return [field.name for field in pq.ParquetFile(path).schema_arrow]
    except Exception:
        return list(pd.read_parquet(path).head(0).columns)


def discover_files(table_type: str, fallback_keywords: tuple[str, ...]) -> list[Path]:
    if AUDIT_INVENTORY_FILE.exists():
        inventory = pd.read_csv(AUDIT_INVENTORY_FILE)

        if {"path", "table_type"}.issubset(inventory.columns):
            paths = (
                inventory.loc[inventory["table_type"] == table_type, "path"]
                .dropna()
                .astype(str)
                .map(Path)
                .tolist()
            )
            existing = sorted(path for path in paths if path.exists())
            if existing:
                return existing

    if not PARQUET_ROOT.exists():
        raise FileNotFoundError(f"Parquet root not found: {PARQUET_ROOT}")

    paths = sorted(
        path
        for path in PARQUET_ROOT.rglob("*.parquet")
        if any(keyword in str(path).lower() for keyword in fallback_keywords)
    )
    if not paths:
        raise FileNotFoundError(f"No {table_type} parquet files found")

    return paths


# ============================================================
# MATCHED OBSERVATIONS
# ============================================================

def load_matched_dataset(path: Path, sample_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{sample_name} input not found: {path}")

    df = pd.read_parquet(path)
    required = {
        "match_id",
        "participant_id",
        "observation_timestamp",
        "matched_set_id",
        "treatment",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{sample_name} is missing required columns: {missing}")

    df = df.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")
    df["observation_timestamp"] = pd.to_numeric(
        df["observation_timestamp"], errors="coerce"
    )
    df = df.dropna(
        subset=["match_id", "participant_id", "observation_timestamp"]
    )
    df["participant_id"] = df["participant_id"].astype(int)
    df["observation_timestamp"] = df["observation_timestamp"].astype(int)

    return df.reset_index(drop=True)


def build_observation_table(sensitivity: pd.DataFrame) -> pd.DataFrame:
    columns = ["match_id", "participant_id", "observation_timestamp"]

    for candidates in [TEAM_ALIASES, POSITION_ALIASES]:
        column = first_existing(sensitivity.columns, candidates)
        if column is not None and column not in columns:
            columns.append(column)

    observations = sensitivity[columns].drop_duplicates(
        subset=["match_id", "participant_id", "observation_timestamp"]
    )
    observations = observations.reset_index(drop=True)

    team_column = first_existing(observations.columns, TEAM_ALIASES)
    position_column = first_existing(observations.columns, POSITION_ALIASES)

    if team_column is not None and team_column != "team_id":
        observations = observations.rename(columns={team_column: "team_id"})

    if position_column is not None and position_column != "team_position":
        observations = observations.rename(
            columns={position_column: "team_position"}
        )

    observations["observation_id"] = (
        observations["match_id"].astype(str)
        + "_"
        + observations["participant_id"].astype(str)
        + "_"
        + observations["observation_timestamp"].astype(str)
    )

    return observations


# ============================================================
# PARTICIPANT METADATA
# ============================================================

def load_participant_metadata(
    participant_files: list[Path], relevant_match_ids: set[str]
) -> pd.DataFrame:
    frames = []
    relevant_match_ids = set(map(str, relevant_match_ids))

    for index, path in enumerate(participant_files, start=1):
        try:
            available = parquet_columns(path)
            match_col = first_existing(available, MATCH_ALIASES)
            participant_col = first_existing(available, PARTICIPANT_ALIASES)
            team_col = first_existing(available, TEAM_ALIASES)
            position_col = first_existing(available, POSITION_ALIASES)

            if None in {match_col, participant_col, team_col, position_col}:
                continue

            frame = pd.read_parquet(
                path,
                columns=[match_col, participant_col, team_col, position_col],
            )
            frame = frame.rename(
                columns={
                    match_col: "match_id",
                    participant_col: "participant_id",
                    team_col: "team_id",
                    position_col: "team_position",
                }
            )
            frame["match_id"] = frame["match_id"].astype(str)
            frame = frame[frame["match_id"].isin(relevant_match_ids)].copy()
            if frame.empty:
                continue

            frame["participant_id"] = pd.to_numeric(
                frame["participant_id"], errors="coerce"
            )
            frame["team_id"] = pd.to_numeric(frame["team_id"], errors="coerce")
            frame = frame.dropna(
                subset=["participant_id", "team_id", "team_position"]
            )
            frame["participant_id"] = frame["participant_id"].astype(int)
            frame["team_id"] = frame["team_id"].astype(int)
            frame["team_position"] = frame["team_position"].map(
                normalize_position
            )

            frames.append(
                frame[
                    ["match_id", "participant_id", "team_id", "team_position"]
                ]
            )

        except Exception as error:
            warnings.warn(f"Could not process participant file {path}: {error}")

        if index % LOG_EVERY_FILES == 0 or index == len(participant_files):
            log(
                f"Participant files processed: {index:,} / "
                f"{len(participant_files):,}"
            )

    if not frames:
        raise ValueError("No relevant participant metadata was loaded")

    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata.drop_duplicates(
        subset=["match_id", "participant_id"], keep="last"
    )

    duplicate_roles = metadata.duplicated(
        subset=["match_id", "team_id", "team_position"], keep=False
    )
    if duplicate_roles.any():
        count = int(duplicate_roles.sum())
        warnings.warn(
            f"{count:,} participant rows have duplicate match/team/position "
            "assignments. The first valid role opponent will be used."
        )

    return metadata.reset_index(drop=True)


# ============================================================
# SNAPSHOT LOADING
# ============================================================

def load_relevant_snapshots(
    snapshot_files: list[Path], relevant_match_ids: set[str]
) -> pd.DataFrame:
    frames = []
    retained_rows = 0
    relevant_match_ids = set(map(str, relevant_match_ids))

    for index, path in enumerate(snapshot_files, start=1):
        try:
            available = parquet_columns(path)
            match_col = first_existing(available, MATCH_ALIASES)
            participant_col = first_existing(available, PARTICIPANT_ALIASES)
            timestamp_col = first_existing(available, TIMESTAMP_ALIASES)
            gold_col = first_existing(available, GOLD_ALIASES)
            xp_col = first_existing(available, XP_ALIASES)

            if None in {
                match_col,
                participant_col,
                timestamp_col,
                gold_col,
                xp_col,
            }:
                continue

            frame = pd.read_parquet(
                path,
                columns=[
                    match_col,
                    participant_col,
                    timestamp_col,
                    gold_col,
                    xp_col,
                ],
            )
            frame = frame.rename(
                columns={
                    match_col: "match_id",
                    participant_col: "participant_id",
                    timestamp_col: "snapshot_timestamp",
                    gold_col: "snapshot_total_gold",
                    xp_col: "snapshot_xp",
                }
            )
            frame["match_id"] = frame["match_id"].astype(str)
            frame = frame[frame["match_id"].isin(relevant_match_ids)].copy()
            if frame.empty:
                continue

            for column in [
                "participant_id",
                "snapshot_timestamp",
                "snapshot_total_gold",
                "snapshot_xp",
            ]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

            frame = frame.dropna(
                subset=[
                    "participant_id",
                    "snapshot_timestamp",
                    "snapshot_total_gold",
                    "snapshot_xp",
                ]
            )
            frame["participant_id"] = frame["participant_id"].astype(int)
            frame["snapshot_timestamp"] = frame["snapshot_timestamp"].astype(int)

            retained_rows += len(frame)
            frames.append(frame)

        except Exception as error:
            warnings.warn(f"Could not process snapshot file {path}: {error}")

        if index % LOG_EVERY_FILES == 0 or index == len(snapshot_files):
            log(
                f"Snapshot files processed: {index:,} / "
                f"{len(snapshot_files):,} | retained rows: {retained_rows:,}"
            )

    if not frames:
        raise ValueError("No relevant snapshot rows were loaded")

    snapshots = pd.concat(frames, ignore_index=True)
    snapshots = snapshots.drop_duplicates(
        subset=["match_id", "participant_id", "snapshot_timestamp"],
        keep="last",
    )
    snapshots = snapshots.sort_values(
        ["match_id", "participant_id", "snapshot_timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    log(f"Relevant snapshots loaded: {len(snapshots):,}")
    return snapshots


# ============================================================
# STATE LOOKUP
# ============================================================

def attach_metadata_to_observations(
    observations: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    enriched = observations.copy()

    metadata_columns = [
        "match_id",
        "participant_id",
        "team_id",
        "team_position",
    ]
    joined = enriched.merge(
        metadata[metadata_columns],
        on=["match_id", "participant_id"],
        how="left",
        suffixes=("", "_metadata"),
        validate="many_to_one",
    )

    if "team_id_metadata" in joined.columns:
        if "team_id" not in joined.columns:
            joined["team_id"] = joined["team_id_metadata"]
        else:
            joined["team_id"] = joined["team_id"].fillna(
                joined["team_id_metadata"]
            )
        joined = joined.drop(columns=["team_id_metadata"])

    if "team_position_metadata" in joined.columns:
        if "team_position" not in joined.columns:
            joined["team_position"] = joined["team_position_metadata"]
        else:
            joined["team_position"] = joined["team_position"].fillna(
                joined["team_position_metadata"]
            )
        joined = joined.drop(columns=["team_position_metadata"])

    joined["team_id"] = pd.to_numeric(joined["team_id"], errors="coerce")
    joined["team_position"] = joined["team_position"].map(normalize_position)

    missing = joined[["team_id", "team_position"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum()):,} observations are missing team or position metadata"
        )

    joined["team_id"] = joined["team_id"].astype(int)
    return joined


def build_role_opponent_lookup(metadata: pd.DataFrame) -> pd.DataFrame:
    opponents = metadata.rename(
        columns={
            "participant_id": "opponent_participant_id",
            "team_id": "opponent_team_id",
        }
    )

    lookup = metadata.merge(
        opponents,
        on=["match_id", "team_position"],
        how="left",
        suffixes=("", "_candidate"),
    )
    lookup = lookup[
        lookup["team_id"] != lookup["opponent_team_id"]
    ].copy()
    lookup = lookup.sort_values(
        ["match_id", "participant_id", "opponent_participant_id"]
    )
    lookup = lookup.drop_duplicates(
        subset=["match_id", "participant_id"],
        keep="first",
    )

    return lookup[
        ["match_id", "participant_id", "opponent_participant_id"]
    ].reset_index(drop=True)


def nearest_snapshot_before_observation(
    observations: pd.DataFrame,
    snapshots: pd.DataFrame,
    participant_column: str,
    prefix: str,
) -> pd.DataFrame:
    left = observations[
        ["observation_id", "match_id", participant_column, "observation_timestamp"]
    ].copy()
    left = left.rename(columns={participant_column: "lookup_participant_id"})

    left["match_id"] = left["match_id"].astype(str)
    left["lookup_participant_id"] = pd.to_numeric(
        left["lookup_participant_id"], errors="coerce"
    )
    left["observation_timestamp"] = pd.to_numeric(
        left["observation_timestamp"], errors="coerce"
    )
    left = left.dropna(
        subset=["lookup_participant_id", "observation_timestamp"]
    )
    left["lookup_participant_id"] = left["lookup_participant_id"].astype(int)
    left["observation_timestamp"] = left["observation_timestamp"].astype(int)

    right = snapshots.rename(
        columns={"participant_id": "lookup_participant_id"}
    ).copy()
    right["match_id"] = right["match_id"].astype(str)
    right["lookup_participant_id"] = pd.to_numeric(
        right["lookup_participant_id"], errors="coerce"
    )
    right["snapshot_timestamp"] = pd.to_numeric(
        right["snapshot_timestamp"], errors="coerce"
    )
    right = right.dropna(
        subset=["lookup_participant_id", "snapshot_timestamp"]
    )
    right["lookup_participant_id"] = right["lookup_participant_id"].astype(int)
    right["snapshot_timestamp"] = right["snapshot_timestamp"].astype(int)

    # merge_asof requires the merge key itself to be globally sorted.
    # Sorting by match/participant first can still trigger:
    # ValueError: left keys must be sorted.
    left = left.sort_values(
        ["observation_timestamp", "match_id", "lookup_participant_id"],
        kind="stable",
    ).reset_index(drop=True)
    right = right.sort_values(
        ["snapshot_timestamp", "match_id", "lookup_participant_id"],
        kind="stable",
    ).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="observation_timestamp",
        right_on="snapshot_timestamp",
        by=["match_id", "lookup_participant_id"],
        direction="backward",
        allow_exact_matches=True,
    )

    return merged[
        [
            "observation_id",
            "snapshot_timestamp",
            "snapshot_total_gold",
            "snapshot_xp",
        ]
    ].rename(
        columns={
            "snapshot_timestamp": f"{prefix}_snapshot_timestamp",
            "snapshot_total_gold": f"{prefix}_total_gold",
            "snapshot_xp": f"{prefix}_xp",
        }
    )


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def resolve_required_column(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    column = first_existing(df.columns, aliases)
    if column is None:
        raise ValueError(
            f"Could not find {label}. Tried columns: {aliases}"
        )
    return column


def derive_carry_state_features(
    observations: pd.DataFrame,
    snapshots: pd.DataFrame,
    metadata: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    observations = attach_metadata_to_observations(observations, metadata)

    opponent_lookup = build_role_opponent_lookup(metadata)
    observations = observations.merge(
        opponent_lookup,
        on=["match_id", "participant_id"],
        how="left",
        validate="many_to_one",
    )

    missing_opponents = observations["opponent_participant_id"].isna()
    if missing_opponents.any():
        warnings.warn(
            f"{int(missing_opponents.sum()):,} observations have no role opponent"
        )

    player_state = nearest_snapshot_before_observation(
        observations,
        snapshots,
        participant_column="participant_id",
        prefix="player",
    )
    opponent_state = nearest_snapshot_before_observation(
        observations.dropna(subset=["opponent_participant_id"]).assign(
            opponent_participant_id=lambda frame: frame[
                "opponent_participant_id"
            ].astype(int)
        ),
        snapshots,
        participant_column="opponent_participant_id",
        prefix="opponent",
    )

    features = observations[
        [
            "observation_id",
            "match_id",
            "participant_id",
            "observation_timestamp",
            "team_id",
            "team_position",
            "opponent_participant_id",
        ]
    ].copy()
    features = features.merge(
        player_state,
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        opponent_state,
        on="observation_id",
        how="left",
        validate="one_to_one",
    )

    state_columns = [
        "match_id",
        "participant_id",
        "observation_timestamp",
    ]
    team_gold_diff_column = resolve_required_column(
        sensitivity,
        TEAM_GOLD_DIFF_ALIASES,
        "team gold difference",
    )
    team_xp_diff_column = resolve_required_column(
        sensitivity,
        TEAM_XP_DIFF_ALIASES,
        "team XP difference",
    )

    matched_state = sensitivity[
        state_columns + [team_gold_diff_column, team_xp_diff_column]
    ].drop_duplicates(subset=state_columns)

    matched_state = matched_state.rename(
        columns={
            team_gold_diff_column: "team_gold_diff_for_carry_state",
            team_xp_diff_column: "team_xp_diff_for_carry_state",
        }
    )
    matched_state["match_id"] = matched_state["match_id"].astype(str)

    features = features.merge(
        matched_state,
        on=state_columns,
        how="left",
        validate="one_to_one",
    )

    features["player_gold_diff_vs_role_opponent"] = (
        features["player_total_gold"] - features["opponent_total_gold"]
    )
    features["player_xp_diff_vs_role_opponent"] = (
        features["player_xp"] - features["opponent_xp"]
    )
    features["rest_of_team_gold_diff"] = (
        features["team_gold_diff_for_carry_state"]
        - features["player_gold_diff_vs_role_opponent"]
    )
    features["rest_of_team_xp_diff"] = (
        features["team_xp_diff_for_carry_state"]
        - features["player_xp_diff_vs_role_opponent"]
    )

    feature_columns = [
        "observation_id",
        "team_id",
        "team_position",
        "opponent_participant_id",
        "player_snapshot_timestamp",
        "opponent_snapshot_timestamp",
        "player_total_gold",
        "opponent_total_gold",
        "player_xp",
        "opponent_xp",
        "player_gold_diff_vs_role_opponent",
        "player_xp_diff_vs_role_opponent",
        "rest_of_team_gold_diff",
        "rest_of_team_xp_diff",
    ]

    return features[feature_columns]


# ============================================================
# MERGE AND DIAGNOSTICS
# ============================================================

def attach_features(matched: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    enriched = matched.copy()
    enriched["observation_id"] = (
        enriched["match_id"].astype(str)
        + "_"
        + enriched["participant_id"].astype(int).astype(str)
        + "_"
        + enriched["observation_timestamp"].astype(int).astype(str)
    )

    original_rows = len(enriched)
    enriched = enriched.merge(
        features,
        on="observation_id",
        how="left",
        validate="many_to_one",
    )

    if len(enriched) != original_rows:
        raise ValueError("Carry-state feature merge changed row count")

    return enriched


def validate_features(enriched: pd.DataFrame, sample_name: str) -> None:
    required = [
        "player_gold_diff_vs_role_opponent",
        "player_xp_diff_vs_role_opponent",
        "rest_of_team_gold_diff",
        "rest_of_team_xp_diff",
    ]
    missing = [column for column in required if column not in enriched.columns]
    if missing:
        raise ValueError(f"{sample_name} is missing carry-state features: {missing}")

    coverage = enriched[required].notna().mean()
    low_coverage = coverage[coverage < 0.95]
    if not low_coverage.empty:
        warnings.warn(
            f"{sample_name} has carry-state feature coverage below 95%:\n"
            f"{low_coverage.to_string()}"
        )


def build_diagnostics(sensitivity: pd.DataFrame) -> pd.DataFrame:
    features = [
        "player_gold_diff_vs_role_opponent",
        "player_xp_diff_vs_role_opponent",
        "rest_of_team_gold_diff",
        "rest_of_team_xp_diff",
    ]

    rows = []
    for feature in features:
        numeric = pd.to_numeric(sensitivity[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "non_missing_count": int(numeric.notna().sum()),
                "non_missing_ratio": float(numeric.notna().mean()),
                "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "std": float(numeric.std()) if numeric.notna().any() else np.nan,
                "minimum": float(numeric.min()) if numeric.notna().any() else np.nan,
                "maximum": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def print_summary(
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    log("")
    log("=" * 80)
    log("CARRY-STATE FEATURE SUMMARY")
    log("=" * 80)
    log(f"Primary rows: {len(primary):,}")
    log(f"Sensitivity rows: {len(sensitivity):,}")
    log("")
    log("Feature coverage:")

    display = diagnostics[
        ["feature", "non_missing_ratio", "mean", "minimum", "maximum"]
    ].copy()
    display["non_missing_ratio"] = display["non_missing_ratio"].map(
        lambda value: f"{value:.2%}"
    )
    log(display.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 80)
    log("BUILD CARRY-STATE FEATURES")
    log("=" * 80)

    primary = load_matched_dataset(PRIMARY_INPUT, "primary")
    sensitivity = load_matched_dataset(SENSITIVITY_INPUT, "sensitivity")

    log(f"Primary rows loaded: {len(primary):,}")
    log(f"Sensitivity rows loaded: {len(sensitivity):,}")

    observations = build_observation_table(sensitivity)
    log(f"Unique observations to feature: {len(observations):,}")

    relevant_match_ids = set(observations["match_id"])
    participant_files = discover_files(
        "participants",
        ("participant",),
    )
    snapshot_files = discover_files(
        "snapshots",
        ("snapshot", "frame"),
    )

    log(f"Participant parquet files found: {len(participant_files):,}")
    log(f"Snapshot parquet files found: {len(snapshot_files):,}")

    metadata = load_participant_metadata(
        participant_files,
        relevant_match_ids,
    )
    log(f"Participant metadata rows loaded: {len(metadata):,}")

    snapshots = load_relevant_snapshots(
        snapshot_files,
        relevant_match_ids,
    )

    features = derive_carry_state_features(
        observations,
        snapshots,
        metadata,
        sensitivity,
    )

    primary_enriched = attach_features(primary, features)
    sensitivity_enriched = attach_features(sensitivity, features)

    validate_features(primary_enriched, "primary")
    validate_features(sensitivity_enriched, "sensitivity")

    diagnostics = build_diagnostics(sensitivity_enriched)
    print_summary(primary_enriched, sensitivity_enriched, diagnostics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_enriched.to_parquet(PRIMARY_OUTPUT, index=False)
    sensitivity_enriched.to_parquet(SENSITIVITY_OUTPUT, index=False)
    diagnostics.to_csv(DIAGNOSTICS_OUTPUT, index=False)

    log("")
    log(f"[SAVED] {PRIMARY_OUTPUT}")
    log(f"[SAVED] {SENSITIVITY_OUTPUT}")
    log(f"[SAVED] {DIAGNOSTICS_OUTPUT}")
    log("")
    log("[PASSED] CARRY-STATE FEATURES CONSTRUCTED")


if __name__ == "__main__":
    main()
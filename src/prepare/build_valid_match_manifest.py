from pathlib import Path

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

REQUIRED_MATCH_COLUMNS = [
    "match_id",
    "queue_id",
    "end_of_game_result",
]

REQUIRED_PARTICIPANT_COLUMNS = [
    "match_id",
    "participant_id",
    "team_id",
    "win",
]


# ============================================================
# LOGGING
# ============================================================

def log(message: str = "") -> None:
    print(message)


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_files(table: str, lane: str) -> list[Path]:
    directory = PARQUET_DIR / table

    if not directory.exists():
        return []

    return sorted(
        directory.glob(
            f"{lane}_part_*.parquet"
        )
    )


def get_parquet_columns(path: Path) -> set[str]:
    try:
        schema = pq.ParquetFile(path).schema_arrow
        return {
            field.name
            for field in schema
        }
    except Exception as error:
        raise RuntimeError(
            f"Could not inspect Parquet schema for {path}: {error}"
        ) from error


# ============================================================
# TABLE LOADING
# ============================================================

def load_lane_table(
    table: str,
    lane: str,
    required_columns: list[str],
) -> pd.DataFrame:
    files = find_files(table, lane)

    if not files:
        log(
            f"[WARNING] No files found for "
            f"{lane}/{table}"
        )
        return pd.DataFrame()

    frames = []

    for file_number, path in enumerate(
        files,
        start=1,
    ):
        available_columns = get_parquet_columns(path)

        missing_columns = [
            column
            for column in required_columns
            if column not in available_columns
        ]

        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{missing_columns}"
            )

        try:
            frame = pd.read_parquet(
                path,
                columns=required_columns,
                engine="pyarrow",
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not read {path}: {error}"
            ) from error

        if not frame.empty:
            frames.append(frame)

        if (
            file_number % 100 == 0
            or file_number == len(files)
        ):
            log(
                f"  {table}: "
                f"{file_number:,}/{len(files):,} files"
            )

    if not frames:
        return pd.DataFrame(
            columns=required_columns
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def collect_match_ids(
    table: str,
    lane: str,
) -> set[str]:
    """
    Collect distinct match IDs without concatenating the full snapshot
    or event tables into memory.
    """

    files = find_files(table, lane)
    match_ids: set[str] = set()

    if not files:
        log(
            f"[WARNING] No files found for "
            f"{lane}/{table}"
        )
        return match_ids

    for file_number, path in enumerate(
        files,
        start=1,
    ):
        available_columns = get_parquet_columns(path)

        if "match_id" not in available_columns:
            raise ValueError(
                f"{path} does not contain match_id"
            )

        try:
            frame = pd.read_parquet(
                path,
                columns=["match_id"],
                engine="pyarrow",
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not read {path}: {error}"
            ) from error

        values = (
            frame["match_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        match_ids.update(values)

        if (
            file_number % 100 == 0
            or file_number == len(files)
        ):
            log(
                f"  {table}: "
                f"{file_number:,}/{len(files):,} files | "
                f"{len(match_ids):,} matches found"
            )

    return match_ids


# ============================================================
# VALUE NORMALISATION
# ============================================================

def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        return numeric.map(
            {
                1: True,
                0: False,
                1.0: True,
                0.0: False,
            }
        ).astype("boolean")

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "win": True,
            "loss": False,
            "won": True,
            "lost": False,
        }
    ).astype("boolean")


# ============================================================
# LANE MANIFEST
# ============================================================

def prepare_matches(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    if matches.empty:
        return matches

    matches = matches.copy()

    matches["match_id"] = (
        matches["match_id"]
        .astype(str)
    )

    matches["queue_id"] = pd.to_numeric(
        matches["queue_id"],
        errors="coerce",
    )

    matches = (
        matches.sort_values("match_id")
        .drop_duplicates(
            subset=["match_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return matches


def prepare_participants(
    participants: pd.DataFrame,
) -> pd.DataFrame:
    if participants.empty:
        return participants

    participants = participants.copy()

    participants["match_id"] = (
        participants["match_id"]
        .astype(str)
    )

    participants["participant_id"] = pd.to_numeric(
        participants["participant_id"],
        errors="coerce",
    )

    participants["team_id"] = pd.to_numeric(
        participants["team_id"],
        errors="coerce",
    )

    participants["win_normalized"] = normalize_boolean(
        participants["win"]
    )

    participants = participants.dropna(
        subset=[
            "match_id",
            "participant_id",
        ]
    )

    participants["participant_id"] = (
        participants["participant_id"]
        .astype(int)
    )

    participants = (
        participants.sort_values(
            [
                "match_id",
                "participant_id",
            ]
        )
        .drop_duplicates(
            subset=[
                "match_id",
                "participant_id",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return participants


def build_exclusion_reason(
    row: pd.Series,
) -> str | None:
    reasons = []

    if row["queue_id"] != 420:
        reasons.append("wrong_queue")

    if row["end_of_game_result"] != "GameComplete":
        reasons.append("incomplete_game")

    if row["participant_count"] != 10:
        reasons.append("invalid_participant_count")

    if row["winner_count"] != 5:
        reasons.append("invalid_winner_count")

    if row["team_count"] != 2:
        reasons.append("invalid_team_count")

    if not row["participant_ids_valid"]:
        reasons.append("invalid_participant_ids")

    if not row["has_snapshots"]:
        reasons.append("missing_snapshots")

    if not row["has_events"]:
        reasons.append("missing_events")

    if not reasons:
        return None

    return "|".join(reasons)


def build_lane_manifest(
    lane: str,
) -> pd.DataFrame:
    log("")
    log("=" * 70)
    log(f"BUILDING MANIFEST: {lane.upper()}")
    log("=" * 70)

    log("Loading match rows...")

    matches = load_lane_table(
        table="matches",
        lane=lane,
        required_columns=REQUIRED_MATCH_COLUMNS,
    )

    log("Loading participant rows...")

    participants = load_lane_table(
        table="participants",
        lane=lane,
        required_columns=REQUIRED_PARTICIPANT_COLUMNS,
    )

    if matches.empty:
        log(
            f"[WARNING] No match rows found for {lane}"
        )
        return pd.DataFrame()

    if participants.empty:
        raise ValueError(
            f"No participant rows found for {lane}"
        )

    matches = prepare_matches(matches)
    participants = prepare_participants(
        participants
    )

    log(
        f"Unique matches loaded: "
        f"{len(matches):,}"
    )

    log(
        f"Unique participant rows loaded: "
        f"{len(participants):,}"
    )

    # --------------------------------------------------------
    # PARTICIPANT STRUCTURE
    # --------------------------------------------------------

    grouped = participants.groupby(
        "match_id",
        sort=False,
        observed=True,
    )

    participant_count = grouped[
        "participant_id"
    ].nunique()

    winner_count = grouped[
        "win_normalized"
    ].sum(
        min_count=1
    )

    team_count = grouped[
        "team_id"
    ].nunique()

    participant_ids_valid = grouped[
        "participant_id"
    ].apply(
        lambda values: (
            set(values.astype(int))
            == set(range(1, 11))
        )
    )

    # --------------------------------------------------------
    # TIMELINE COVERAGE
    # --------------------------------------------------------

    log("Collecting matches with snapshots...")

    snapshot_match_ids = collect_match_ids(
        table="snapshots",
        lane=lane,
    )

    log("Collecting matches with events...")

    event_match_ids = collect_match_ids(
        table="events",
        lane=lane,
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = matches[
        [
            "match_id",
            "queue_id",
            "end_of_game_result",
        ]
    ].copy()

    manifest["storage_partition"] = lane

    manifest["participant_count"] = (
        manifest["match_id"]
        .map(participant_count)
        .fillna(0)
        .astype(int)
    )

    manifest["winner_count"] = (
        manifest["match_id"]
        .map(winner_count)
        .fillna(0)
        .astype(int)
    )

    manifest["team_count"] = (
        manifest["match_id"]
        .map(team_count)
        .fillna(0)
        .astype(int)
    )

    manifest["participant_ids_valid"] = (
        manifest["match_id"]
        .map(participant_ids_valid)
        .fillna(False)
        .astype(bool)
    )

    manifest["has_snapshots"] = (
        manifest["match_id"]
        .isin(snapshot_match_ids)
    )

    manifest["has_events"] = (
        manifest["match_id"]
        .isin(event_match_ids)
    )

    manifest["analysis_eligible"] = (
        manifest["queue_id"].eq(420)
        & manifest["end_of_game_result"].eq(
            "GameComplete"
        )
        & manifest["participant_count"].eq(10)
        & manifest["winner_count"].eq(5)
        & manifest["team_count"].eq(2)
        & manifest["participant_ids_valid"]
        & manifest["has_snapshots"]
        & manifest["has_events"]
    )

    manifest["exclusion_reason"] = (
        manifest.apply(
            build_exclusion_reason,
            axis=1,
        )
    )

    log("")
    log(
        f"{lane} matches: "
        f"{len(manifest):,}"
    )

    log(
        f"{lane} eligible: "
        f"{int(manifest['analysis_eligible'].sum()):,}"
    )

    log(
        f"{lane} excluded: "
        f"{int((~manifest['analysis_eligible']).sum()):,}"
    )

    return manifest


# ============================================================
# VALIDATION
# ============================================================

def validate_manifest(
    manifest: pd.DataFrame,
) -> None:
    required_columns = {
        "match_id",
        "storage_partition",
        "queue_id",
        "end_of_game_result",
        "participant_count",
        "winner_count",
        "team_count",
        "participant_ids_valid",
        "has_snapshots",
        "has_events",
        "analysis_eligible",
        "exclusion_reason",
    }

    missing_columns = sorted(
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: "
            f"{missing_columns}"
        )

    duplicate_match_ids = manifest[
        "match_id"
    ].duplicated(
        keep=False
    )

    if duplicate_match_ids.any():
        examples = (
            manifest.loc[
                duplicate_match_ids,
                [
                    "match_id",
                    "storage_partition",
                ],
            ]
            .sort_values("match_id")
            .head(30)
        )

        raise ValueError(
            "Match IDs appear more than once in the "
            "manifest:\n"
            f"{examples.to_string(index=False)}"
        )

    eligible = manifest[
        manifest["analysis_eligible"]
    ]

    checks = {
        "wrong queue": ~eligible[
            "queue_id"
        ].eq(420),

        "incomplete game": ~eligible[
            "end_of_game_result"
        ].eq("GameComplete"),

        "invalid participant count": ~eligible[
            "participant_count"
        ].eq(10),

        "invalid winner count": ~eligible[
            "winner_count"
        ].eq(5),

        "invalid team count": ~eligible[
            "team_count"
        ].eq(2),

        "invalid participant IDs": ~eligible[
            "participant_ids_valid"
        ],

        "missing snapshots": ~eligible[
            "has_snapshots"
        ],

        "missing events": ~eligible[
            "has_events"
        ],
    }

    failed = {
        name: int(mask.sum())
        for name, mask in checks.items()
        if mask.any()
    }

    if failed:
        raise ValueError(
            f"Eligible manifest rows failed rules: "
            f"{failed}"
        )


# ============================================================
# REPORTING
# ============================================================

def print_summary(
    manifest: pd.DataFrame,
) -> None:
    log("")
    log("=" * 70)
    log("VALID-MATCH MANIFEST SUMMARY")
    log("=" * 70)

    total = len(manifest)

    eligible_count = int(
        manifest["analysis_eligible"].sum()
    )

    excluded_count = (
        total - eligible_count
    )

    log(f"Total matches:    {total:,}")
    log(f"Eligible matches: {eligible_count:,}")
    log(f"Excluded matches: {excluded_count:,}")

    if total:
        log(
            f"Eligibility rate: "
            f"{eligible_count / total:.4%}"
        )

    log("")
    log("By storage partition:")

    partition_summary = (
        manifest.groupby(
            "storage_partition",
            dropna=False,
        )
        .agg(
            total_matches=(
                "match_id",
                "size",
            ),
            eligible_matches=(
                "analysis_eligible",
                "sum",
            ),
        )
    )

    partition_summary[
        "excluded_matches"
    ] = (
        partition_summary["total_matches"]
        - partition_summary["eligible_matches"]
    )

    partition_summary[
        "eligibility_rate"
    ] = (
        partition_summary["eligible_matches"]
        / partition_summary["total_matches"]
    )

    log(
        partition_summary.to_string()
    )

    log("")
    log("Exclusion reasons:")

    exclusions = (
        manifest.loc[
            ~manifest["analysis_eligible"],
            "exclusion_reason",
        ]
        .fillna("unknown")
        .value_counts()
    )

    if exclusions.empty:
        log("No matches were excluded.")
    else:
        log(
            exclusions.to_string()
        )

    if excluded_count:
        log("")
        log("Excluded match examples:")

        display_columns = [
            "match_id",
            "storage_partition",
            "queue_id",
            "end_of_game_result",
            "participant_count",
            "winner_count",
            "team_count",
            "participant_ids_valid",
            "has_snapshots",
            "has_events",
            "exclusion_reason",
        ]

        log(
            manifest.loc[
                ~manifest["analysis_eligible"],
                display_columns,
            ]
            .head(30)
            .to_string(index=False)
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 70)
    log("BUILDING VALID-MATCH MANIFEST")
    log("=" * 70)
    log(f"Parquet directory: {PARQUET_DIR}")
    log(f"Output file:       {VALID_MATCH_MANIFEST}")

    manifests = []

    for lane in LANES:
        lane_manifest = build_lane_manifest(
            lane
        )

        if not lane_manifest.empty:
            manifests.append(
                lane_manifest
            )

    if not manifests:
        raise RuntimeError(
            "No manifest rows were constructed"
        )

    manifest = pd.concat(
        manifests,
        ignore_index=True,
    )

    manifest = manifest.sort_values(
        [
            "storage_partition",
            "match_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    validate_manifest(manifest)
    print_summary(manifest)

    VALID_MATCH_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(VALID_MATCH_MANIFEST) + ".tmp"
    )

    manifest.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        VALID_MATCH_MANIFEST
    )

    log("")
    log(
        f"[SAVED] Valid-match manifest written to: "
        f"{VALID_MATCH_MANIFEST}"
    )

    log("")
    log(
        "[PASSED] VALID-MATCH MANIFEST COMPLETE"
    )


if __name__ == "__main__":
    main()
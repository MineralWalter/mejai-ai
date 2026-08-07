from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESEARCH_FILE = Path(
    "data/analysis/mejai_research_dataset.parquet"
)

RAW_SNAPSHOT_FILES = {
    "sea": Path("data/raw/sea/snapshots.parquet"),
    "asia": Path("data/raw/asia/snapshots.parquet"),
    "europe": Path("data/raw/europe/snapshots.parquet"),
    "americas": Path("data/raw/americas/snapshots.parquet"),
}


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message)


# ============================================================
# LOAD RESEARCH DATASET
# ============================================================

def load_research_dataset():

    if not RESEARCH_FILE.exists():
        log(
            f"[ERROR] Research dataset not found: "
            f"{RESEARCH_FILE}"
        )
        return pd.DataFrame()

    try:
        return pd.read_parquet(
            RESEARCH_FILE,
            engine="pyarrow",
        )

    except Exception as error:
        log(
            f"[ERROR] Could not read research dataset: "
            f"{error}"
        )
        return pd.DataFrame()


# ============================================================
# SNAPSHOT ALIGNMENT
# ============================================================

def validate_snapshot_alignment(df):

    log("")
    log("========== SNAPSHOT / PURCHASE ALIGNMENT ==========")

    required = [
        "case_id",
        "match_id",
        "participant_id",
        "region",
        "purchase_timestamp",
        "snapshot_timestamp",
        "snapshot_age_ms",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        log("[ERROR] Required columns missing:")

        for column in missing:
            log(f"  - {column}")

        return

    # --------------------------------------------------------
    # RECOMPUTE SNAPSHOT AGE
    # --------------------------------------------------------

    calculated_age = (
        df["purchase_timestamp"]
        - df["snapshot_timestamp"]
    )

    age_mismatch = (
        calculated_age
        != df["snapshot_age_ms"]
    ).sum()

    log(
        f"Snapshot-age calculation mismatches: "
        f"{age_mismatch:,}"
    )

    if age_mismatch == 0:
        log(
            "[PASSED] snapshot_age_ms matches "
            "purchase_timestamp - snapshot_timestamp"
        )

    # --------------------------------------------------------
    # BASIC AGE DISTRIBUTION
    # --------------------------------------------------------

    age_seconds = (
        df["snapshot_age_ms"] / 1000
    )

    log("")
    log("Snapshot age in seconds:")

    log(
        age_seconds.describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    # --------------------------------------------------------
    # EXACT PURCHASE SNAPSHOTS
    # --------------------------------------------------------

    exact = (
        df["snapshot_timestamp"]
        == df["purchase_timestamp"]
    ).sum()

    log("")
    log(
        f"Snapshot exactly at purchase: "
        f"{exact:,}"
    )

    # --------------------------------------------------------
    # AGE BUCKETS
    # --------------------------------------------------------

    buckets = pd.cut(
        age_seconds,
        bins=[
            -0.001,
            1,
            5,
            10,
            20,
            30,
            40,
            50,
            60,
            float("inf"),
        ],
        right=True,
    )

    log("")
    log("Snapshot age distribution:")

    log(
        buckets
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # PURCHASE / SNAPSHOT ORDER
    # --------------------------------------------------------

    future = (
        df["snapshot_timestamp"]
        > df["purchase_timestamp"]
    ).sum()

    log("")
    log(
        f"Snapshots after purchase: "
        f"{future:,}"
    )

    if future == 0:
        log(
            "[PASSED] No research snapshot occurs "
            "after purchase"
        )


# ============================================================
# CASE UNIQUENESS
# ============================================================

def validate_case_snapshot_uniqueness(df):

    log("")
    log("========== CASE SNAPSHOT UNIQUENESS ==========")

    required = [
        "case_id",
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "snapshot_timestamp",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        log(
            "[SKIPPED] Required columns missing"
        )
        return

    case_counts = (
        df.groupby("case_id")
        .agg(
            rows=("case_id", "size"),
            matches=("match_id", "nunique"),
            participants=("participant_id", "nunique"),
            purchases=("purchase_timestamp", "nunique"),
            snapshots=("snapshot_timestamp", "nunique"),
        )
    )

    multiple_rows = (
        case_counts["rows"] != 1
    ).sum()

    multiple_matches = (
        case_counts["matches"] != 1
    ).sum()

    multiple_participants = (
        case_counts["participants"] != 1
    ).sum()

    multiple_purchases = (
        case_counts["purchases"] != 1
    ).sum()

    multiple_snapshots = (
        case_counts["snapshots"] != 1
    ).sum()

    log(
        f"Cases with multiple rows: "
        f"{multiple_rows:,}"
    )

    log(
        f"Cases with multiple matches: "
        f"{multiple_matches:,}"
    )

    log(
        f"Cases with multiple participants: "
        f"{multiple_participants:,}"
    )

    log(
        f"Cases with multiple purchase timestamps: "
        f"{multiple_purchases:,}"
    )

    log(
        f"Cases with multiple snapshot timestamps: "
        f"{multiple_snapshots:,}"
    )

    if (
        multiple_rows == 0
        and multiple_matches == 0
        and multiple_participants == 0
        and multiple_purchases == 0
        and multiple_snapshots == 0
    ):
        log(
            "[PASSED] Every case maps to exactly "
            "one participant, purchase, and snapshot"
        )


# ============================================================
# REGION ALIGNMENT
# ============================================================

def validate_region_alignment(df):

    log("")
    log("========== REGION SNAPSHOT ALIGNMENT ==========")

    if not {
        "region",
        "snapshot_timestamp",
        "purchase_timestamp",
    }.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )
        return

    summary = (
        df.assign(
            snapshot_age_seconds=(
                df["purchase_timestamp"]
                - df["snapshot_timestamp"]
            ) / 1000
        )
        .groupby("region")["snapshot_age_seconds"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    log(
        summary.to_string()
    )


# ============================================================
# PLAYER / TEAM STATE SNAPSHOT CHECK
# ============================================================

def validate_state_presence(df):

    log("")
    log("========== SNAPSHOT STATE PRESENCE ==========")

    state_columns = [
        "player_current_gold",
        "player_total_gold",
        "player_level",
        "player_xp",
        "player_minions_killed",
        "player_jungle_minions_killed",
        "team_total_gold_sum",
        "enemy_total_gold_sum",
        "team_total_gold_diff",
        "team_xp_diff",
        "team_cs_diff",
    ]

    available = [
        column
        for column in state_columns
        if column in df.columns
    ]

    if not available:
        log(
            "[SKIPPED] No state columns available"
        )
        return

    log(
        "State columns checked:"
    )

    for column in available:

        missing = df[column].isna().sum()

        log(
            f"{column}: "
            f"{missing:,} missing"
        )

        if missing == 0:
            continue

        log(
            f"[WARNING] Missing snapshot state "
            f"for {column}"
        )


# ============================================================
# EXTREME ALIGNMENT CASES
# ============================================================

def inspect_extreme_cases(df):

    log("")
    log("========== EXTREME SNAPSHOT AGES ==========")

    if not {
        "case_id",
        "match_id",
        "participant_id",
        "region",
        "purchase_timestamp",
        "snapshot_timestamp",
        "snapshot_age_ms",
        "champion_name",
        "team_position",
        "lifecycle_status",
    }.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )
        return

    columns = [
        "case_id",
        "match_id",
        "participant_id",
        "region",
        "champion_name",
        "team_position",
        "lifecycle_status",
        "purchase_timestamp",
        "snapshot_timestamp",
        "snapshot_age_ms",
    ]

    oldest = (
        df.sort_values(
            "snapshot_age_ms",
            ascending=False,
        )
        .head(20)
    )

    log("")
    log("20 oldest snapshots relative to purchase:")

    log(
        oldest[columns]
        .to_string(index=False)
    )

    newest = (
        df.sort_values(
            "snapshot_age_ms",
            ascending=True,
        )
        .head(20)
    )

    log("")
    log("20 closest snapshots to purchase:")

    log(
        newest[columns]
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("===========================================")
    log("     SNAPSHOT ALIGNMENT VALIDATION")
    log("===========================================")

    df = load_research_dataset()

    if df.empty:
        log("[ERROR] Research dataset is empty")
        return

    log(
        f"Research cases: {len(df):,}"
    )

    validate_snapshot_alignment(df)

    validate_case_snapshot_uniqueness(df)

    validate_region_alignment(df)

    validate_state_presence(df)

    inspect_extreme_cases(df)

    log("")
    log("===========================================")
    log("     SNAPSHOT ALIGNMENT CHECK COMPLETE")
    log("===========================================")


if __name__ == "__main__":
    main()
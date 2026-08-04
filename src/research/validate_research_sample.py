from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/analysis/mejai_research_dataset.parquet"
)


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():
        log(
            f"[ERROR] Research dataset not found: "
            f"{INPUT_FILE}"
        )
        return pd.DataFrame()

    try:
        df = pd.read_parquet(
            INPUT_FILE,
            engine="pyarrow",
        )

    except Exception as error:
        log(
            f"[ERROR] Could not read research dataset: "
            f"{error}"
        )
        return pd.DataFrame()

    return df


# ============================================================
# RESEARCH SAMPLE STRUCTURE
# ============================================================

def validate_sample_structure(df):

    log("")
    log("========== RESEARCH SAMPLE STRUCTURE ==========")

    required = [
        "case_id",
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "snapshot_timestamp",
        "lifecycle_status",
        "outcome_win",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        log("[SKIPPED] Required columns missing:")

        for column in missing:
            log(f"  - {column}")

        return

    # --------------------------------------------------------
    # CASE UNIQUENESS
    # --------------------------------------------------------

    duplicate_cases = (
        df["case_id"]
        .duplicated()
        .sum()
    )

    log(
        f"Duplicate case IDs: "
        f"{duplicate_cases:,}"
    )

    # --------------------------------------------------------
    # MATCH / PARTICIPANT STRUCTURE
    # --------------------------------------------------------

    unique_matches = df["match_id"].nunique()
    unique_participants = df["participant_id"].nunique()

    log(
        f"Unique matches represented: "
        f"{unique_matches:,}"
    )

    log(
        f"Unique participant IDs represented: "
        f"{unique_participants:,}"
    )

    # --------------------------------------------------------
    # CASES PER MATCH
    # --------------------------------------------------------

    cases_per_match = (
        df.groupby("match_id")
        .size()
    )

    log("")
    log("Cases per match:")

    log(
        cases_per_match.describe().to_string()
    )

    log(
        f"Matches represented by exactly one case: "
        f"{(cases_per_match == 1).sum():,}"
    )

    log(
        f"Matches represented by multiple cases: "
        f"{(cases_per_match > 1).sum():,}"
    )

    # --------------------------------------------------------
    # CASES PER PARTICIPANT
    # --------------------------------------------------------

    cases_per_participant = (
        df.groupby("participant_id")
        .size()
    )

    log("")
    log("Cases per participant ID:")

    log(
        cases_per_participant.describe().to_string()
    )


# ============================================================
# PURCHASE TIMING DISTRIBUTION
# ============================================================

def validate_purchase_timing(df):

    log("")
    log("========== PURCHASE TIMING DISTRIBUTION ==========")

    required = [
        "purchase_time_seconds",
        "outcome_game_duration",
        "lifecycle_status",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        log(
            "[SKIPPED] Required purchase timing "
            "columns missing"
        )
        return

    purchase = df["purchase_time_seconds"]
    duration = df["outcome_game_duration"]

    # --------------------------------------------------------
    # PURCHASE BEFORE GAME END
    # --------------------------------------------------------

    after_game_end = (
        purchase > duration
    ).sum()

    log(
        f"Purchases after game end: "
        f"{after_game_end:,}"
    )

    # --------------------------------------------------------
    # PURCHASE FRACTION
    # --------------------------------------------------------

    fraction = (
        purchase / duration
    )

    log("")
    log("Purchase time as fraction of game duration:")

    log(
        fraction.describe(
            percentiles=[
                0.01,
                0.05,
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
    # BY LIFECYCLE
    # --------------------------------------------------------

    log("")
    log("Purchase timing by lifecycle:")

    timing = (
        df.groupby("lifecycle_status")[
            "purchase_time_seconds"
        ]
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
        timing.to_string()
    )


# ============================================================
# OUTCOME BALANCE
# ============================================================

def validate_outcome_balance(df):

    log("")
    log("========== OUTCOME BALANCE ==========")

    if "outcome_win" not in df.columns:
        log("[SKIPPED] outcome_win missing")
        return

    counts = (
        df["outcome_win"]
        .value_counts()
        .sort_index()
    )

    log("Outcome counts:")

    for value, count in counts.items():
        log(
            f"  {value}: {count:,}"
        )

    total = len(df)

    if total > 0:
        wins = int(
            df["outcome_win"].sum()
        )

        losses = total - wins

        log("")
        log(
            f"Win rate: "
            f"{wins / total * 100:.2f}%"
        )

        log(
            f"Loss rate: "
            f"{losses / total * 100:.2f}%"
        )


# ============================================================
# LIFECYCLE BALANCE
# ============================================================

def validate_lifecycle_balance(df):

    log("")
    log("========== LIFECYCLE BALANCE ==========")

    if "lifecycle_status" not in df.columns:
        log("[SKIPPED] lifecycle_status missing")
        return

    counts = (
        df["lifecycle_status"]
        .value_counts()
    )

    log("Lifecycle counts:")

    for status, count in counts.items():
        log(
            f"  {status}: {count:,}"
        )

    log("")
    log("Lifecycle percentages:")

    percentages = (
        counts
        / len(df)
        * 100
    )

    for status, percentage in percentages.items():
        log(
            f"  {status}: "
            f"{percentage:.2f}%"
        )


# ============================================================
# REGION BALANCE
# ============================================================

def validate_region_balance(df):

    log("")
    log("========== REGION BALANCE ==========")

    if "region" not in df.columns:
        log("[SKIPPED] region missing")
        return

    counts = (
        df["region"]
        .value_counts()
        .sort_index()
    )

    log("Region counts:")

    for region, count in counts.items():
        log(
            f"  {region}: {count:,}"
        )

    log("")
    log("Region percentages:")

    percentages = (
        counts
        / len(df)
        * 100
    )

    for region, percentage in percentages.items():
        log(
            f"  {region}: "
            f"{percentage:.2f}%"
        )


# ============================================================
# REGION × OUTCOME
# ============================================================

def validate_region_outcome(df):

    log("")
    log("========== REGION × OUTCOME ==========")

    required = [
        "region",
        "outcome_win",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        log(
            "[SKIPPED] Required region/outcome "
            "columns missing"
        )
        return

    table = pd.crosstab(
        df["region"],
        df["outcome_win"],
    )

    log(
        table.to_string()
    )

    log("")
    log("Win rate by region:")

    win_rate = (
        df.groupby("region")[
            "outcome_win"
        ]
        .mean()
        * 100
    )

    for region, rate in win_rate.items():
        log(
            f"  {region}: "
            f"{rate:.2f}%"
        )


# ============================================================
# REGION × LIFECYCLE
# ============================================================

def validate_region_lifecycle(df):

    log("")
    log("========== REGION × LIFECYCLE ==========")

    required = [
        "region",
        "lifecycle_status",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        log(
            "[SKIPPED] Required region/lifecycle "
            "columns missing"
        )
        return

    table = pd.crosstab(
        df["region"],
        df["lifecycle_status"],
    )

    log(
        table.to_string()
    )

    log("")
    log("Lifecycle percentages within region:")

    percentage_table = pd.crosstab(
        df["region"],
        df["lifecycle_status"],
        normalize="index",
    ) * 100

    log(
        percentage_table.round(2).to_string()
    )


# ============================================================
# POSITION BALANCE
# ============================================================

def validate_position_balance(df):

    log("")
    log("========== TEAM POSITION DISTRIBUTION ==========")

    if "team_position" not in df.columns:
        log("[SKIPPED] team_position missing")
        return

    counts = (
        df["team_position"]
        .value_counts()
    )

    log(
        counts.to_string()
    )

    log("")
    log("Position percentages:")

    percentages = (
        counts
        / len(df)
        * 100
    )

    for position, percentage in percentages.items():
        log(
            f"  {position}: "
            f"{percentage:.2f}%"
        )


# ============================================================
# CHAMPION DISTRIBUTION
# ============================================================

def validate_champion_distribution(df):

    log("")
    log("========== CHAMPION DISTRIBUTION ==========")

    required = [
        "champion_name",
        "team_position",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        log(
            "[SKIPPED] Champion distribution "
            "columns missing"
        )
        return

    champion_counts = (
        df["champion_name"]
        .value_counts()
    )

    log("Top 20 champions:")

    log(
        champion_counts
        .head(20)
        .to_string()
    )

    log("")
    log(
        f"Unique champions: "
        f"{df['champion_name'].nunique():,}"
    )

    log("")
    log("Champion × position:")

    cross = pd.crosstab(
        df["champion_name"],
        df["team_position"],
    )

    log(
        cross.to_string()
    )


# ============================================================
# PURCHASE TIMING × LIFECYCLE
# ============================================================

def validate_lifecycle_timing(df):

    log("")
    log("========== LIFECYCLE × PURCHASE TIMING ==========")

    required = [
        "lifecycle_status",
        "purchase_time_seconds",
        "outcome_game_duration",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        log(
            "[SKIPPED] Required lifecycle timing "
            "columns missing"
        )
        return

    temp = df[
        [
            "lifecycle_status",
            "purchase_time_seconds",
            "outcome_game_duration",
        ]
    ].copy()

    temp["purchase_fraction"] = (
        temp["purchase_time_seconds"]
        / temp["outcome_game_duration"]
    )

    summary = (
        temp.groupby("lifecycle_status")[
            [
                "purchase_time_seconds",
                "purchase_fraction",
            ]
        ]
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
# FINAL SUMMARY
# ============================================================

def final_summary(df):

    log("")
    log("========== RESEARCH SAMPLE SUMMARY ==========")

    log(
        f"Research cases: "
        f"{len(df):,}"
    )

    log(
        f"Unique matches: "
        f"{df['match_id'].nunique():,}"
        if "match_id" in df.columns
        else "Unique matches: unavailable"
    )

    log(
        f"Unique champions: "
        f"{df['champion_name'].nunique():,}"
        if "champion_name" in df.columns
        else "Unique champions: unavailable"
    )

    log(
        f"Regions: "
        f"{df['region'].nunique():,}"
        if "region" in df.columns
        else "Regions: unavailable"
    )

    log(
        "[INFO] This script describes the "
        "research sample and checks its composition."
    )

    log(
        "[INFO] It does not determine whether "
        "Mejai causes winning or losing."
    )

    log(
        "[INFO] It does not evaluate predictive "
        "model performance."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("===========================================")
    log("       RESEARCH SAMPLE VALIDATION")
    log("===========================================")

    df = load_dataset()

    if df.empty:
        log("[ERROR] Dataset is empty")
        return

    log(
        f"Research cases: "
        f"{len(df):,}"
    )

    log(
        f"Columns: "
        f"{len(df.columns):,}"
    )

    validate_sample_structure(df)
    validate_purchase_timing(df)
    validate_outcome_balance(df)
    validate_lifecycle_balance(df)
    validate_region_balance(df)
    validate_region_outcome(df)
    validate_region_lifecycle(df)
    validate_position_balance(df)
    validate_champion_distribution(df)
    validate_lifecycle_timing(df)

    final_summary(df)

    log("")
    log("===========================================")
    log("       RESEARCH SAMPLE CHECK COMPLETE")
    log("===========================================")


if __name__ == "__main__":
    main()
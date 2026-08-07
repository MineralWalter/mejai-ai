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
# PURCHASE TIME VS GAME DURATION
# ============================================================

def validate_purchase_vs_duration(df):

    log("")
    log("========== PURCHASE VS GAME DURATION ==========")

    required = {
        "purchase_time_seconds",
        "outcome_game_duration",
    }

    if not required.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )

        return

    purchase = df["purchase_time_seconds"]
    duration = df["outcome_game_duration"]

    # Purchase occurring after game end
    after_end = (
        purchase > duration
    )

    log(
        f"Purchases after game end: "
        f"{after_end.sum():,}"
    )

    if after_end.any():

        log(
            "[WARNING] Some purchases occur "
            "after the recorded game duration"
        )

        examples = df.loc[
            after_end,
            [
                "case_id",
                "purchase_time_seconds",
                "outcome_game_duration",
                "lifecycle_status",
            ],
        ].head(10)

        log("")
        log("Examples:")
        log(examples.to_string(index=False))

    else:

        log(
            "[PASSED] All purchases occur "
            "before game end"
        )

    # Purchase / game-duration ratio
    ratio = (
        purchase / duration
    )

    log("")
    log("Purchase time as fraction of game duration:")

    log(
        ratio.describe(
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


# ============================================================
# LIFECYCLE LOGIC
# ============================================================

def validate_lifecycle_logic(df):

    log("")
    log("========== LIFECYCLE LOGIC ==========")

    required = {
        "lifecycle_status",
        "purchase_time_seconds",
        "outcome_game_duration",
    }

    if not required.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )

        return

    statuses = (
        df["lifecycle_status"]
        .value_counts()
    )

    log("Lifecycle counts:")

    for status, count in statuses.items():

        log(
            f"{status}: "
            f"{count:,}"
        )

    log("")

    # Purchase timing by lifecycle
    summary = (
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
        "Purchase timing by lifecycle:"
    )

    log(
        summary.to_string()
    )

    # Game duration by lifecycle
    log("")

    duration_summary = (
        df.groupby("lifecycle_status")[
            "outcome_game_duration"
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
        "Game duration by lifecycle:"
    )

    log(
        duration_summary.to_string()
    )


# ============================================================
# OUTCOME CONSISTENCY
# ============================================================

def validate_outcome_consistency(df):

    log("")
    log("========== OUTCOME CONSISTENCY ==========")

    required = {
        "outcome_win",
        "outcome_game_result",
    }

    if not required.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )

        return

    log("Outcome values:")

    log(
        df["outcome_game_result"]
        .value_counts(dropna=False)
        .to_string()
    )

    log("")

    log("Outcome win values:")

    log(
        df["outcome_win"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Display cross-tab rather than assuming
    # exact string representation of game results.
    table = pd.crosstab(
        df["outcome_game_result"],
        df["outcome_win"],
        margins=True,
    )

    log("")
    log("Game result vs outcome_win:")

    log(
        table.to_string()
    )


# ============================================================
# PLAYER STATE SANITY
# ============================================================

def validate_player_state(df):

    log("")
    log("========== PLAYER STATE SANITY ==========")

    columns = [
        "player_current_gold",
        "player_total_gold",
        "player_level",
        "player_xp",
        "player_minions_killed",
        "player_jungle_minions_killed",
    ]

    for column in columns:

        if column not in df.columns:

            log(
                f"[SKIPPED] {column} missing"
            )

            continue

        negative = (
            df[column] < 0
        ).sum()

        log(
            f"{column}: "
            f"negative values = "
            f"{negative:,}"
        )

    # Current gold should not exceed total gold.
    if {
        "player_current_gold",
        "player_total_gold",
    }.issubset(df.columns):

        invalid_gold = (
            df["player_current_gold"]
            > df["player_total_gold"]
        ).sum()

        log(
            f"Current gold > total gold: "
            f"{invalid_gold:,}"
        )

        if invalid_gold == 0:

            log(
                "[PASSED] Current gold never "
                "exceeds total gold"
            )

        else:

            log(
                "[WARNING] Impossible gold relationship detected"
            )

    # Basic level bounds
    if "player_level" in df.columns:

        invalid_levels = (
            (df["player_level"] < 1)
            | (df["player_level"] > 18)
        ).sum()

        log(
            f"Player levels outside 1-18: "
            f"{invalid_levels:,}"
        )


# ============================================================
# TEAM STATE SANITY
# ============================================================

def validate_team_state(df):

    log("")
    log("========== TEAM STATE SANITY ==========")

    required = [
        "team_total_gold_sum",
        "enemy_total_gold_sum",
        "team_current_gold_sum",
        "enemy_current_gold_sum",
        "team_total_gold_diff",
        "team_current_gold_diff",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        log(
            "[SKIPPED] Missing columns:"
        )

        for column in missing:

            log(
                f"  - {column}"
            )

        return

    # Total gold cannot be negative.
    negative_team_gold = (
        df["team_total_gold_sum"] < 0
    ).sum()

    negative_enemy_gold = (
        df["enemy_total_gold_sum"] < 0
    ).sum()

    log(
        f"Negative team total gold: "
        f"{negative_team_gold:,}"
    )

    log(
        f"Negative enemy total gold: "
        f"{negative_enemy_gold:,}"
    )

    # Recalculate differences.
    calculated_total = (
        df["team_total_gold_sum"]
        - df["enemy_total_gold_sum"]
    )

    calculated_current = (
        df["team_current_gold_sum"]
        - df["enemy_current_gold_sum"]
    )

    total_mismatch = (
        calculated_total
        != df["team_total_gold_diff"]
    ).sum()

    current_mismatch = (
        calculated_current
        != df["team_current_gold_diff"]
    ).sum()

    log(
        f"Total-gold difference mismatches: "
        f"{total_mismatch:,}"
    )

    log(
        f"Current-gold difference mismatches: "
        f"{current_mismatch:,}"
    )

    if (
        total_mismatch == 0
        and current_mismatch == 0
    ):

        log(
            "[PASSED] Team gold differences "
            "are internally consistent"
        )

    else:

        log(
            "[FAILED] Team gold differences "
            "are inconsistent"
        )


# ============================================================
# REGION × LIFECYCLE
# ============================================================

def validate_region_lifecycle(df):

    log("")
    log("========== REGION × LIFECYCLE ==========")

    required = {
        "region",
        "lifecycle_status",
    }

    if not required.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )

        return

    table = pd.crosstab(
        df["region"],
        df["lifecycle_status"],
        margins=True,
    )

    log(
        table.to_string()
    )

    log("")
    log("Lifecycle percentages within region:")

    percentages = pd.crosstab(
        df["region"],
        df["lifecycle_status"],
        normalize="index",
    ) * 100

    log(
        percentages.round(2).to_string()
    )


# ============================================================
# CHAMPION / POSITION DISTRIBUTION
# ============================================================

def validate_champion_position(df):

    log("")
    log("========== CHAMPION / POSITION DISTRIBUTION ==========")

    required = {
        "champion_name",
        "team_position",
    }

    if not required.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )

        return

    log("Champion frequency:")

    log(
        df["champion_name"]
        .value_counts()
        .head(20)
        .to_string()
    )

    log("")
    log("Team position frequency:")

    log(
        df["team_position"]
        .value_counts(dropna=False)
        .to_string()
    )

    log("")
    log("Champion × position:")

    table = pd.crosstab(
        df["champion_name"],
        df["team_position"],
    )

    log(
        table.to_string()
    )


# ============================================================
# CASE REPRESENTATION
# ============================================================

def validate_case_representation(df):

    log("")
    log("========== CASE REPRESENTATION ==========")

    required = {
        "case_id",
        "match_id",
        "participant_id",
        "purchase_timestamp",
    }

    if not required.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )

        return

    # One case should correspond to one participant
    # and one purchase event.
    case_participants = (
        df.groupby("case_id")[
            "participant_id"
        ].nunique()
    )

    case_purchases = (
        df.groupby("case_id")[
            "purchase_timestamp"
        ].nunique()
    )

    participant_problems = (
        case_participants != 1
    ).sum()

    purchase_problems = (
        case_purchases != 1
    ).sum()

    log(
        f"Cases with multiple participants: "
        f"{participant_problems:,}"
    )

    log(
        f"Cases with multiple purchase timestamps: "
        f"{purchase_problems:,}"
    )

    if (
        participant_problems == 0
        and purchase_problems == 0
    ):

        log(
            "[PASSED] Each case represents "
            "one participant and one purchase event"
        )

    else:

        log(
            "[WARNING] Case representation "
            "needs investigation"
        )


# ============================================================
# PURCHASE TIME DUPLICATION
# ============================================================

def validate_purchase_clustering(df):

    log("")
    log("========== PURCHASE TIME CLUSTERING ==========")

    required = {
        "match_id",
        "purchase_timestamp",
        "participant_id",
    }

    if not required.issubset(df.columns):

        log(
            "[SKIPPED] Required columns missing"
        )

        return

    # Multiple participants can buy Mejai around
    # the same time, so this is descriptive rather
    # than an error check.
    purchase_counts = (
        df.groupby(
            [
                "match_id",
                "purchase_timestamp",
            ]
        )
        .size()
    )

    log(
        "Cases sharing the same match/timestamp:"
    )

    log(
        purchase_counts.describe().to_string()
    )

    log(
        f"Unique match/timestamp combinations: "
        f"{len(purchase_counts):,}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def main():

    log("===========================================")
    log("       RESEARCH DATASET QUALITY CHECK")
    log("===========================================")

    df = load_dataset()

    if df.empty:

        log(
            "[ERROR] Dataset is empty"
        )

        return

    log(
        f"Research cases: {len(df):,}"
    )

    log(
        f"Columns: {len(df.columns):,}"
    )

    validate_purchase_vs_duration(df)
    validate_lifecycle_logic(df)
    validate_outcome_consistency(df)
    validate_player_state(df)
    validate_team_state(df)
    validate_region_lifecycle(df)
    validate_champion_position(df)
    validate_case_representation(df)
    validate_purchase_clustering(df)

    log("")
    log("===========================================")
    log("       RESEARCH QUALITY CHECK COMPLETE")
    log("===========================================")


if __name__ == "__main__":
    main()
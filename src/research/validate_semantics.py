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
# TEAM GOLD SEMANTICS
# ============================================================

def validate_team_gold(df):

    log("")
    log("========== TEAM GOLD SEMANTICS ==========")

    required = [
        "team_current_gold_sum",
        "enemy_current_gold_sum",
        "team_total_gold_sum",
        "enemy_total_gold_sum",
        "team_current_gold_diff",
        "team_total_gold_diff",
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
    # CURRENT GOLD DIFFERENCE
    # --------------------------------------------------------

    calculated_current_diff = (
        df["team_current_gold_sum"]
        - df["enemy_current_gold_sum"]
    )

    current_diff_mismatch = (
        calculated_current_diff
        != df["team_current_gold_diff"]
    ).sum()

    log(
        f"Current-gold difference mismatches: "
        f"{current_diff_mismatch:,}"
    )

    # --------------------------------------------------------
    # TOTAL GOLD DIFFERENCE
    # --------------------------------------------------------

    calculated_total_diff = (
        df["team_total_gold_sum"]
        - df["enemy_total_gold_sum"]
    )

    total_diff_mismatch = (
        calculated_total_diff
        != df["team_total_gold_diff"]
    ).sum()

    log(
        f"Total-gold difference mismatches: "
        f"{total_diff_mismatch:,}"
    )

    # --------------------------------------------------------
    # CURRENT GOLD CANNOT EXCEED TOTAL GOLD
    # --------------------------------------------------------

    team_current_exceeds_total = (
        df["team_current_gold_sum"]
        > df["team_total_gold_sum"]
    ).sum()

    enemy_current_exceeds_total = (
        df["enemy_current_gold_sum"]
        > df["enemy_total_gold_sum"]
    ).sum()

    log(
        f"Team current gold > team total gold: "
        f"{team_current_exceeds_total:,}"
    )

    log(
        f"Enemy current gold > enemy total gold: "
        f"{enemy_current_exceeds_total:,}"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if (
        current_diff_mismatch == 0
        and total_diff_mismatch == 0
        and team_current_exceeds_total == 0
        and enemy_current_exceeds_total == 0
    ):
        log(
            "[PASSED] Team gold relationships are "
            "internally consistent"
        )

    else:
        log(
            "[FAILED] Team gold relationship "
            "problems detected"
        )


# ============================================================
# XP SEMANTICS
# ============================================================

def validate_xp(df):

    log("")
    log("========== XP SEMANTICS ==========")

    required = [
        "team_xp_sum",
        "enemy_xp_sum",
        "team_xp_diff",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        log("[SKIPPED] Required XP columns missing")

        for column in missing:
            log(f"  - {column}")

        return

    # --------------------------------------------------------
    # XP DIFFERENCE
    # --------------------------------------------------------

    calculated_diff = (
        df["team_xp_sum"]
        - df["enemy_xp_sum"]
    )

    mismatch = (
        calculated_diff
        != df["team_xp_diff"]
    ).sum()

    log(
        f"XP difference mismatches: "
        f"{mismatch:,}"
    )

    # --------------------------------------------------------
    # BASIC RANGES
    # --------------------------------------------------------

    negative_team_xp = (
        df["team_xp_sum"] < 0
    ).sum()

    negative_enemy_xp = (
        df["enemy_xp_sum"] < 0
    ).sum()

    log(
        f"Negative team XP totals: "
        f"{negative_team_xp:,}"
    )

    log(
        f"Negative enemy XP totals: "
        f"{negative_enemy_xp:,}"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if (
        mismatch == 0
        and negative_team_xp == 0
        and negative_enemy_xp == 0
    ):
        log(
            "[PASSED] XP features are "
            "internally consistent"
        )

    else:
        log(
            "[FAILED] XP feature consistency "
            "problems detected"
        )


# ============================================================
# CS SEMANTICS
# ============================================================

def validate_cs(df):

    log("")
    log("========== CS SEMANTICS ==========")

    required = [
        "team_minions_killed_sum",
        "enemy_minions_killed_sum",
        "team_jungle_minions_killed_sum",
        "enemy_jungle_minions_killed_sum",
        "team_cs_diff",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        log("[SKIPPED] Required CS columns missing")

        for column in missing:
            log(f"  - {column}")

        return

    # --------------------------------------------------------
    # TEAM CS DIFFERENCE
    #
    # team_cs_diff represents lane CS difference.
    #
    # Jungle CS is tracked separately and is therefore
    # intentionally NOT included in this calculation.
    # --------------------------------------------------------

    calculated_lane_cs_diff = (
        df["team_minions_killed_sum"]
        - df["enemy_minions_killed_sum"]
    )

    lane_cs_mismatch = (
        calculated_lane_cs_diff
        != df["team_cs_diff"]
    ).sum()

    log(
        f"Lane-CS difference mismatches: "
        f"{lane_cs_mismatch:,}"
    )

    # --------------------------------------------------------
    # JUNGLE CS DIFFERENCE
    #
    # This is a separate derived quantity and is not
    # expected to equal team_cs_diff.
    # --------------------------------------------------------

    calculated_jungle_cs_diff = (
        df["team_jungle_minions_killed_sum"]
        - df["enemy_jungle_minions_killed_sum"]
    )

    log(
        "Jungle CS difference is tracked separately "
        "from team_cs_diff"
    )

    # --------------------------------------------------------
    # NON-NEGATIVE CS
    # --------------------------------------------------------

    negative_team_lane = (
        df["team_minions_killed_sum"] < 0
    ).sum()

    negative_enemy_lane = (
        df["enemy_minions_killed_sum"] < 0
    ).sum()

    negative_team_jungle = (
        df["team_jungle_minions_killed_sum"] < 0
    ).sum()

    negative_enemy_jungle = (
        df["enemy_jungle_minions_killed_sum"] < 0
    ).sum()

    log(
        f"Negative team lane CS: "
        f"{negative_team_lane:,}"
    )

    log(
        f"Negative enemy lane CS: "
        f"{negative_enemy_lane:,}"
    )

    log(
        f"Negative team jungle CS: "
        f"{negative_team_jungle:,}"
    )

    log(
        f"Negative enemy jungle CS: "
        f"{negative_enemy_jungle:,}"
    )

    # --------------------------------------------------------
    # TOTAL CS DIFFERENCE
    #
    # Informational only.
    #
    # This deliberately does NOT need to match team_cs_diff.
    # --------------------------------------------------------

    total_cs_diff = (
        calculated_lane_cs_diff
        + calculated_jungle_cs_diff
    )

    total_cs_diff_difference = (
        total_cs_diff
        - df["team_cs_diff"]
    )

    rows_differing_from_total_cs = (
        total_cs_diff_difference != 0
    ).sum()

    log(
        f"Rows where total (lane + jungle) CS differs "
        f"from team_cs_diff: "
        f"{rows_differing_from_total_cs:,}"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if (
        lane_cs_mismatch == 0
        and negative_team_lane == 0
        and negative_enemy_lane == 0
        and negative_team_jungle == 0
        and negative_enemy_jungle == 0
    ):
        log(
            "[PASSED] CS features are "
            "internally consistent"
        )

    else:
        log(
            "[FAILED] CS feature consistency "
            "problems detected"
        )


# ============================================================
# PLAYER / TEAM RELATIONSHIP
# ============================================================

def validate_player_team_relationship(df):

    log("")
    log("========== PLAYER / TEAM RELATIONSHIP ==========")

    required = [
        "player_total_gold",
        "player_current_gold",
        "player_xp",
        "player_minions_killed",
        "player_jungle_minions_killed",
        "team_total_gold_sum",
        "team_current_gold_sum",
        "team_xp_sum",
        "team_minions_killed_sum",
        "team_jungle_minions_killed_sum",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        log(
            "[SKIPPED] Required player/team "
            "columns missing"
        )

        for column in missing:
            log(f"  - {column}")

        return

    # --------------------------------------------------------
    # PLAYER VALUES CANNOT EXCEED TEAM TOTALS
    # --------------------------------------------------------

    player_gold_exceeds_team = (
        df["player_total_gold"]
        > df["team_total_gold_sum"]
    ).sum()

    player_current_exceeds_team = (
        df["player_current_gold"]
        > df["team_current_gold_sum"]
    ).sum()

    player_xp_exceeds_team = (
        df["player_xp"]
        > df["team_xp_sum"]
    ).sum()

    player_lane_cs_exceeds_team = (
        df["player_minions_killed"]
        > df["team_minions_killed_sum"]
    ).sum()

    player_jungle_cs_exceeds_team = (
        df["player_jungle_minions_killed"]
        > df["team_jungle_minions_killed_sum"]
    ).sum()

    log(
        f"Player total gold > team total gold: "
        f"{player_gold_exceeds_team:,}"
    )

    log(
        f"Player current gold > team current gold: "
        f"{player_current_exceeds_team:,}"
    )

    log(
        f"Player XP > team XP: "
        f"{player_xp_exceeds_team:,}"
    )

    log(
        f"Player lane CS > team lane CS: "
        f"{player_lane_cs_exceeds_team:,}"
    )

    log(
        f"Player jungle CS > team jungle CS: "
        f"{player_jungle_cs_exceeds_team:,}"
    )

    if (
        player_gold_exceeds_team == 0
        and player_current_exceeds_team == 0
        and player_xp_exceeds_team == 0
        and player_lane_cs_exceeds_team == 0
        and player_jungle_cs_exceeds_team == 0
    ):
        log(
            "[PASSED] Player-level values do not "
            "exceed corresponding team totals"
        )

    else:
        log(
            "[FAILED] Player-level values exceed "
            "corresponding team totals"
        )


# ============================================================
# TEAM AGGREGATE MAGNITUDES
# ============================================================

def validate_team_aggregate_magnitudes(df):

    log("")
    log("========== TEAM AGGREGATE MAGNITUDES ==========")

    columns = [
        "team_total_gold_sum",
        "enemy_total_gold_sum",
        "team_xp_sum",
        "enemy_xp_sum",
        "team_minions_killed_sum",
        "enemy_minions_killed_sum",
        "team_jungle_minions_killed_sum",
        "enemy_jungle_minions_killed_sum",
    ]

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        log(
            "[SKIPPED] Required aggregate "
            "columns missing"
        )

        for column in missing:
            log(f"  - {column}")

        return

    for column in columns:

        values = df[column]

        log("")
        log(column)

        log(
            values.describe().to_string()
        )


# ============================================================
# OUTCOME NAMESPACE CHECK
# ============================================================

def validate_outcome_namespace(df):

    log("")
    log("========== OUTCOME NAMESPACE CHECK ==========")

    outcome_columns = [
        column
        for column in df.columns
        if column.startswith("outcome_")
    ]

    expected = {
        "outcome_win",
        "outcome_final_gold_earned",
        "outcome_final_gold_spent",
        "outcome_final_champ_level",
        "outcome_final_champ_experience",
        "outcome_final_kills",
        "outcome_final_deaths",
        "outcome_final_assists",
        "outcome_final_damage_dealt_to_champions",
        "outcome_final_damage_taken",
        "outcome_game_duration",
        "outcome_game_result",
    }

    actual = set(outcome_columns)

    unexpected = sorted(
        actual - expected
    )

    missing = sorted(
        expected - actual
    )

    log(
        f"Outcome columns detected: "
        f"{len(outcome_columns)}"
    )

    if unexpected:
        log("[WARNING] Unexpected outcome columns:")

        for column in unexpected:
            log(f"  - {column}")

    if missing:
        log("[WARNING] Expected outcome columns missing:")

        for column in missing:
            log(f"  - {column}")

    if not unexpected and not missing:
        log(
            "[PASSED] Outcome columns match "
            "expected outcome namespace"
        )


# ============================================================
# DERIVED FEATURE NAMESPACE
# ============================================================

def validate_feature_names(df):

    log("")
    log("========== FEATURE NAMESPACE CHECK ==========")

    derived_keywords = [
        "diff",
        "sum",
        "per_second",
        "per_minute",
        "ratio",
        "share",
        "rate",
    ]

    candidates = []

    for column in df.columns:

        lower = column.lower()

        if any(
            keyword in lower
            for keyword in derived_keywords
        ):
            candidates.append(column)

    log("Potentially derived columns:")

    if candidates:

        for column in candidates:
            log(f"  - {column}")

    else:
        log("  None detected")

    log("")
    log(
        "[INFO] These columns are candidates for "
        "derived-feature treatment."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(df):

    log("")
    log("========== FEATURE SEMANTICS SUMMARY ==========")

    log(
        "[INFO] This script validates internal "
        "relationships and feature meaning."
    )

    log(
        "[INFO] It does not determine whether "
        "features are statistically useful."
    )

    log(
        "[INFO] It does not validate model "
        "performance or causal interpretation."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("===========================================")
    log("       FEATURE SEMANTICS VALIDATION")
    log("===========================================")

    df = load_dataset()

    if df.empty:
        log("[ERROR] Dataset is empty")
        return

    log(
        f"Research cases: {len(df):,}"
    )

    log(
        f"Columns: {len(df.columns):,}"
    )

    validate_team_gold(df)
    validate_xp(df)
    validate_cs(df)
    validate_player_team_relationship(df)
    validate_team_aggregate_magnitudes(df)
    validate_outcome_namespace(df)
    validate_feature_names(df)

    final_summary(df)

    log("")
    log("===========================================")
    log("       FEATURE SEMANTICS CHECK COMPLETE")
    log("===========================================")


if __name__ == "__main__":
    main()
